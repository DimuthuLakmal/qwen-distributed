"""
RouterService: coordinates worker selection, forwarding, failure bookkeeping,
and router-side observability. Contains no HTTP/framework code of its own --
`src/inference/transformers/router/routes/inference.py` is the only caller.
"""

from __future__ import annotations

import asyncio
import time

from src.config.schema import InferenceLimitsConfig
from src.inference.exception import WorkerConnectionError, WorkerUnavailableError
from src.inference.transformers.router.registry import WorkerRegistry
from src.inference.transformers.router.replica_router import ReplicaRouter
from src.inference.transformers.router.worker_client import WorkerClient
from src.inference.schema import WorkerGenerateRequestSchema, WorkerGenerateResponseSchema
from src.utils.logger import logger


class RouterService:
    def __init__(
        self,
        registry: WorkerRegistry,
        worker_client: WorkerClient,
        replica_router: ReplicaRouter,
        limits: InferenceLimitsConfig,
    ) -> None:
        self.limits = limits
        self._registry = registry
        self._worker_client = worker_client
        self._replica_router = replica_router

    @property
    def worker_client(self) -> WorkerClient:
        return self._worker_client

    async def refresh_worker_status(self) -> None:
        """
        Probe every configured worker's `/internal/status` concurrently and
        update the registry. Called at router startup and from
        `GET /health/ready` -- Stage 1B intentionally has no background
        poller (see design notes); this on-demand refresh is the mechanism
        by which a previously-unavailable worker is allowed to recover.
        """

        async def _probe(worker) -> None:
            try:
                status = await self._worker_client.get_status(worker.descriptor)
            except WorkerConnectionError:
                self._registry.mark_unavailable(worker.descriptor.worker_id)
                return

            if status.ready:
                self._registry.mark_ready(worker.descriptor.worker_id, status.active_requests)
            else:
                self._registry.mark_unavailable(worker.descriptor.worker_id)

        await asyncio.gather(*(_probe(worker) for worker in self._registry.all_workers()))

    async def is_ready(self) -> bool:
        await self.refresh_worker_status()
        return bool(self._registry.healthy_workers())

    async def generate(
        self, worker_request: WorkerGenerateRequestSchema
    ) -> WorkerGenerateResponseSchema:
        worker = await self._replica_router.select_worker()

        start_time = time.perf_counter()
        try:
            response = await self._worker_client.generate(worker.descriptor, worker_request)
        except (WorkerConnectionError, WorkerUnavailableError) as exc:
            self._registry.mark_unavailable(worker.descriptor.worker_id)
            logger.warning(
                "Router request failed | request_id=%s | worker_id=%s | success=False | "
                "router_latency_seconds=%.3f | error_type=%s",
                worker_request.request_id,
                worker.descriptor.worker_id,
                time.perf_counter() - start_time,
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Router request failed | request_id=%s | worker_id=%s | success=False | "
                "router_latency_seconds=%.3f | error_type=%s",
                worker_request.request_id,
                worker.descriptor.worker_id,
                time.perf_counter() - start_time,
                type(exc).__name__,
            )
            raise

        logger.info(
            "Router request completed | request_id=%s | worker_id=%s | success=True | "
            "router_latency_seconds=%.3f",
            worker_request.request_id,
            worker.descriptor.worker_id,
            time.perf_counter() - start_time,
        )

        return response
