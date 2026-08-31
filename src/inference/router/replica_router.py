"""
Selects which healthy worker replica a request should be forwarded to.
Independent of Qwen/Transformers -- operates purely over `WorkerRegistry`
state built from `/internal/status` probes.
"""

from __future__ import annotations

import asyncio

from src.inference.exception import NoHealthyWorkerError
from src.inference.router.registry import WorkerRecord, WorkerRegistry


class ReplicaRouter:
    def __init__(self, registry: WorkerRegistry, strategy: str = "round_robin") -> None:
        self._registry = registry
        self._strategy = strategy
        self._round_robin_counter = 0
        # Guards the counter read/increment and the healthy-worker snapshot.
        # The router event loop is single-threaded/cooperative so this
        # section has no `await` and is technically already safe without a
        # lock, but that's a fragile invariant to rely on as this method
        # evolves -- an explicit lock makes correctness independent of it.
        self._lock = asyncio.Lock()

    async def select_worker(self) -> WorkerRecord:
        async with self._lock:
            healthy = self._registry.healthy_workers()
            if not healthy:
                raise NoHealthyWorkerError("No healthy workers are available.")

            if self._strategy == "least_loaded":
                return self._select_least_loaded(healthy)

            worker = healthy[self._round_robin_counter % len(healthy)]
            self._round_robin_counter += 1
            return worker

    @staticmethod
    def _select_least_loaded(healthy: list[WorkerRecord]) -> WorkerRecord:
        # Deterministic tie-break on worker_id so equal-load selection is
        # reproducible rather than dependent on dict/list ordering.
        return min(healthy, key=lambda record: (record.active_requests, record.descriptor.worker_id))
