"""
Async HTTP client the public router uses to talk to a single worker's
internal API (`/internal/status`, `/internal/generate`). Centralizes
timeout configuration, connection pooling, request-ID propagation, and
translation of network/HTTP failures into typed errors (see
`src/inference/exception.py`) -- callers never see raw `httpx`
exceptions.
"""

from __future__ import annotations

import httpx

from src.config.schema import RoutingConfig
from src.inference.exception import (
    WorkerConnectionError,
    WorkerInferenceError,
    WorkerRequestRejectedError,
    WorkerUnavailableError,
)
from src.inference.transformers.router.registry import WorkerDescriptor
from src.inference.schema import (
    WorkerGenerateRequestSchema,
    WorkerGenerateResponseSchema,
    WorkerStatusSchema,
)


class WorkerClient:
    def __init__(self, http_client: httpx.AsyncClient, status_timeout_seconds: float) -> None:
        self._http_client = http_client
        self._status_timeout_seconds = status_timeout_seconds

    @classmethod
    def build(cls, routing_config: RoutingConfig) -> "WorkerClient":
        """Build a `WorkerClient` backed by one pooled `httpx.AsyncClient` shared across workers."""

        timeout = httpx.Timeout(
            connect=routing_config.connect_timeout_seconds,
            read=routing_config.read_timeout_seconds,
            write=routing_config.read_timeout_seconds,
            pool=routing_config.connect_timeout_seconds,
        )
        http_client = httpx.AsyncClient(timeout=timeout)
        return cls(http_client, status_timeout_seconds=routing_config.status_poll_timeout_seconds)

    async def generate(
        self,
        worker: WorkerDescriptor,
        payload: WorkerGenerateRequestSchema,
    ) -> WorkerGenerateResponseSchema:
        try:
            response = await self._http_client.post(
                f"{worker.base_url}/internal/generate",
                json=payload.model_dump(),
                headers={"X-Request-ID": payload.request_id},
            )
        except httpx.HTTPError as exc:
            raise WorkerConnectionError(f"Failed to reach worker {worker.worker_id!r}: {exc}") from exc

        return self._parse_generate_response(worker, response)

    async def get_status(self, worker: WorkerDescriptor) -> WorkerStatusSchema:
        try:
            response = await self._http_client.get(
                f"{worker.base_url}/internal/status",
                timeout=self._status_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise WorkerConnectionError(f"Failed to reach worker {worker.worker_id!r}: {exc}") from exc

        if response.status_code != 200:
            raise WorkerConnectionError(
                f"Worker {worker.worker_id!r} status check failed with HTTP {response.status_code}."
            )

        return WorkerStatusSchema.model_validate(response.json())

    async def aclose(self) -> None:
        await self._http_client.aclose()

    def _parse_generate_response(
        self, worker: WorkerDescriptor, response: httpx.Response
    ) -> WorkerGenerateResponseSchema:
        if response.status_code == 200:
            return WorkerGenerateResponseSchema.model_validate(response.json())

        if response.status_code == 503:
            raise WorkerUnavailableError(f"Worker {worker.worker_id!r} is not ready.")

        body = _safe_json(response)

        if 400 <= response.status_code < 500:
            raise WorkerRequestRejectedError(response.status_code, body)

        raise WorkerInferenceError(
            f"Worker {worker.worker_id!r} returned HTTP {response.status_code}: {body}"
        )


def _safe_json(response: httpx.Response) -> dict:
    try:
        return response.json()
    except ValueError:
        return {"error": {"code": "UnknownError", "message": response.text}}
