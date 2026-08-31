"""
InferenceService: coordinates validation/preprocessing, engine execution,
and postprocessing. Contains no model/runtime-specific code -- it only
depends on the `InferenceEngine` interface, so it can be fully exercised in
tests with `FakeInferenceEngine` (see `src/inference/engines/fake.py`).
"""

from __future__ import annotations

import asyncio
import time

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.engines.base import InferenceEngine
from src.inference.exception import (
    EngineNotReadyError,
    InferenceExecutionError,
    InvalidGenerationOptionsError,
    InvalidRequestError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from src.inference.postprocessing import postprocess_response
from src.inference.preprocessing import RawImage, validate_and_build_request
from src.inference.schema import InferenceResponse
from src.tracking.inference_metrics import InferenceMetricsRecorder
from src.utils.logger import logger

# Re-exported so existing call sites (`from src.inference.service import
# EngineNotReadyError, ...`) keep working -- the classes themselves are
# defined once in `src.inference.exception` (see that module's docstring).
__all__ = [
    "InferenceService",
    "InvalidRequestError",
    "UnsupportedMediaTypeError",
    "PayloadTooLargeError",
    "InvalidGenerationOptionsError",
    "EngineNotReadyError",
    "InferenceExecutionError",
]


class InferenceService:
    def __init__(
        self,
        engine: InferenceEngine,
        limits: InferenceLimitsConfig,
        defaults: GenerationDefaultsConfig,
        worker_id: str | None = None,
        metrics: InferenceMetricsRecorder | None = None,
    ) -> None:
        self.engine = engine
        self.limits = limits
        self.defaults = defaults
        # Logging-only identity; the service itself stays HTTP/worker-agnostic.
        self.worker_id = worker_id
        # No-ops throughout when tracking is disabled/unset -- see
        # InferenceMetricsRecorder's docstring.
        self.metrics = metrics
        self._active_requests = 0
        # Guards only the counter below, never the `await self.engine.generate(...)`
        # call itself -- multiple requests must be able to be in-flight in the
        # engine concurrently (required for vLLM's internal continuous batching;
        # see claude.md's Concurrency section). An asyncio.Lock is correct here
        # since `generate()` now runs on the event loop, not in a worker thread.
        self._active_requests_lock = asyncio.Lock()

    async def is_ready(self) -> bool:
        return await self.engine.health()

    @property
    def active_request_count(self) -> int:
        return self._active_requests

    async def generate(
        self,
        request_id: str,
        prompt: str | None,
        system_prompt: str | None = None,
        raw_images: list[RawImage] | None = None,
        generation_overrides: dict | None = None,
    ) -> InferenceResponse:
        """
        Validate a raw request, run it through the engine, and postprocess
        the result. Raises `InvalidRequestError`/`UnsupportedMediaTypeError`/
        `PayloadTooLargeError`/`InvalidGenerationOptionsError` for expected
        input problems, `EngineNotReadyError` if the engine isn't ready yet,
        and `InferenceExecutionError` for unexpected engine failures.
        """

        # Fail fast if the engine isn't ready yet.
        if not await self.is_ready():
            raise EngineNotReadyError("Inference engine is not ready yet.")

        request = validate_and_build_request(
            prompt=prompt,
            system_prompt=system_prompt,
            raw_images=raw_images or [],
            generation_overrides=generation_overrides,
            limits=self.limits,
            defaults=self.defaults,
            request_id=request_id,
        )

        start_time = time.perf_counter()

        async with self._active_requests_lock:
            self._active_requests += 1

        try:
            response = await self.engine.generate(request)
        except Exception as exc:
            logger.error(
                "Inference request failed | request_id=%s | worker_id=%s | success=False | "
                "latency_seconds=%.3f | error_type=%s",
                request.request_id,
                self.worker_id,
                time.perf_counter() - start_time,
                type(exc).__name__,
            )
            raise InferenceExecutionError("Inference engine failed to generate a response.") from exc
        finally:
            async with self._active_requests_lock:
                self._active_requests -= 1

        response = postprocess_response(response)

        tokens_per_second = response.output_token_count / max(response.latency_seconds, 1e-8)

        logger.info(
            "Inference request completed | request_id=%s | worker_id=%s | model_version=%s | "
            "success=True | latency_seconds=%.3f | input_tokens=%d | output_tokens=%d | "
            "tokens_per_second=%.1f | active_requests=%d",
            response.request_id,
            self.worker_id,
            response.model_version,
            response.latency_seconds,
            response.input_token_count,
            response.output_token_count,
            tokens_per_second,
            self.active_request_count,
        )

        if self.metrics is not None:
            self.metrics.record_request(
                input_tokens=response.input_token_count,
                output_tokens=response.output_token_count,
                latency_seconds=response.latency_seconds,
            )

        return response
