"""
InferenceService: coordinates validation/preprocessing, engine execution,
and postprocessing. Contains no model/runtime-specific code -- it only
depends on the `InferenceEngine` interface, so it can be fully exercised in
tests with `FakeInferenceEngine` (see `src/inference/engines/fake.py`).
"""

from __future__ import annotations

import time

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.engines.base import InferenceEngine
from src.inference.postprocessing import postprocess_response
from src.inference.preprocessing import (
    InvalidGenerationOptionsError,
    InvalidRequestError,
    PayloadTooLargeError,
    RawImage,
    UnsupportedMediaTypeError,
    validate_and_build_request,
)
from src.inference.schema import InferenceResponse
from src.utils.logger import logger

__all__ = [
    "InferenceService",
    "InvalidRequestError",
    "UnsupportedMediaTypeError",
    "PayloadTooLargeError",
    "InvalidGenerationOptionsError",
    "EngineNotReadyError",
    "InferenceExecutionError",
]


class EngineNotReadyError(Exception):
    """The inference engine has not finished loading / is not ready."""


class InferenceExecutionError(Exception):
    """The engine raised an unexpected error while generating a response."""


class InferenceService:
    def __init__(
        self,
        engine: InferenceEngine,
        limits: InferenceLimitsConfig,
        defaults: GenerationDefaultsConfig,
    ) -> None:
        self.engine = engine
        self.limits = limits
        self.defaults = defaults

    def is_ready(self) -> bool:
        return self.engine.health()

    def generate(
        self,
        *,
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

        if not self.is_ready():
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

        try:
            response = self.engine.generate(request)
        except Exception as exc:
            logger.error(
                "Inference request failed | request_id=%s | success=False | "
                "latency_seconds=%.3f | error_type=%s",
                request.request_id,
                time.perf_counter() - start_time,
                type(exc).__name__,
            )
            raise InferenceExecutionError(
                "Inference engine failed to generate a response."
            ) from exc

        response = postprocess_response(response)

        logger.info(
            "Inference request completed | request_id=%s | model_version=%s | "
            "success=True | latency_seconds=%.3f | input_tokens=%d | output_tokens=%d",
            response.request_id,
            response.model_version,
            response.latency_seconds,
            response.input_token_count,
            response.output_token_count,
        )

        return response
