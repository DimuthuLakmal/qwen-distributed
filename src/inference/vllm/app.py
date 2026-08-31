"""
vLLM (single-process) FastAPI application factory.

Used by the vLLM backend topology: `Public API -> InferenceService(VLLMEngine)
-> vLLM internal DP`. No replica router, no worker HTTP hop, no per-GPU
application-managed processes -- vLLM owns multi-GPU replica creation and
request balancing internally (see `src/inference/engines/vllm.py`).

`create_vllm_app` takes an `engine_factory` callable rather than a
concrete engine so the same app wiring works for both production (a factory
that builds the real engine via `src.inference.engines.factory.build_engine`)
and tests (a factory that returns `FakeInferenceEngine()`).

Model loading happens exactly once, inside `lifespan`, when the app starts
-- never per request and never at import time.
"""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.engines.base import InferenceEngine
from src.inference.middleware.error_handler import register_error_handlers
from src.inference.middleware.logging import AccessLogMiddleware
from src.inference.middleware.request_id import RequestIDMiddleware
from src.inference.service import InferenceService
from src.inference.vllm.routes.health import router as health_router
from src.inference.vllm.routes.inference import router as inference_router
from src.tracking.inference_metrics import InferenceMetricsRecorder
from src.utils.logger import logger


def create_vllm_app(
    engine_factory: Callable[[], InferenceEngine],
    limits: InferenceLimitsConfig,
    defaults: GenerationDefaultsConfig,
    metrics: InferenceMetricsRecorder | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading inference engine (vLLM app)...")
        engine = engine_factory()
        service = InferenceService(engine=engine, limits=limits, defaults=defaults, metrics=metrics)
        app.state.inference_service = service
        if metrics is not None:
            metrics.start(is_active=lambda: service.active_request_count > 0)
        logger.info("Inference engine ready.")

        yield

        logger.info("Shutting down inference service (vLLM app).")
        # VLLMEngine.close() is async (it awaits stop_profile() before
        # shutdown); TransformersQwenVLEngine.close() stays sync -- support
        # both without the app layer needing to know which engine this is.
        close = getattr(engine, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        if metrics is not None:
            await metrics.finish()
        app.state.inference_service = None

    app = FastAPI(title="Qwen-VL vLLM Inference Service", lifespan=lifespan)

    # Registration order matters: Starlette makes the *last*-added
    # middleware the outermost one. RequestIDMiddleware must run first (be
    # outermost) so request.state.request_id exists by the time
    # AccessLogMiddleware and the route handlers run.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(inference_router)

    return app