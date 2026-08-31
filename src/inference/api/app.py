"""
Worker FastAPI application factory.

Serves only the internal, GPU-worker-facing API (`/internal/*`) consumed by
the public router (`src/inference/router/`) -- never traffic directly from
external clients. `create_worker_app` takes an `engine_factory` callable
rather than a concrete engine so the same app wiring works for both
production (a factory that loads the real Transformers engine) and tests (a
factory that returns `FakeInferenceEngine()`), without ever touching
Transformers in tests.

Model loading happens exactly once, inside `lifespan`, when the app starts
-- never per request and never at import time.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.api.routes.internal_generate import router as internal_generate_router
from src.inference.api.routes.internal_health import router as internal_health_router
from src.inference.api.routes.internal_status import router as internal_status_router
from src.inference.engines.base import InferenceEngine
from src.inference.middleware.error_handler import register_error_handlers
from src.inference.middleware.logging import AccessLogMiddleware
from src.inference.middleware.request_id import RequestIDMiddleware
from src.inference.service import InferenceService
from src.utils.logger import logger


def create_worker_app(
    worker_id: str,
    engine_factory: Callable[[], InferenceEngine],
    limits: InferenceLimitsConfig,
    defaults: GenerationDefaultsConfig,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading inference engine | worker_id=%s", worker_id)
        engine = engine_factory()
        app.state.worker_id = worker_id
        app.state.inference_service = InferenceService(
            engine=engine, limits=limits, defaults=defaults, worker_id=worker_id
        )
        logger.info("Inference engine ready | worker_id=%s", worker_id)

        yield

        logger.info("Shutting down inference service | worker_id=%s", worker_id)
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        app.state.inference_service = None

    app = FastAPI(title=f"Qwen-VL Inference Worker ({worker_id})", lifespan=lifespan)

    # Registration order matters: Starlette makes the *last*-added
    # middleware the outermost one. RequestIDMiddleware must run first (be
    # outermost) so request.state.request_id exists by the time
    # AccessLogMiddleware and the route handlers run.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    register_error_handlers(app)

    app.include_router(internal_health_router)
    app.include_router(internal_status_router)
    app.include_router(internal_generate_router)

    return app
