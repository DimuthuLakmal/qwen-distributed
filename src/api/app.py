"""
FastAPI application factory.

`create_app` takes an `engine_factory` callable rather than a concrete
engine so the same app wiring works for both production (a factory that
loads the real Transformers engine) and tests (a factory that returns
`FakeInferenceEngine()`), without ever touching Transformers in tests.

Model loading happens exactly once, inside `lifespan`, when the app starts
-- never per request and never at import time.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from src.api.middleware.error_handler import register_error_handlers
from src.api.middleware.logging import AccessLogMiddleware
from src.api.middleware.request_id import RequestIDMiddleware
from src.api.routes.health import router as health_router
from src.api.routes.inference import router as inference_router
from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.engines.base import InferenceEngine
from src.inference.service import InferenceService
from src.utils.logger import logger


def create_app(
    engine_factory: Callable[[], InferenceEngine],
    limits: InferenceLimitsConfig,
    defaults: GenerationDefaultsConfig,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading inference engine...")
        engine = engine_factory()
        app.state.inference_service = InferenceService(
            engine=engine, limits=limits, defaults=defaults
        )
        logger.info("Inference engine ready.")

        yield

        logger.info("Shutting down inference service.")
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        app.state.inference_service = None

    app = FastAPI(title="Qwen-VL Inference Service", lifespan=lifespan)

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
