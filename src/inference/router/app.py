"""
Public router FastAPI application factory.

This is the only process external clients talk to. It never loads model
weights or imports `src.inference.engines.transformers` -- it only forwards
validated requests to a healthy GPU worker via `RouterService`/`WorkerClient`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.inference.middleware.error_handler import register_error_handlers
from src.inference.middleware.logging import AccessLogMiddleware
from src.inference.middleware.request_id import RequestIDMiddleware
from src.inference.router.routes.health import router as health_router
from src.inference.router.routes.inference import router as inference_router
from src.inference.router.service import RouterService
from src.utils.logger import logger


def create_router_app(router_service: RouterService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_service = router_service

        logger.info("Probing configured workers...")
        await router_service.refresh_worker_status()
        logger.info("Initial worker probe complete.")

        yield

        logger.info("Shutting down public router.")
        await router_service.worker_client.aclose()

    app = FastAPI(title="Qwen-VL Public Router", lifespan=lifespan)

    # Registration order matters: Starlette makes the *last*-added
    # middleware the outermost one. RequestIDMiddleware must run first (be
    # outermost) so request.state.request_id exists by the time
    # AccessLogMiddleware and the route handlers run, and so it's set before
    # WorkerClient propagates it to the selected worker.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(inference_router)

    return app
