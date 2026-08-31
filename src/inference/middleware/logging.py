"""
Structured HTTP access logging: request ID, method, path, status code, and
total latency only. Never logs request bodies (prompts/images) or response
bodies (generated text).

Must be registered *before* `RequestIDMiddleware` via `app.add_middleware`
(Starlette applies middleware to the request in reverse-registration order,
so the request-ID assignment needs to be the outermost/first-run middleware)
-- see `src/inference/api/app.py` and `src/inference/router/app.py`.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from src.utils.logger import logger


class AccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        latency_seconds = time.perf_counter() - start_time

        request_id = getattr(request.state, "request_id", None)

        logger.info(
            "HTTP request | request_id=%s | method=%s | path=%s | "
            "status_code=%d | latency_seconds=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_seconds,
        )

        return response
