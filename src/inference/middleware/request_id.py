"""
Assigns/propagates a request ID for every HTTP request.

Stored on `request.state.request_id` so route handlers, the InferenceService
call, and error handlers can all trace a single request end-to-end, and
echoed back as the `X-Request-ID` response header.

Shared by both the worker (`src/inference/api/`) and the public router
(`src/inference/router/`) apps so a request ID set by the router survives
into the forwarded worker call and back.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response
