"""
Single source of truth for mapping inference-module exceptions (see
`src/inference/exception.py`) to sanitized HTTP responses, and for ensuring
unexpected errors never leak stack traces / internal details to clients.

Shared unchanged by both the worker (`src/inference/api/`) and the public
router (`src/inference/router/`) apps: both call `register_error_handlers(app)`
with no arguments. Neither process ever raises every type in the map below
(e.g. a worker never raises `NoHealthyWorkerError`), but registering a
handler for a type that's never thrown there is harmless -- this keeps the
status-code mapping in one place instead of duplicated/parameterized per app.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.inference.exception import (
    EngineNotReadyError,
    InferenceExecutionError,
    InvalidGenerationOptionsError,
    InvalidRequestError,
    NoHealthyWorkerError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    WorkerConnectionError,
    WorkerInferenceError,
    WorkerRequestRejectedError,
    WorkerUnavailableError,
)
from src.utils.logger import logger

_STATUS_CODE_BY_ERROR_TYPE = {
    InvalidRequestError: 400,
    InvalidGenerationOptionsError: 400,
    UnsupportedMediaTypeError: 415,
    PayloadTooLargeError: 413,
    EngineNotReadyError: 503,
    InferenceExecutionError: 500,
    NoHealthyWorkerError: 503,
    WorkerConnectionError: 503,
    WorkerUnavailableError: 503,
    WorkerInferenceError: 502,
}


def register_error_handlers(app: FastAPI) -> None:
    for error_type, status_code in _STATUS_CODE_BY_ERROR_TYPE.items():
        app.add_exception_handler(error_type, _make_known_error_handler(status_code))

    # Special-cased: the worker already produced a sanitized error body via
    # this same handler map, one hop earlier. The router passes that body
    # through unchanged instead of re-wrapping it with a fresh one.
    app.add_exception_handler(WorkerRequestRejectedError, _handle_worker_request_rejected)

    app.add_exception_handler(Exception, _handle_unexpected_error)


def _make_known_error_handler(status_code: int):
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=_error_body(request, exc),
        )

    return _handler


async def _handle_worker_request_rejected(
    request: Request, exc: WorkerRequestRejectedError
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.body)


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        "Unhandled server error | request_id=%s | error_type=%s",
        request_id,
        type(exc).__name__,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "InternalServerError",
                "message": "An unexpected error occurred.",
                "request_id": request_id,
            }
        },
    )


def _error_body(request: Request, exc: Exception) -> dict:
    return {
        "error": {
            "code": type(exc).__name__,
            "message": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        }
    }
