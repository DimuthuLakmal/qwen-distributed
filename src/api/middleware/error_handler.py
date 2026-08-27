"""
Maps expected inference errors to sanitized HTTP responses and ensures
unexpected errors never leak stack traces / internal details to clients.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.inference.preprocessing import (
    InvalidGenerationOptionsError,
    InvalidRequestError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from src.inference.service import EngineNotReadyError, InferenceExecutionError
from src.utils.logger import logger

_STATUS_CODE_BY_ERROR_TYPE = {
    InvalidRequestError: 400,
    InvalidGenerationOptionsError: 400,
    UnsupportedMediaTypeError: 415,
    PayloadTooLargeError: 413,
    EngineNotReadyError: 503,
    InferenceExecutionError: 500,
}


def register_error_handlers(app: FastAPI) -> None:
    for error_type, status_code in _STATUS_CODE_BY_ERROR_TYPE.items():
        app.add_exception_handler(error_type, _make_known_error_handler(status_code))

    app.add_exception_handler(Exception, _handle_unexpected_error)


def _make_known_error_handler(status_code: int):
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=_error_body(request, exc),
        )

    return _handler


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
