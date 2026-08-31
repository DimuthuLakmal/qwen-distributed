"""
All exception types raised anywhere in the inference module -- validation,
worker service, and router/worker-communication errors alike -- live here so
error handling has a single source of truth. `src/inference/middleware/error_handler.py`
maps every type below to an HTTP status code in one place and is shared by
both the worker (`src/inference/transformers/worker_app/`) and the router (`src/inference/transformers/router/`)
apps; neither process raises all of these, but registering a handler for a
type that's never thrown there is harmless.
"""


class InvalidRequestError(Exception):
    """The request body itself is malformed or missing required fields."""


class UnsupportedMediaTypeError(Exception):
    """An uploaded image is not a supported or decodable image type."""


class PayloadTooLargeError(Exception):
    """An uploaded image exceeds configured byte-size or pixel limits."""


class InvalidGenerationOptionsError(Exception):
    """Requested generation options are invalid or exceed server limits."""


class EngineNotReadyError(Exception):
    """The inference engine has not finished loading / is not ready."""


class InferenceExecutionError(Exception):
    """The engine raised an unexpected error while generating a response."""


class NoHealthyWorkerError(Exception):
    """No configured worker is currently healthy/ready to receive traffic."""


class WorkerClientError(Exception):
    """Base class for all worker-communication failures."""


class WorkerConnectionError(WorkerClientError):
    """The worker was unreachable or the request timed out."""


class WorkerUnavailableError(WorkerClientError):
    """The worker reported itself as not ready (HTTP 503)."""


class WorkerInferenceError(WorkerClientError):
    """The worker returned an unexpected server error (HTTP >=500, not 503)."""


class WorkerRequestRejectedError(WorkerClientError):
    """
    The worker rejected the request as invalid (HTTP 4xx). Status code and
    sanitized body come straight from the worker's own error handler and are
    passed through to the public client unchanged.
    """

    def __init__(self, status_code: int, body: dict) -> None:
        super().__init__(f"Worker rejected request with status {status_code}: {body}")
        self.status_code = status_code
        self.body = body
