from fastapi import Request

from src.inference.service import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    """
    Return the InferenceService created once during application startup
    (see `src/api/app.py`'s lifespan). Never constructs a new engine/service
    per request.
    """

    return request.app.state.inference_service
