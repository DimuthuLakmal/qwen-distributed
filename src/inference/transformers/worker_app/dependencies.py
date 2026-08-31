from fastapi import Request

from src.inference.service import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    """
    Return the InferenceService created once during worker startup (see
    `src/inference/transformers/worker_app/app.py`'s lifespan). Never constructs a new
    engine/service per request.
    """

    return request.app.state.inference_service


def get_worker_id(request: Request) -> str:
    return request.app.state.worker_id
