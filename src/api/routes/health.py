from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_inference_service
from src.inference.service import InferenceService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    """Succeeds whenever the process/API is alive, regardless of engine state."""

    return {"status": "alive"}


@router.get("/ready")
async def readiness(
    service: InferenceService = Depends(get_inference_service),
) -> dict:
    """Succeeds only once the engine/model has finished loading."""

    if service is None or not service.is_ready():
        raise HTTPException(status_code=503, detail="Engine not ready")

    return {"status": "ready"}
