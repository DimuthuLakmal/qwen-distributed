"""
GET /internal/health/live, GET /internal/health/ready -- worker liveness and
readiness, consumed by the public router (and any external process
supervisor) to decide whether this worker should receive traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.inference.api.dependencies import get_inference_service
from src.inference.service import InferenceService

router = APIRouter(prefix="/internal/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    return {"status": "alive"}


@router.get("/ready")
async def readiness(service: InferenceService = Depends(get_inference_service)) -> dict:
    if service is None or not service.is_ready():
        raise HTTPException(status_code=503, detail="Engine not ready")

    return {"status": "ready"}
