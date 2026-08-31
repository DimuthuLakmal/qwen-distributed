"""
GET /internal/health/live, GET /internal/health/ready -- worker liveness and
readiness, consumed by the public router (and any external process
supervisor) to decide whether this worker should receive traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.inference.service import InferenceService
from src.inference.transformers.worker_app.dependencies import get_inference_service

router = APIRouter(prefix="/internal/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    return {"status": "alive"}


@router.get("/ready")
async def readiness(service: InferenceService = Depends(get_inference_service)) -> dict:
    if service is None or not await service.is_ready():
        raise HTTPException(status_code=503, detail="Engine not ready")

    return {"status": "ready"}
