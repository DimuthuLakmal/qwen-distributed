"""
GET /health/live, GET /health/ready -- public liveness/readiness for the
single-process topology used by the vLLM backend. Same paths as the
Transformers router's health routes, so the external contract is identical
regardless of backend.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.inference.service import InferenceService
from src.inference.vllm.dependencies import get_inference_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    return {"status": "alive"}


@router.get("/ready")
async def readiness(service: InferenceService = Depends(get_inference_service)) -> dict:
    if service is None or not await service.is_ready():
        raise HTTPException(status_code=503, detail="Engine not ready")

    return {"status": "ready"}