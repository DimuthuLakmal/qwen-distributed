"""
GET /health/live, GET /health/ready -- public liveness/readiness. Readiness
is "at least one configured worker is healthy and ready" (re-probed on every
call; see `RouterService.refresh_worker_status`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.inference.router.dependencies import get_router_service
from src.inference.router.service import RouterService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    return {"status": "alive"}


@router.get("/ready")
async def readiness(service: RouterService = Depends(get_router_service)) -> dict:
    if not await service.is_ready():
        raise HTTPException(status_code=503, detail="No healthy workers available")

    return {"status": "ready"}
