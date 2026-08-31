"""
GET /internal/status -- worker identity, readiness, and current in-flight
request count. Polled by the public router (`RouterService.refresh_worker_status`)
to decide which workers are eligible for routing and, for the `least_loaded`
strategy, which one to prefer.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.inference.schema import WorkerStatusSchema
from src.inference.service import InferenceService
from src.inference.transformers.worker_app.dependencies import get_inference_service, get_worker_id

router = APIRouter(prefix="/internal", tags=["worker-status"])


@router.get("/status", response_model=WorkerStatusSchema)
async def status(
    worker_id: str = Depends(get_worker_id),
    service: InferenceService = Depends(get_inference_service),
) -> WorkerStatusSchema:
    return WorkerStatusSchema(
        worker_id=worker_id,
        ready=service is not None and await service.is_ready(),
        active_requests=service.active_request_count if service is not None else 0,
    )
