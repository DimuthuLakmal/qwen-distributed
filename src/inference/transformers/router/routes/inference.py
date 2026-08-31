"""
POST /v1/generate -- the public, external-client-facing inference endpoint.

Owns the multipart HTTP contract (this is what changed from Stage 1A's flat
implementation: this contract used to be served directly by the worker,
now the worker only speaks the internal JSON contract defined alongside the
canonical domain objects in `src/inference/schema.py`). This module parses/validates the
HTTP-level request shape, reads uploaded image bytes, applies the same
prompt/image-count edge checks the worker enforces (reused from
`src/inference/preprocessing.py` rather than duplicated), base64-encodes
images for the worker hop, and delegates selection/forwarding to
`RouterService`. Full image decode/format/pixel-count validation stays
worker-side only -- the worker's own rejection responses pass straight
through via `WorkerRequestRejectedError`.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from src.inference.http_utils import read_bounded_upload
from src.inference.preprocessing import validate_image_count, validate_prompt
from src.inference.schema import (
    GenerateRequestSchema,
    GenerateResponseSchema,
    WorkerGenerateRequestSchema,
    WorkerGenerationOptions,
    WorkerImagePayload,
)
from src.inference.transformers.router.dependencies import get_router_service
from src.inference.transformers.router.service import RouterService

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post("/generate", response_model=GenerateResponseSchema)
async def generate(
    request: Request,
    payload: str = Form(..., alias="request"),
    images: list[UploadFile] = File(default=[]),
    service: RouterService = Depends(get_router_service),
) -> GenerateResponseSchema:
    try:
        parsed_request = GenerateRequestSchema.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    validate_prompt(parsed_request.prompt, service.limits)
    validate_image_count(len(images), service.limits)

    image_payloads = [
        WorkerImagePayload(
            data_base64=base64.b64encode(
                await read_bounded_upload(image, service.limits.max_image_bytes)
            ).decode("ascii"),
            content_type=image.content_type,
        )
        for image in images
    ]

    generation_overrides = None
    if parsed_request.generation is not None:
        generation_overrides = {
            key: value
            for key, value in parsed_request.generation.model_dump().items()
            if value is not None
        }

    worker_request = WorkerGenerateRequestSchema(
        request_id=request.state.request_id,
        prompt=parsed_request.prompt,
        system_prompt=parsed_request.system_prompt,
        images=image_payloads,
        generation=WorkerGenerationOptions(**generation_overrides) if generation_overrides else None,
    )

    response = await service.generate(worker_request)
    return GenerateResponseSchema(**response.model_dump())
