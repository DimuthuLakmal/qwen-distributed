"""
POST /internal/generate -- the worker-facing generation endpoint, called only
by the public router's `WorkerClient` (never by external clients directly).

Takes the JSON `WorkerGenerateRequestSchema` contract (base64-encoded images)
rather than the public multipart contract -- the router owns decoding
uploads/enforcing byte limits on the way in and re-encodes images once for
this hop. This module intentionally contains no Qwen/tokenizer/generation
logic: it decodes the base64 image payloads and delegates everything else to
`InferenceService`.
"""

from __future__ import annotations

import base64
from dataclasses import asdict

from fastapi import APIRouter, Depends, Request

from src.inference.preprocessing import RawImage
from src.inference.schema import WorkerGenerateRequestSchema, WorkerGenerateResponseSchema
from src.inference.service import InferenceService
from src.inference.transformers.worker_app.dependencies import get_inference_service

router = APIRouter(prefix="/internal", tags=["worker-generate"])


@router.post("/generate", response_model=WorkerGenerateResponseSchema)
async def generate(
    request: Request,
    payload: WorkerGenerateRequestSchema,
    service: InferenceService = Depends(get_inference_service),
) -> WorkerGenerateResponseSchema:
    raw_images = [
        RawImage(data=base64.b64decode(image.data_base64), content_type=image.content_type)
        for image in payload.images
    ]

    generation_overrides = None
    if payload.generation is not None:
        generation_overrides = {
            key: value
            for key, value in payload.generation.model_dump().items()
            if value is not None
        }

    response = await service.generate(
        request_id=request.state.request_id,
        prompt=payload.prompt,
        system_prompt=payload.system_prompt,
        raw_images=raw_images,
        generation_overrides=generation_overrides,
    )

    return WorkerGenerateResponseSchema(**asdict(response))
