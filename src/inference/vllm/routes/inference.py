"""
POST /v1/generate -- the public, external-client-facing inference endpoint
for the single-process topology used by the vLLM backend.

Unlike the Transformers router's version of this route, there is no worker
hop here: this module parses/validates the HTTP-level request shape and
calls `InferenceService.generate(...)` directly against the engine loaded
in this same process. Images never leave this process, so there's no
base64 re-encoding step.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from src.inference.http_utils import read_bounded_upload
from src.inference.schema import GenerateRequestSchema, GenerateResponseSchema, RawImage
from src.inference.service import InferenceService
from src.inference.vllm.dependencies import get_inference_service

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post("/generate", response_model=GenerateResponseSchema)
async def generate(
    request: Request,
    payload: str = Form(..., alias="request"),
    images: list[UploadFile] = File(default=[]),
    service: InferenceService = Depends(get_inference_service),
) -> GenerateResponseSchema:
    try:
        parsed_request = GenerateRequestSchema.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_images = [
        RawImage(
            data=await read_bounded_upload(image, service.limits.max_image_bytes),
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

    response = await service.generate(
        request_id=request.state.request_id,
        prompt=parsed_request.prompt,
        system_prompt=parsed_request.system_prompt,
        raw_images=raw_images,
        generation_overrides=generation_overrides,
    )

    return GenerateResponseSchema(
        request_id=response.request_id,
        generated_text=response.generated_text,
        model_version=response.model_version,
        input_token_count=response.input_token_count,
        output_token_count=response.output_token_count,
        latency_seconds=response.latency_seconds,
    )