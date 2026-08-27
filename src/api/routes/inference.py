"""
POST /v1/generate -- the only inference endpoint in Stage 1.

This module intentionally contains no Qwen/tokenizer/generation logic: it
parses/validates the HTTP-level request shape, reads uploaded image bytes,
and delegates everything else to `InferenceService`.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError

from src.api.dependencies import get_inference_service
from src.inference.preprocessing import PayloadTooLargeError, RawImage
from src.inference.service import InferenceService

router = APIRouter(prefix="/v1", tags=["inference"])

_READ_CHUNK_BYTES = 1024 * 1024


class GenerationOptionsSchema(BaseModel):
    do_sample: bool | None = None
    max_new_tokens: int | None = None
    num_beams: int | None = None
    temperature: float | None = None


class GenerateRequestSchema(BaseModel):
    prompt: str
    system_prompt: str | None = None
    generation: GenerationOptionsSchema | None = None


class GenerateResponseSchema(BaseModel):
    request_id: str
    generated_text: str
    model_version: str
    input_token_count: int
    output_token_count: int
    latency_seconds: float


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
            data=await _read_bounded(image, service.limits.max_image_bytes),
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

    response = await asyncio.to_thread(
        service.generate,
        request_id=request.state.request_id,
        prompt=parsed_request.prompt,
        system_prompt=parsed_request.system_prompt,
        raw_images=raw_images,
        generation_overrides=generation_overrides,
    )

    return GenerateResponseSchema(**asdict(response))


async def _read_bounded(upload_file: UploadFile, max_bytes: int) -> bytes:
    """
    Read an upload incrementally and abort as soon as it exceeds `max_bytes`,
    instead of buffering an arbitrarily large upload fully into memory first.
    """

    chunks: list[bytes] = []
    total_bytes = 0

    while True:
        chunk = await upload_file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise PayloadTooLargeError(
                f"image payload exceeds the configured limit of {max_bytes} bytes."
            )

        chunks.append(chunk)

    return b"".join(chunks)
