"""
Inference-module schemas: canonical domain objects plus the worker wire
contract, kept in one file for discoverability.

The dataclasses below (`ImageInput`, `RawImage`, `GenerationConfig`,
`InferenceRequest`, `InferenceResponse`) are what `InferenceService` and
`InferenceEngine` implementations communicate with. They intentionally do
not depend on Pydantic/FastAPI or on any Hugging Face/Transformers types, so
the service/engine layers stay testable and swappable (see
`src/inference/engines/base.py`) -- keep new additions to this section
dependency-free too.

The Pydantic models in the "Worker wire contract" section below are a
separate concern: the JSON contract between the public router
(`src/inference/transformers/router/`) and a worker's internal API (`src/inference/transformers/worker_app/`).
Images cross that hop base64-encoded rather than proxied as multipart,
trading a small encode/decode cost for a simple, typed contract. They are
intentionally distinct from the "Public HTTP contract" section further
below, which owns the multipart HTTP contract external clients see (shared
by both the Transformers router and the vLLM direct app).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image
from pydantic import BaseModel


@dataclass
class ImageInput:
    """A single decoded, validated image ready for engine consumption."""
    image: Image.Image


@dataclass
class RawImage:
    """An uploaded image before decoding/validation."""
    data: bytes
    content_type: str | None = None


@dataclass
class GenerationConfig:
    do_sample: bool = False
    max_new_tokens: int = 128
    num_beams: int = 1
    temperature: float = 1.0


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    system_prompt: str | None = None
    images: list[ImageInput] = field(default_factory=list)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class InferenceResponse:
    request_id: str
    generated_text: str
    model_version: str
    input_token_count: int
    output_token_count: int
    latency_seconds: float


# --- Worker wire contract -------------------------------------------------
# See module docstring: Pydantic models for the router<->worker JSON hop.


class WorkerImagePayload(BaseModel):
    data_base64: str
    content_type: str | None = None


class WorkerGenerationOptions(BaseModel):
    do_sample: bool | None = None
    max_new_tokens: int | None = None
    num_beams: int | None = None
    temperature: float | None = None


class WorkerGenerateRequestSchema(BaseModel):
    request_id: str
    prompt: str
    system_prompt: str | None = None
    images: list[WorkerImagePayload] = []
    generation: WorkerGenerationOptions | None = None


class WorkerGenerateResponseSchema(BaseModel):
    request_id: str
    generated_text: str
    model_version: str
    input_token_count: int
    output_token_count: int
    latency_seconds: float


class WorkerStatusSchema(BaseModel):
    worker_id: str
    ready: bool
    active_requests: int


# --- Public HTTP contract --------------------------------------------------
# The multipart `POST /v1/generate` request/response schemas external
# clients see. Shared by `src/inference/transformers/router/routes/inference.py`
# (Transformers: forwards to a worker) and `src/inference/vllm/routes/inference.py`
# (vLLM: calls InferenceService directly), so the external contract stays
# identical regardless of backend without duplicating these definitions.
# The bounded-upload-reading helper both apps use to read these requests
# lives in `src/inference/http_utils.py`, not here -- this file is schema
# definitions only.


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
