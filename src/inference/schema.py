"""
Canonical, model-independent inference domain objects.

These are the objects `InferenceService` and `InferenceEngine` implementations
communicate with. They intentionally do not depend on FastAPI/Pydantic or on
any Hugging Face/Transformers types, so the service/engine layers stay
testable and swappable (see `src/inference/engines/base.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass
class ImageInput:
    """A single decoded, validated image ready for engine consumption."""

    image: Image.Image


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
