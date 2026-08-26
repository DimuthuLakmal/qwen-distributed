"""
Request validation and Qwen multimodal message construction.

Everything here runs before any GPU/model work, so invalid requests are
rejected cheaply. Nothing in this module calls into Transformers or touches
`model.generate()` -- it only produces the canonical `InferenceRequest` and,
separately, the Qwen chat-template `messages` list that an engine can feed
into its processor.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.schema import GenerationConfig, ImageInput, InferenceRequest

# Formats we are confident the Qwen processor / downstream pipeline handles
# correctly. Anything else is rejected rather than guessed at.
_ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

_ALLOWED_GENERATION_OVERRIDE_KEYS = {
    "do_sample",
    "max_new_tokens",
    "num_beams",
    "temperature",
}


class InvalidRequestError(Exception):
    """The request body itself is malformed or missing required fields."""


class UnsupportedMediaTypeError(Exception):
    """An uploaded image is not a supported or decodable image type."""


class PayloadTooLargeError(Exception):
    """An uploaded image exceeds configured byte-size or pixel limits."""


class InvalidGenerationOptionsError(Exception):
    """Requested generation options are invalid or exceed server limits."""


@dataclass
class RawImage:
    """An uploaded image before decoding/validation."""

    data: bytes
    content_type: str | None = None


def validate_and_build_request(
    *,
    prompt: str | None,
    system_prompt: str | None,
    raw_images: list[RawImage],
    generation_overrides: dict[str, Any] | None,
    limits: InferenceLimitsConfig,
    defaults: GenerationDefaultsConfig,
    request_id: str | None = None,
) -> InferenceRequest:
    """
    Validate a raw request and normalize it into a canonical `InferenceRequest`.

    Raises one of the exceptions above on any invalid input. Never touches
    the model/GPU.
    """

    _validate_prompt(prompt, limits)
    _validate_image_count(raw_images, limits)

    images = [_decode_and_validate_image(raw_image, limits) for raw_image in raw_images]
    generation = _build_generation_config(generation_overrides, limits, defaults)

    return InferenceRequest(
        request_id=request_id or str(uuid.uuid4()),
        prompt=prompt.strip(),
        system_prompt=system_prompt.strip() if system_prompt else None,
        images=images,
        generation=generation,
    )


def build_conversation_messages(request: InferenceRequest) -> list[dict[str, Any]]:
    """
    Build a Qwen-style multimodal conversation from a canonical request:

        [
            {"role": "system", "content": [{"type": "text", "text": ...}]},   # optional
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": PIL.Image.Image(...)},
                    ...
                    {"type": "text", "text": ...},
                ],
            },
        ]

    Mirrors `src/data/adapters/llava.py`'s message shape so the same
    processor.apply_chat_template() call used in training/eval works
    unchanged at inference time.
    """

    messages: list[dict[str, Any]] = []

    if request.system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": request.system_prompt}],
            }
        )

    content: list[dict[str, Any]] = [
        {"type": "image", "image": image_input.image} for image_input in request.images
    ]
    content.append({"type": "text", "text": request.prompt})

    messages.append({"role": "user", "content": content})

    return messages


def _validate_prompt(prompt: str | None, limits: InferenceLimitsConfig) -> None:
    if prompt is None or not prompt.strip():
        raise InvalidRequestError("prompt must be a non-empty string.")

    if len(prompt) > limits.max_prompt_chars:
        raise InvalidRequestError(
            f"prompt length ({len(prompt)} chars) exceeds the configured "
            f"limit of {limits.max_prompt_chars} chars."
        )


def _validate_image_count(raw_images: list[RawImage], limits: InferenceLimitsConfig) -> None:
    if len(raw_images) > limits.max_images:
        raise InvalidRequestError(
            f"received {len(raw_images)} images, which exceeds the "
            f"configured limit of {limits.max_images}."
        )


def _decode_and_validate_image(
    raw_image: RawImage,
    limits: InferenceLimitsConfig,
) -> ImageInput:
    if len(raw_image.data) > limits.max_image_bytes:
        raise PayloadTooLargeError(
            f"image payload ({len(raw_image.data)} bytes) exceeds the "
            f"configured limit of {limits.max_image_bytes} bytes."
        )

    try:
        image = Image.open(io.BytesIO(raw_image.data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedMediaTypeError(
            "Uploaded file could not be decoded as an image."
        ) from exc

    if image.format not in _ALLOWED_IMAGE_FORMATS:
        raise UnsupportedMediaTypeError(
            f"Unsupported image format {image.format!r}. "
            f"Allowed formats: {sorted(_ALLOWED_IMAGE_FORMATS)}."
        )

    pixel_count = image.width * image.height
    if pixel_count > limits.max_image_pixels:
        raise PayloadTooLargeError(
            f"image has {pixel_count} pixels, which exceeds the configured "
            f"limit of {limits.max_image_pixels} pixels."
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    return ImageInput(image=image)


def _build_generation_config(
    generation_overrides: dict[str, Any] | None,
    limits: InferenceLimitsConfig,
    defaults: GenerationDefaultsConfig,
) -> GenerationConfig:
    overrides = generation_overrides or {}

    unknown_keys = set(overrides) - _ALLOWED_GENERATION_OVERRIDE_KEYS
    if unknown_keys:
        raise InvalidGenerationOptionsError(
            f"Unknown generation option(s): {sorted(unknown_keys)}."
        )

    max_new_tokens = overrides.get("max_new_tokens", defaults.max_new_tokens)
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise InvalidGenerationOptionsError("max_new_tokens must be a positive integer.")

    if max_new_tokens > limits.max_new_tokens:
        raise InvalidGenerationOptionsError(
            f"max_new_tokens ({max_new_tokens}) exceeds the configured "
            f"limit of {limits.max_new_tokens}."
        )

    num_beams = overrides.get("num_beams", defaults.num_beams)
    if not isinstance(num_beams, int) or num_beams < 1:
        raise InvalidGenerationOptionsError("num_beams must be a positive integer.")

    do_sample = overrides.get("do_sample", defaults.do_sample)
    if not isinstance(do_sample, bool):
        raise InvalidGenerationOptionsError("do_sample must be a boolean.")

    temperature = overrides.get("temperature", defaults.temperature)
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise InvalidGenerationOptionsError("temperature must be a positive number.")

    return GenerationConfig(
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=float(temperature),
    )
