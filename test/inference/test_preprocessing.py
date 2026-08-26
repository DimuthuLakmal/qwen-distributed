import io

import pytest
from PIL import Image

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.preprocessing import (
    InvalidGenerationOptionsError,
    InvalidRequestError,
    PayloadTooLargeError,
    RawImage,
    UnsupportedMediaTypeError,
    build_conversation_messages,
    validate_and_build_request,
)


def _limits(**overrides) -> InferenceLimitsConfig:
    defaults = dict(
        max_prompt_chars=4000,
        max_images=4,
        max_image_bytes=8_000_000,
        max_image_pixels=20_000_000,
        max_new_tokens=512,
    )
    defaults.update(overrides)
    return InferenceLimitsConfig(**defaults)


def _defaults(**overrides) -> GenerationDefaultsConfig:
    defaults = dict(do_sample=False, max_new_tokens=128, num_beams=1, temperature=1.0)
    defaults.update(overrides)
    return GenerationDefaultsConfig(**defaults)


def _png_bytes(size: tuple[int, int] = (8, 8), color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="blue").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_text_plus_valid_image_builds_request_and_messages():
    request = validate_and_build_request(
        prompt="What is in this image?",
        system_prompt=None,
        raw_images=[RawImage(data=_png_bytes(), content_type="image/png")],
        generation_overrides=None,
        limits=_limits(),
        defaults=_defaults(),
    )

    assert request.prompt == "What is in this image?"
    assert len(request.images) == 1
    assert request.images[0].image.mode == "RGB"

    messages = build_conversation_messages(request)
    assert messages[-1]["role"] == "user"
    content_types = [part["type"] for part in messages[-1]["content"]]
    assert content_types == ["image", "text"]


def test_missing_prompt_raises_invalid_request_error():
    with pytest.raises(InvalidRequestError):
        validate_and_build_request(
            prompt=None,
            system_prompt=None,
            raw_images=[],
            generation_overrides=None,
            limits=_limits(),
            defaults=_defaults(),
        )


def test_blank_prompt_raises_invalid_request_error():
    with pytest.raises(InvalidRequestError):
        validate_and_build_request(
            prompt="   ",
            system_prompt=None,
            raw_images=[],
            generation_overrides=None,
            limits=_limits(),
            defaults=_defaults(),
        )


def test_unsupported_image_type_raises_unsupported_media_type_error():
    with pytest.raises(UnsupportedMediaTypeError):
        validate_and_build_request(
            prompt="describe",
            system_prompt=None,
            raw_images=[RawImage(data=b"not an image", content_type="image/png")],
            generation_overrides=None,
            limits=_limits(),
            defaults=_defaults(),
        )


def test_oversized_image_payload_raises_payload_too_large_error():
    oversized = b"0" * 1000
    with pytest.raises(PayloadTooLargeError):
        validate_and_build_request(
            prompt="describe",
            system_prompt=None,
            raw_images=[RawImage(data=oversized, content_type="image/png")],
            generation_overrides=None,
            limits=_limits(max_image_bytes=100),
            defaults=_defaults(),
        )


def test_excessive_image_count_raises_invalid_request_error():
    images = [RawImage(data=_png_bytes(), content_type="image/png") for _ in range(3)]
    with pytest.raises(InvalidRequestError):
        validate_and_build_request(
            prompt="describe",
            system_prompt=None,
            raw_images=images,
            generation_overrides=None,
            limits=_limits(max_images=2),
            defaults=_defaults(),
        )


def test_excessive_pixel_dimensions_raises_payload_too_large_error():
    large_image = RawImage(data=_png_bytes(size=(200, 200)), content_type="image/png")
    with pytest.raises(PayloadTooLargeError):
        validate_and_build_request(
            prompt="describe",
            system_prompt=None,
            raw_images=[large_image],
            generation_overrides=None,
            limits=_limits(max_image_pixels=1000),
            defaults=_defaults(),
        )


def test_excessive_max_new_tokens_raises_invalid_generation_options_error():
    with pytest.raises(InvalidGenerationOptionsError):
        validate_and_build_request(
            prompt="describe",
            system_prompt=None,
            raw_images=[],
            generation_overrides={"max_new_tokens": 10_000},
            limits=_limits(max_new_tokens=512),
            defaults=_defaults(),
        )


def test_system_prompt_present_is_included_in_messages():
    request = validate_and_build_request(
        prompt="describe",
        system_prompt="You are a helpful assistant.",
        raw_images=[],
        generation_overrides=None,
        limits=_limits(),
        defaults=_defaults(),
    )

    messages = build_conversation_messages(request)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == "You are a helpful assistant."


def test_multiple_images_are_supported():
    raw_images = [
        RawImage(data=_png_bytes(), content_type="image/png"),
        RawImage(data=_jpeg_bytes(), content_type="image/jpeg"),
    ]

    request = validate_and_build_request(
        prompt="compare these images",
        system_prompt=None,
        raw_images=raw_images,
        generation_overrides=None,
        limits=_limits(max_images=4),
        defaults=_defaults(),
    )

    assert len(request.images) == 2

    messages = build_conversation_messages(request)
    image_parts = [part for part in messages[-1]["content"] if part["type"] == "image"]
    assert len(image_parts) == 2


def test_generation_overrides_cannot_bypass_server_limit():
    request = validate_and_build_request(
        prompt="describe",
        system_prompt=None,
        raw_images=[],
        generation_overrides={"max_new_tokens": 300},
        limits=_limits(max_new_tokens=512),
        defaults=_defaults(max_new_tokens=128),
    )

    assert request.generation.max_new_tokens == 300


def test_unknown_generation_override_key_is_rejected():
    with pytest.raises(InvalidGenerationOptionsError):
        validate_and_build_request(
            prompt="describe",
            system_prompt=None,
            raw_images=[],
            generation_overrides={"top_p": 0.9},
            limits=_limits(),
            defaults=_defaults(),
        )
