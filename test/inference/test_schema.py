from PIL import Image

from src.inference.schema import (
    GenerationConfig,
    ImageInput,
    InferenceRequest,
    InferenceResponse,
)


def test_generation_config_defaults_are_deterministic():
    generation = GenerationConfig()

    assert generation.do_sample is False
    assert generation.num_beams == 1
    assert generation.max_new_tokens > 0


def test_inference_request_defaults_to_no_images_and_default_generation():
    request = InferenceRequest(request_id="req-1", prompt="describe this")

    assert request.system_prompt is None
    assert request.images == []
    assert isinstance(request.generation, GenerationConfig)


def test_inference_request_holds_image_inputs():
    image = Image.new("RGB", (4, 4))
    request = InferenceRequest(
        request_id="req-2",
        prompt="what is in the image?",
        images=[ImageInput(image=image)],
    )

    assert len(request.images) == 1
    assert request.images[0].image.size == (4, 4)


def test_inference_request_default_images_are_independent_per_instance():
    request_a = InferenceRequest(request_id="a", prompt="p")
    request_b = InferenceRequest(request_id="b", prompt="p")

    request_a.images.append(ImageInput(image=Image.new("RGB", (2, 2))))

    assert request_b.images == []


def test_inference_response_carries_canonical_fields():
    response = InferenceResponse(
        request_id="req-3",
        generated_text="a cat sitting on a mat",
        model_version="qwen-vl-sft-v1",
        input_token_count=42,
        output_token_count=8,
        latency_seconds=0.123,
    )

    assert response.request_id == "req-3"
    assert response.generated_text
    assert response.model_version == "qwen-vl-sft-v1"
    assert response.input_token_count == 42
    assert response.output_token_count == 8
    assert response.latency_seconds == 0.123
