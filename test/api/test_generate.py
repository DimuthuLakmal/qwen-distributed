import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import create_app
from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.engines.fake import FakeInferenceEngine


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


def _defaults() -> GenerationDefaultsConfig:
    return GenerationDefaultsConfig(do_sample=False, max_new_tokens=128, num_beams=1, temperature=1.0)


def _client(engine: FakeInferenceEngine, **limit_overrides) -> TestClient:
    app = create_app(
        engine_factory=lambda: engine,
        limits=_limits(**limit_overrides),
        defaults=_defaults(),
    )
    return TestClient(app)


def _png_bytes(size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_generate_text_only_returns_200_and_expected_shape():
    engine = FakeInferenceEngine(generated_text="hello there", model_version="v1")

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": "say hello"})},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["generated_text"] == "hello there"
    assert body["model_version"] == "v1"
    assert "request_id" in body


def test_generate_with_image_upload_reaches_engine():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": "what is this?"})},
            files={"images": ("test.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert len(engine.received_requests) == 1
    assert len(engine.received_requests[0].images) == 1


def test_generate_missing_prompt_returns_400():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": ""})},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidRequestError"


def test_generate_malformed_json_payload_returns_400():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": "not json"},
        )

    assert response.status_code == 400


def test_generate_oversized_image_returns_413():
    engine = FakeInferenceEngine()

    with _client(engine, max_image_bytes=10) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": "describe"})},
            files={"images": ("test.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PayloadTooLargeError"


def test_generate_unsupported_image_type_returns_415():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": "describe"})},
            files={"images": ("test.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UnsupportedMediaTypeError"


def test_generate_excessive_max_new_tokens_returns_400():
    engine = FakeInferenceEngine()

    with _client(engine, max_new_tokens=64) as client:
        response = client.post(
            "/v1/generate",
            data={
                "request": json.dumps(
                    {"prompt": "describe", "generation": {"max_new_tokens": 5000}}
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidGenerationOptionsError"


def test_generate_when_engine_not_ready_returns_503():
    engine = FakeInferenceEngine(ready=False)

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": "describe"})},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "EngineNotReadyError"


def test_generate_unexpected_engine_error_returns_500_without_stack_trace():
    class _BrokenEngine(FakeInferenceEngine):
        def generate(self, request):
            raise RuntimeError("/secret/internal/path was not found")

    engine = _BrokenEngine()

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": "describe"})},
        )

    assert response.status_code == 500
    body = response.json()
    assert "secret" not in json.dumps(body)
    assert "Traceback" not in json.dumps(body)


def test_generate_response_includes_request_id_header():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/v1/generate",
            data={"request": json.dumps({"prompt": "describe"})},
        )

    assert "x-request-id" in response.headers
    assert response.json()["request_id"]
