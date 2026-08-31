"""
End-to-end tests for the worker-internal FastAPI app (`create_worker_app`),
exercised entirely against FakeInferenceEngine. This app speaks the internal
JSON contract (`/internal/generate`, `/internal/health/*`, `/internal/status`)
consumed only by the public router's `WorkerClient` -- never by external
clients directly. There was previously zero automated coverage for this
module; this closes that gap so the Transformers regression baseline
(todo.md Task 18) has real teeth.
"""

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.transformers.worker_app.app import create_worker_app
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


def _client(engine: FakeInferenceEngine, worker_id: str = "gpu-0", **limit_overrides) -> TestClient:
    app = create_worker_app(
        worker_id=worker_id,
        engine_factory=lambda: engine,
        limits=_limits(**limit_overrides),
        defaults=_defaults(),
    )
    return TestClient(app)


def _png_base64(size=(8, 8)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_generate_happy_path_returns_200_and_expected_shape():
    # The request id crosses via the X-Request-ID header (set by
    # RequestIDMiddleware, shared with the public router), not the JSON
    # body's `request_id` field -- see src/inference/transformers/worker_app/routes/internal_generate.py.
    engine = FakeInferenceEngine(generated_text="hello there", model_version="v1")

    with _client(engine) as client:
        response = client.post(
            "/internal/generate",
            json={"request_id": "req-1", "prompt": "say hello", "images": []},
            headers={"X-Request-ID": "req-1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["generated_text"] == "hello there"
    assert body["model_version"] == "v1"


def test_generate_with_base64_image_reaches_engine():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/internal/generate",
            json={
                "request_id": "req-2",
                "prompt": "what is this?",
                "images": [{"data_base64": _png_base64(), "content_type": "image/png"}],
            },
        )

    assert response.status_code == 200
    assert len(engine.received_requests) == 1
    assert len(engine.received_requests[0].images) == 1


def test_generate_missing_prompt_returns_400():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/internal/generate",
            json={"request_id": "req-3", "prompt": "", "images": []},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "InvalidRequestError"


def test_generate_unsupported_image_type_returns_415():
    engine = FakeInferenceEngine()

    with _client(engine) as client:
        response = client.post(
            "/internal/generate",
            json={
                "request_id": "req-4",
                "prompt": "describe",
                "images": [
                    {
                        "data_base64": base64.b64encode(b"not an image").decode("ascii"),
                        "content_type": "text/plain",
                    }
                ],
            },
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UnsupportedMediaTypeError"


def test_generate_when_engine_not_ready_returns_503():
    engine = FakeInferenceEngine(ready=False)

    with _client(engine) as client:
        response = client.post(
            "/internal/generate",
            json={"request_id": "req-5", "prompt": "describe", "images": []},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "EngineNotReadyError"


def test_liveness_always_succeeds_even_when_engine_not_ready():
    engine = FakeInferenceEngine(ready=False)

    with _client(engine) as client:
        response = client.get("/internal/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_succeeds_when_engine_ready():
    engine = FakeInferenceEngine(ready=True)

    with _client(engine) as client:
        response = client.get("/internal/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_fails_when_engine_not_ready():
    engine = FakeInferenceEngine(ready=False)

    with _client(engine) as client:
        response = client.get("/internal/health/ready")

    assert response.status_code == 503


def test_status_reports_worker_id_and_readiness():
    engine = FakeInferenceEngine(ready=True)

    with _client(engine, worker_id="gpu-3") as client:
        response = client.get("/internal/status")

    assert response.status_code == 200
    body = response.json()
    assert body["worker_id"] == "gpu-3"
    assert body["ready"] is True
    assert body["active_requests"] == 0