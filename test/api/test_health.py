from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.engines.fake import FakeInferenceEngine


def _limits() -> InferenceLimitsConfig:
    return InferenceLimitsConfig(
        max_prompt_chars=4000,
        max_images=4,
        max_image_bytes=8_000_000,
        max_image_pixels=20_000_000,
        max_new_tokens=512,
    )


def _defaults() -> GenerationDefaultsConfig:
    return GenerationDefaultsConfig(do_sample=False, max_new_tokens=128, num_beams=1, temperature=1.0)


def test_liveness_always_succeeds_even_when_engine_not_ready():
    app = create_app(
        engine_factory=lambda: FakeInferenceEngine(ready=False),
        limits=_limits(),
        defaults=_defaults(),
    )

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_succeeds_when_engine_ready():
    app = create_app(
        engine_factory=lambda: FakeInferenceEngine(ready=True),
        limits=_limits(),
        defaults=_defaults(),
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_fails_when_engine_not_ready():
    app = create_app(
        engine_factory=lambda: FakeInferenceEngine(ready=False),
        limits=_limits(),
        defaults=_defaults(),
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503


def test_responses_include_request_id_header():
    app = create_app(
        engine_factory=lambda: FakeInferenceEngine(ready=True),
        limits=_limits(),
        defaults=_defaults(),
    )

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert "x-request-id" in response.headers
