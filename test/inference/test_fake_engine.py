from src.inference.engines.fake import FakeInferenceEngine
from src.inference.schema import InferenceRequest


def test_fake_engine_is_ready_by_default():
    engine = FakeInferenceEngine()
    assert engine.health() is True


def test_fake_engine_can_report_not_ready():
    engine = FakeInferenceEngine(ready=False)
    assert engine.health() is False


def test_fake_engine_generate_returns_configured_response_and_records_request():
    engine = FakeInferenceEngine(generated_text="a cat on a mat", model_version="v-test")
    request = InferenceRequest(request_id="req-1", prompt="describe this image")

    response = engine.generate(request)

    assert response.request_id == "req-1"
    assert response.generated_text == "a cat on a mat"
    assert response.model_version == "v-test"
    assert response.output_token_count == 5
    assert engine.received_requests == [request]
