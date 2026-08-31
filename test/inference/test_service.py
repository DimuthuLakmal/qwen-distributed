"""
InferenceService tests -- exercised entirely against FakeInferenceEngine,
so no Transformers/torch model is ever loaded here.
"""

import asyncio

import pytest

from src.config.schema import GenerationDefaultsConfig, InferenceLimitsConfig
from src.inference.engines.fake import FakeInferenceEngine
from src.inference.preprocessing import InvalidRequestError, RawImage
from src.inference.service import EngineNotReadyError, InferenceExecutionError, InferenceService


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


def test_generate_happy_path_returns_postprocessed_response():
    engine = FakeInferenceEngine(generated_text="  a cat on a mat  ")
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    response = asyncio.run(service.generate(request_id="req-1", prompt="describe this"))

    assert response.request_id == "req-1"
    assert response.generated_text == "a cat on a mat"
    assert len(engine.received_requests) == 1


def test_generate_preserves_caller_supplied_request_id():
    engine = FakeInferenceEngine()
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    response = asyncio.run(service.generate(request_id="caller-id-123", prompt="hello"))

    assert response.request_id == "caller-id-123"
    assert engine.received_requests[0].request_id == "caller-id-123"


def test_generate_raises_when_engine_not_ready():
    engine = FakeInferenceEngine(ready=False)
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    with pytest.raises(EngineNotReadyError):
        asyncio.run(service.generate(request_id="req-1", prompt="hello"))

    assert engine.received_requests == []


def test_generate_propagates_validation_errors_before_calling_engine():
    engine = FakeInferenceEngine()
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    with pytest.raises(InvalidRequestError):
        asyncio.run(service.generate(request_id="req-1", prompt=None))

    assert engine.received_requests == []


def test_generate_wraps_unexpected_engine_errors():
    class _BrokenEngine(FakeInferenceEngine):
        async def generate(self, request):
            raise RuntimeError("GPU exploded")

    engine = _BrokenEngine()
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    with pytest.raises(InferenceExecutionError):
        asyncio.run(service.generate(request_id="req-1", prompt="hello"))


def test_generate_supports_images_and_system_prompt():
    engine = FakeInferenceEngine()
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buffer, format="PNG")

    response = asyncio.run(
        service.generate(
            request_id="req-1",
            prompt="what is this?",
            system_prompt="You are concise.",
            raw_images=[RawImage(data=buffer.getvalue(), content_type="image/png")],
        )
    )

    assert response.request_id == "req-1"
    sent_request = engine.received_requests[0]
    assert sent_request.system_prompt == "You are concise."
    assert len(sent_request.images) == 1


def test_generate_success_logs_never_include_raw_prompt_or_response_text(caplog):
    engine = FakeInferenceEngine(generated_text="THE_SECRET_GENERATED_RESPONSE")
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    with caplog.at_level("INFO", logger="ML_logger"):
        asyncio.run(service.generate(request_id="req-log-1", prompt="THE_SECRET_PROMPT_TEXT"))

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "THE_SECRET_PROMPT_TEXT" not in log_text
    assert "THE_SECRET_GENERATED_RESPONSE" not in log_text
    assert "req-log-1" in log_text


def test_generate_failure_logs_include_request_id_but_not_prompt(caplog):
    class _BrokenEngine(FakeInferenceEngine):
        async def generate(self, request):
            raise RuntimeError("boom")

    engine = _BrokenEngine()
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    with caplog.at_level("INFO", logger="ML_logger"):
        with pytest.raises(InferenceExecutionError):
            asyncio.run(service.generate(request_id="req-log-2", prompt="ANOTHER_SECRET_PROMPT"))

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "ANOTHER_SECRET_PROMPT" not in log_text
    assert "req-log-2" in log_text
    assert "success=False" in log_text


def test_is_ready_reflects_engine_health():
    ready_service = InferenceService(
        engine=FakeInferenceEngine(ready=True), limits=_limits(), defaults=_defaults()
    )
    not_ready_service = InferenceService(
        engine=FakeInferenceEngine(ready=False), limits=_limits(), defaults=_defaults()
    )

    assert asyncio.run(ready_service.is_ready()) is True
    assert asyncio.run(not_ready_service.is_ready()) is False


def test_generate_allows_concurrent_requests_to_overlap_in_the_engine():
    """
    The active-requests lock must guard only the counter, never the
    `await self.engine.generate(...)` call itself -- otherwise vLLM's
    internal continuous batching would be serialized away at this layer
    (see claude.md's Concurrency section / todo.md Task 12).
    """

    class _SlowConcurrentEngine(FakeInferenceEngine):
        def __init__(self):
            super().__init__()
            self.in_flight = 0
            self.max_observed_overlap = 0

        async def generate(self, request):
            self.in_flight += 1
            self.max_observed_overlap = max(self.max_observed_overlap, self.in_flight)
            try:
                await asyncio.sleep(0.05)
                return await super().generate(request)
            finally:
                self.in_flight -= 1

    engine = _SlowConcurrentEngine()
    service = InferenceService(engine=engine, limits=_limits(), defaults=_defaults())

    async def _run_two_concurrent_requests():
        await asyncio.gather(
            service.generate(request_id="req-a", prompt="a"),
            service.generate(request_id="req-b", prompt="b"),
        )

    asyncio.run(_run_two_concurrent_requests())

    assert engine.max_observed_overlap == 2
