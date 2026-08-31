"""
Unit tests for VLLMEngine using a mocked AsyncLLM.

No real vLLM engine or GPU is started here -- these tests only exercise the
engine's prompt/sampling-params construction, response assembly, health
check, and cancellation logic. Requires the `vllm` package to be importable
(run in the `vllm` conda env).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.config.schema import VLLMConfig
from src.inference.engines.vllm import VLLMEngine
from src.inference.exception import InvalidGenerationOptionsError
from src.inference.schema import GenerationConfig, ImageInput, InferenceRequest


def _vllm_config(**overrides) -> VLLMConfig:
    defaults = dict(
        model_id="fake/model/path",
        cache_dir=None,
        dtype="bfloat16",
        data_parallel_size=2,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": 4, "video": 0},
        mm_processor_kwargs={"max_pixels": 1003520},
    )
    defaults.update(overrides)
    return VLLMConfig(**defaults)


@dataclass
class _FakeCompletionOutput:
    text: str
    token_ids: list


@dataclass
class _FakeRequestOutput:
    prompt_token_ids: list
    outputs: list
    finished: bool = True


class _FakeAsyncLLM:
    def __init__(self):
        self.aborted_request_ids: list[str] = []
        self.shutdown_called = False
        self.health_raises: Exception | None = None
        self.last_prompt = None
        self.last_sampling_params = None
        self.last_request_id = None

    async def generate(self, prompt, sampling_params, request_id):
        self.last_prompt = prompt
        self.last_sampling_params = sampling_params
        self.last_request_id = request_id
        yield _FakeRequestOutput(
            prompt_token_ids=[10, 11, 12, 13],
            outputs=[_FakeCompletionOutput(text="a cat", token_ids=[1])],
            finished=False,
        )
        yield _FakeRequestOutput(
            prompt_token_ids=[10, 11, 12, 13],
            outputs=[_FakeCompletionOutput(text="a cat on a mat", token_ids=[1, 2, 3])],
            finished=True,
        )

    async def check_health(self):
        if self.health_raises:
            raise self.health_raises

    async def abort(self, request_id):
        self.aborted_request_ids.append(request_id)

    def shutdown(self, timeout=None):
        self.shutdown_called = True


class _HangingAsyncLLM(_FakeAsyncLLM):
    async def generate(self, prompt, sampling_params, request_id):
        self.last_prompt = prompt
        self.last_request_id = request_id
        yield _FakeRequestOutput(
            prompt_token_ids=[1], outputs=[_FakeCompletionOutput(text="", token_ids=[])], finished=False
        )
        await asyncio.sleep(10)


def _build_engine(fake_engine: _FakeAsyncLLM, vllm_config=None) -> VLLMEngine:
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "RENDERED_PROMPT"

    with patch(
        "src.inference.engines.vllm.AsyncLLM.from_engine_args", return_value=fake_engine
    ), patch(
        "src.inference.engines.vllm.AutoTokenizer.from_pretrained", return_value=fake_tokenizer
    ):
        engine = VLLMEngine(
            vllm_config or _vllm_config(),
            model_version="test-version",
            data_parallel_size=2,
        )

    engine._tokenizer_mock = fake_tokenizer
    return engine


def test_engine_args_built_from_config():
    fake_engine = _FakeAsyncLLM()
    captured = {}

    def _capture_from_engine_args(engine_args, *args, **kwargs):
        captured["engine_args"] = engine_args
        return fake_engine

    with patch(
        "src.inference.engines.vllm.AsyncLLM.from_engine_args",
        side_effect=_capture_from_engine_args,
    ), patch(
        "src.inference.engines.vllm.AutoTokenizer.from_pretrained", return_value=MagicMock()
    ):
        VLLMEngine(
            _vllm_config(
                model_id="Qwen/Qwen3.5-9B",
                cache_dir="/tmp/llm_models",
                data_parallel_size=3,
                gpu_memory_utilization=0.7,
                max_model_len=2048,
                max_num_seqs=8,
            ),
            model_version="v1",
            data_parallel_size=3,
        )

    engine_args = captured["engine_args"]
    assert engine_args.model == "Qwen/Qwen3.5-9B"
    assert engine_args.download_dir == "/tmp/llm_models"
    assert engine_args.dtype == "bfloat16"
    assert engine_args.tensor_parallel_size == 1
    assert engine_args.data_parallel_size == 3
    assert engine_args.data_parallel_backend == "mp"
    assert engine_args.gpu_memory_utilization == 0.7
    assert engine_args.max_model_len == 2048
    assert engine_args.max_num_seqs == 8
    assert engine_args.limit_mm_per_prompt == {"image": 4, "video": 0}
    assert engine_args.mm_processor_kwargs == {"max_pixels": 1003520}


def test_generate_renders_prompt_and_maps_response():
    fake_engine = _FakeAsyncLLM()
    engine = _build_engine(fake_engine)

    request = InferenceRequest(
        request_id="req-1",
        prompt="what is in this image?",
        images=[ImageInput(image="FAKE_PIL_IMAGE")],
        generation=GenerationConfig(max_new_tokens=32, do_sample=False, num_beams=1),
    )

    response = asyncio.run(engine.generate(request))

    assert response.request_id == "req-1"
    assert response.generated_text == "a cat on a mat"
    assert response.model_version == "test-version"
    assert response.input_token_count == 4
    assert response.output_token_count == 3

    assert fake_engine.last_prompt["prompt"] == "RENDERED_PROMPT"
    assert fake_engine.last_prompt["multi_modal_data"] == {"image": ["FAKE_PIL_IMAGE"]}
    engine._tokenizer_mock.apply_chat_template.assert_called_once()
    _, kwargs = engine._tokenizer_mock.apply_chat_template.call_args
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True


def test_generate_omits_multi_modal_data_when_no_images():
    fake_engine = _FakeAsyncLLM()
    engine = _build_engine(fake_engine)

    request = InferenceRequest(request_id="req-text-only", prompt="hello")

    asyncio.run(engine.generate(request))

    assert fake_engine.last_prompt.get("multi_modal_data") is None


def test_generate_maps_sampling_params_greedy_when_not_sampling():
    fake_engine = _FakeAsyncLLM()
    engine = _build_engine(fake_engine)

    request = InferenceRequest(
        request_id="req-2",
        prompt="describe",
        generation=GenerationConfig(max_new_tokens=16, do_sample=False, temperature=0.9),
    )

    asyncio.run(engine.generate(request))

    assert fake_engine.last_sampling_params.max_tokens == 16
    assert fake_engine.last_sampling_params.temperature == 0.0


def test_generate_maps_sampling_params_with_temperature_when_sampling():
    fake_engine = _FakeAsyncLLM()
    engine = _build_engine(fake_engine)

    request = InferenceRequest(
        request_id="req-3",
        prompt="describe",
        generation=GenerationConfig(max_new_tokens=16, do_sample=True, temperature=0.7),
    )

    asyncio.run(engine.generate(request))

    assert fake_engine.last_sampling_params.temperature == 0.7


def test_generate_rejects_num_beams_greater_than_one_without_calling_engine():
    fake_engine = _FakeAsyncLLM()
    engine = _build_engine(fake_engine)

    request = InferenceRequest(
        request_id="req-4",
        prompt="describe",
        generation=GenerationConfig(num_beams=2),
    )

    with pytest.raises(InvalidGenerationOptionsError):
        asyncio.run(engine.generate(request))

    assert fake_engine.last_request_id is None


def test_health_true_when_check_health_succeeds():
    fake_engine = _FakeAsyncLLM()
    engine = _build_engine(fake_engine)

    assert asyncio.run(engine.health()) is True


def test_health_false_when_check_health_raises():
    fake_engine = _FakeAsyncLLM()
    fake_engine.health_raises = RuntimeError("engine dead")
    engine = _build_engine(fake_engine)

    assert asyncio.run(engine.health()) is False


def test_close_calls_engine_shutdown_and_marks_not_ready():
    fake_engine = _FakeAsyncLLM()
    engine = _build_engine(fake_engine)

    asyncio.run(engine.close())

    assert fake_engine.shutdown_called is True
    assert asyncio.run(engine.health()) is False


def test_generate_aborts_engine_request_on_cancellation():
    fake_engine = _HangingAsyncLLM()
    engine = _build_engine(fake_engine)

    request = InferenceRequest(request_id="req-cancel", prompt="describe")

    async def _run_and_cancel():
        task = asyncio.ensure_future(engine.generate(request))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run_and_cancel())

    # The vLLM-internal request id is a fresh UUID, deliberately independent
    # of the service-level request_id (see VLLMEngine.generate) -- abort()
    # must be called with that same internal id.
    assert fake_engine.aborted_request_ids == [fake_engine.last_request_id]