"""
Unit tests for TransformersQwenVLEngine using a mocked model/processor.

No real weights or GPU are loaded here -- these tests only exercise the
engine's message-building, tensor-slicing, and error-handling logic.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config.schema import InferenceModelConfig
from src.inference.engines.transformers import TransformersQwenVLEngine
from src.inference.schema import GenerationConfig, InferenceRequest


def _model_config(**overrides) -> InferenceModelConfig:
    defaults = dict(
        model_path="fake/model/path",
        device="cpu",
        dtype=torch.float32,
        attn_implementation=None,
        min_pixels=0,
        max_pixels=0,
    )
    defaults.update(overrides)
    return InferenceModelConfig(**defaults)


class _FakeProcessor:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["return_tensors"] == "pt"
        # 3-token "prompt".
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ["a cat sitting on a mat"] * token_ids.shape[0]


class _FakeModel:
    def __init__(self):
        self.eval_called = False
        self.generate_kwargs = None

    def eval(self):
        self.eval_called = True

    def to(self, device):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        prompt_len = kwargs["input_ids"].shape[1]
        # Append 2 new tokens beyond the prompt.
        new_tokens = torch.tensor([[99, 98]])
        return torch.cat([kwargs["input_ids"], new_tokens], dim=1)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value


def _build_engine_with_fakes() -> TransformersQwenVLEngine:
    engine = TransformersQwenVLEngine.__new__(TransformersQwenVLEngine)
    engine.config = _model_config()
    engine.model_version = "test-version"
    engine.device = torch.device("cpu")
    engine.processor = _FakeProcessor()
    engine.model = _FakeModel()
    engine._ready = True
    return engine


def test_generate_slices_prompt_tokens_and_returns_canonical_response():
    engine = _build_engine_with_fakes()
    request = InferenceRequest(
        request_id="req-1",
        prompt="what is in this image?",
        generation=GenerationConfig(max_new_tokens=16, num_beams=1, do_sample=False),
    )

    response = engine.generate(request)

    assert response.request_id == "req-1"
    assert response.generated_text == "a cat sitting on a mat"
    assert response.model_version == "test-version"
    assert response.input_token_count == 3
    assert response.output_token_count == 2
    assert response.latency_seconds >= 0

    # No sampling requested -> temperature must not be forwarded to generate().
    assert "temperature" not in engine.model.generate_kwargs
    assert engine.model.generate_kwargs["do_sample"] is False
    assert engine.model.generate_kwargs["max_new_tokens"] == 16


def test_generate_forwards_temperature_only_when_sampling():
    engine = _build_engine_with_fakes()
    request = InferenceRequest(
        request_id="req-2",
        prompt="describe",
        generation=GenerationConfig(do_sample=True, temperature=0.7, max_new_tokens=8),
    )

    engine.generate(request)

    assert engine.model.generate_kwargs["do_sample"] is True
    assert engine.model.generate_kwargs["temperature"] == 0.7


def test_health_reflects_ready_state():
    engine = _build_engine_with_fakes()
    assert engine.health() is True

    engine._ready = False
    assert engine.health() is False


def test_load_wraps_processor_failure_in_clear_runtime_error():
    with patch(
        "src.inference.engines.transformers.AutoTokenizer.from_pretrained",
        side_effect=OSError("not found"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load inference processor"):
            TransformersQwenVLEngine(_model_config(), model_version="v1")


def test_load_wraps_model_failure_in_clear_runtime_error():
    fake_tokenizer = MagicMock(pad_token_id=1)

    with patch(
        "src.inference.engines.transformers.AutoTokenizer.from_pretrained",
        return_value=fake_tokenizer,
    ), patch(
        "src.inference.engines.transformers.AutoModelForImageTextToText.from_pretrained",
        side_effect=OSError("bad weights"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load inference model"):
            TransformersQwenVLEngine(_model_config(), model_version="v1")


def test_load_sets_eval_mode_use_cache_and_engine_becomes_ready():
    fake_tokenizer = MagicMock(pad_token_id=1)
    fake_model = _FakeModel()
    fake_model.config = MagicMock(use_cache=False)

    with patch(
        "src.inference.engines.transformers.AutoTokenizer.from_pretrained",
        return_value=fake_tokenizer,
    ), patch(
        "src.inference.engines.transformers.AutoModelForImageTextToText.from_pretrained",
        return_value=fake_model,
    ):
        engine = TransformersQwenVLEngine(_model_config(), model_version="v1")

    assert engine.health() is True
    assert fake_model.eval_called is True
    assert fake_model.config.use_cache is True
