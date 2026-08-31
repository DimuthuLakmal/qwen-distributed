"""
Unit tests for the backend factory (src.inference.engines.factory).

Everything here runs against mocks/subprocesses -- no real Transformers
model or vLLM engine is loaded.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.schema import EngineConfig, InferenceTransformersModelConfig, VLLMConfig
from src.inference.engines.factory import build_engine

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _fake_vllm_engine_module(mock_engine_cls: MagicMock) -> types.ModuleType:
    """
    A fake `src.inference.engines.vllm` module, injected into `sys.modules`
    so `build_engine`'s deferred `from src.inference.engines.vllm import
    VLLMEngine` never actually imports the real `vllm` package -- these
    tests must pass in an environment where `vllm` isn't installed at all.
    """

    module = types.ModuleType("src.inference.engines.vllm")
    module.VLLMEngine = mock_engine_cls
    return module


def _model_config() -> InferenceTransformersModelConfig:
    return InferenceTransformersModelConfig(model_path="fake/path", device="cpu")


def test_build_engine_dispatches_to_transformers():
    with patch(
        "src.inference.engines.factory.TransformersQwenVLEngine", return_value="fake-transformers-engine"
    ) as mock_engine_cls:
        engine = build_engine(
            EngineConfig(type="transformers"), _model_config(), model_version="v1"
        )

    mock_engine_cls.assert_called_once_with(_model_config(), model_version="v1", profiler_config=None)
    assert engine == "fake-transformers-engine"


def test_build_engine_unknown_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown engine.type"):
        build_engine(EngineConfig(type="not-a-backend"), _model_config(), model_version="v1")


def test_build_engine_dispatches_to_vllm_with_resolved_data_parallel_size():
    vllm_config = VLLMConfig(data_parallel_size=3)
    mock_engine_cls = MagicMock(return_value="fake-vllm-engine")

    with patch.dict(sys.modules, {"src.inference.engines.vllm": _fake_vllm_engine_module(mock_engine_cls)}):
        engine = build_engine(
            EngineConfig(type="vllm", vllm=vllm_config), _model_config(), model_version="v1"
        )

    mock_engine_cls.assert_called_once()
    _, kwargs = mock_engine_cls.call_args
    assert kwargs["data_parallel_size"] == 3
    assert engine == "fake-vllm-engine"


def test_build_engine_resolves_auto_data_parallel_size_from_visible_gpus():
    vllm_config = VLLMConfig(data_parallel_size="auto")
    mock_engine_cls = MagicMock(return_value="fake-vllm-engine")

    with patch.dict(
        sys.modules, {"src.inference.engines.vllm": _fake_vllm_engine_module(mock_engine_cls)}
    ), patch("src.utils.dist_utils.visible_gpu_count", return_value=2):
        build_engine(EngineConfig(type="vllm", vllm=vllm_config), _model_config(), model_version="v1")

    _, kwargs = mock_engine_cls.call_args
    assert kwargs["data_parallel_size"] == 2


def test_build_engine_auto_data_parallel_size_fails_clearly_with_no_gpus():
    vllm_config = VLLMConfig(data_parallel_size="auto")

    with patch("src.utils.dist_utils.visible_gpu_count", return_value=0):
        with pytest.raises(RuntimeError, match="no visible GPUs"):
            build_engine(EngineConfig(type="vllm", vllm=vllm_config), _model_config(), model_version="v1")


def test_build_engine_transformers_backend_never_imports_vllm_module():
    """
    Regression for claude.md rule 5 ("load only the selected backend during
    startup"): building a transformers-typed engine must never trigger
    `import vllm`, even transitively. Run in a subprocess so the assertion
    is meaningful regardless of what other test modules in this same pytest
    session may have already imported.
    """

    script = (
        "import sys\n"
        "from unittest.mock import patch\n"
        "from src.config.schema import EngineConfig\n"
        "from src.inference.engines import factory\n"
        "with patch('src.inference.engines.factory.TransformersQwenVLEngine', return_value='fake'):\n"
        "    factory.build_engine(EngineConfig(type='transformers'), model_config=None, model_version='v1')\n"
        "assert 'vllm' not in sys.modules, 'transformers backend must never import vllm'\n"
        "print('OK')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout