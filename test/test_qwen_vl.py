"""
Smoke test for the Qwen-VL model builder: build_qwen_vl -> QwenVLModelBundle,
wired to the real config (model.yaml) via ConfigurationManager.

NOTE: This downloads/loads the configured model (Qwen3-VL-8B-Instruct) and
builds it with low_cpu_mem_usage=True -- run this on a compute node
(srun/sbatch), not the shared login node.

Run from the repo root with:
    python -m pytest test/test_qwen_vl.py -v -s
or directly:
    python test/test_qwen_vl.py
"""

from pathlib import Path

from src.config.configuration import ConfigurationManager
from src.models.qwen_vl import build_qwen_vl

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _build_model_config():
    config_manager = ConfigurationManager(
        data_config_path=CONFIGS_DIR / "data.yaml",
        model_config_path=CONFIGS_DIR / "model.yaml",
        training_config_path=CONFIGS_DIR / "training.yaml",
    )

    return config_manager.get_model_config()


def test_build_qwen_vl_returns_model_and_processor():
    model_config = _build_model_config()

    bundle = build_qwen_vl(model_config)

    assert bundle.model is not None
    assert bundle.processor is not None
    assert bundle.processor.pad_token_id is not None
    assert bundle.model.config.use_cache is False


def test_build_qwen_vl_respects_freeze_flags():
    model_config = _build_model_config()

    bundle = build_qwen_vl(model_config)
    model = bundle.model

    if model_config.freeze_vision:
        assert not any(
            p.requires_grad for p in model.model.visual.parameters()
        ) or any(
            p.requires_grad for p in model.model.visual.merger.parameters()
        )

    if model_config.freeze_language_model:
        assert not any(
            p.requires_grad for p in model.model.language_model.parameters()
        )
        assert not model.lm_head.weight.requires_grad


if __name__ == "__main__":
    test_build_qwen_vl_returns_model_and_processor()
    test_build_qwen_vl_respects_freeze_flags()
    print("\nAll qwen_vl smoke tests passed.")
