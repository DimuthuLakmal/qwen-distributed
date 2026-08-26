"""
Unit tests for inference configuration loading via ConfigurationManager.

Uses the real configs/inference.yaml plus small temporary yaml files for
the invalid-configuration cases, so no network/model access is required.
"""

from pathlib import Path

import pytest

from src.config.configuration import ConfigurationManager

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _config_manager() -> ConfigurationManager:
    return ConfigurationManager(inference_config_path=CONFIGS_DIR / "inference.yaml")


def test_get_inference_service_config():
    config_manager = _config_manager()
    service_config = config_manager.get_inference_service_config()

    assert service_config.host
    assert 0 < service_config.port < 65536
    assert service_config.model_version


def test_get_inference_model_config():
    config_manager = _config_manager()
    model_config = config_manager.get_inference_model_config()

    assert model_config.model_path
    assert model_config.device
    import torch

    assert model_config.dtype == torch.bfloat16


def test_get_generation_defaults_config():
    config_manager = _config_manager()
    generation_config = config_manager.get_generation_defaults_config()

    assert generation_config.max_new_tokens > 0
    assert generation_config.num_beams >= 1


def test_get_inference_limits_config():
    config_manager = _config_manager()
    limits_config = config_manager.get_inference_limits_config()

    assert limits_config.max_prompt_chars > 0
    assert limits_config.max_images > 0
    assert limits_config.max_image_bytes > 0
    assert limits_config.max_image_pixels > 0
    assert limits_config.max_new_tokens > 0


def test_getters_fail_clearly_when_config_not_loaded():
    config_manager = ConfigurationManager()

    with pytest.raises(RuntimeError, match="inference_config_path"):
        config_manager.get_inference_service_config()


def test_generation_defaults_exceeding_limits_fails_at_load(tmp_path):
    invalid_yaml = tmp_path / "inference.yaml"
    invalid_yaml.write_text(
        """
service:
  host: "0.0.0.0"
  port: 8000
  model_version: "test"

model:
  model_path: "results/final"
  device: "cpu"
  dtype: "bfloat16"
  attn_implementation: null
  min_pixels: 0
  max_pixels: 0

generation:
  do_sample: false
  max_new_tokens: 1000
  num_beams: 1
  temperature: 1.0

limits:
  max_prompt_chars: 4000
  max_images: 4
  max_image_bytes: 8000000
  max_image_pixels: 20000000
  max_new_tokens: 512
"""
    )

    config_manager = ConfigurationManager(inference_config_path=invalid_yaml)

    with pytest.raises(ValueError, match="max_new_tokens"):
        config_manager.get_generation_defaults_config()


def test_existing_training_config_paths_still_work():
    """Regression: making all ConfigurationManager args optional must not
    break the existing training/eval call sites that pass them explicitly."""

    config_manager = ConfigurationManager(
        data_config_path=CONFIGS_DIR / "data.yaml",
        model_config_path=CONFIGS_DIR / "model.yaml",
        training_config_path=CONFIGS_DIR / "training.yaml",
        tracking_config_path=CONFIGS_DIR / "tracking.yaml",
    )

    assert config_manager.get_model_config().model_id
    assert config_manager.get_training_config().num_epochs > 0
