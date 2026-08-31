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


def test_get_inference_transformers_model_config():
    config_manager = _config_manager()
    model_config = config_manager.get_inference_transformers_model_config()

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

transformers:
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


def test_get_engine_config_defaults_to_transformers():
    config_manager = _config_manager()
    engine_config = config_manager.get_engine_config()

    assert engine_config.type == "transformers"
    assert engine_config.vllm is None


def _write_inference_yaml(tmp_path: Path, engine_block: str) -> Path:
    yaml_path = tmp_path / "inference.yaml"
    yaml_path.write_text(
        f"""
service:
  host: "0.0.0.0"
  port: 8000
  model_version: "test"

{engine_block}

transformers:
  model_path: "results/final"
  device: "cpu"
  dtype: "bfloat16"
  attn_implementation: null
  min_pixels: 0
  max_pixels: 0

generation:
  do_sample: false
  max_new_tokens: 128
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
    return yaml_path


def test_get_engine_config_unknown_type_rejected(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, 'engine:\n  type: "not-a-backend"')
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="Unknown engine.type"):
        config_manager.get_engine_config()


def test_get_engine_config_vllm_requires_vllm_section(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, 'engine:\n  type: "vllm"')
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="engine.vllm"):
        config_manager.get_engine_config()


def _valid_vllm_block(**overrides) -> str:
    fields = {
        "model_id": "Qwen/Qwen3.5-9B",
        "cache_dir": "/tmp/llm_models",
        "dtype": "bfloat16",
        "data_parallel_size": 2,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "max_num_seqs": 16,
    }
    fields.update(overrides)
    return (
        'engine:\n'
        '  type: "vllm"\n'
        '  vllm:\n'
        f'    model_id: "{fields["model_id"]}"\n'
        f'    cache_dir: "{fields["cache_dir"]}"\n'
        f'    dtype: "{fields["dtype"]}"\n'
        f'    data_parallel_size: {fields["data_parallel_size"]}\n'
        f'    tensor_parallel_size: {fields["tensor_parallel_size"]}\n'
        f'    gpu_memory_utilization: {fields["gpu_memory_utilization"]}\n'
        f'    max_model_len: {fields["max_model_len"]}\n'
        f'    max_num_seqs: {fields["max_num_seqs"]}\n'
        '    limit_mm_per_prompt:\n'
        '      image: 4\n'
        '      video: 0\n'
    )


def test_get_engine_config_valid_vllm_section(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block())
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    engine_config = config_manager.get_engine_config()

    assert engine_config.type == "vllm"
    assert engine_config.vllm.model_id == "Qwen/Qwen3.5-9B"
    assert engine_config.vllm.cache_dir == "/tmp/llm_models"
    assert engine_config.vllm.dtype == "bfloat16"
    assert engine_config.vllm.data_parallel_size == 2
    assert engine_config.vllm.tensor_parallel_size == 1
    assert engine_config.vllm.limit_mm_per_prompt == {"image": 4, "video": 0}


def test_get_engine_config_vllm_requires_model_id(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block(model_id=""))
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="model_id"):
        config_manager.get_engine_config()


def test_get_engine_config_vllm_data_parallel_size_auto_is_accepted(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block(data_parallel_size='"auto"'))
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    engine_config = config_manager.get_engine_config()

    assert engine_config.vllm.data_parallel_size == "auto"


def test_get_engine_config_rejects_tensor_parallel_size_other_than_one(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block(tensor_parallel_size=2))
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        config_manager.get_engine_config()


def test_get_engine_config_rejects_invalid_data_parallel_size(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block(data_parallel_size=0))
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="data_parallel_size"):
        config_manager.get_engine_config()


def test_get_engine_config_rejects_out_of_range_gpu_memory_utilization(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block(gpu_memory_utilization=1.5))
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        config_manager.get_engine_config()


def test_get_engine_config_rejects_non_positive_max_model_len(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block(max_model_len=0))
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="max_model_len"):
        config_manager.get_engine_config()


def test_get_engine_config_rejects_non_positive_max_num_seqs(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block(max_num_seqs=0))
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="max_num_seqs"):
        config_manager.get_engine_config()


def test_get_engine_config_rejects_image_limit_below_public_limit(tmp_path):
    yaml_path = _write_inference_yaml(tmp_path, _valid_vllm_block())
    # Public limits.max_images is 4 in _write_inference_yaml's fixed template;
    # shrink the vLLM-side image limit below that to trigger the cross-check.
    content = yaml_path.read_text().replace("image: 4\n      video: 0", "image: 2\n      video: 0")
    yaml_path.write_text(content)
    config_manager = ConfigurationManager(inference_config_path=yaml_path)

    with pytest.raises(ValueError, match="limit_mm_per_prompt.image"):
        config_manager.get_engine_config()


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
