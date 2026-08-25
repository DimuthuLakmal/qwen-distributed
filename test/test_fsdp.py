"""
Smoke test for FSDP2 wrapping: build_qwen_vl -> apply_fsdp, wired to the
real configs (model.yaml/training.yaml) via ConfigurationManager.

NOTE: This downloads/loads the configured model (Qwen3-VL-8B-Instruct) and
requires torch.distributed (nccl) + at least one GPU -- run this on a
compute node via torchrun, not the shared login node:

    torchrun --nproc_per_node=1 test/test_fsdp.py

Use --nproc_per_node=2 (or more) to also exercise real cross-rank sharding.
Also runnable via the "Test FSDP" VS Code launch config.
"""

from pathlib import Path

import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from src.config.configuration import ConfigurationManager
from src.models.qwen_vl import build_qwen_vl
from src.training.fsdp import apply_fsdp
from src.utils.dist_utils import cleanup_distributed, initialize_distributed

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _build_config_manager() -> ConfigurationManager:
    return ConfigurationManager(
        data_config_path=CONFIGS_DIR / "data.yaml",
        model_config_path=CONFIGS_DIR / "model.yaml",
        training_config_path=CONFIGS_DIR / "training.yaml",
    )


def _build_sharded_model():
    config_manager = _build_config_manager()
    model_config = config_manager.get_model_config()
    fsdp_config = config_manager.get_fsdp_config()

    bundle = build_qwen_vl(model_config)
    model, context = apply_fsdp(bundle.model, fsdp_config)

    return model, context, fsdp_config


def test_apply_fsdp_wraps_root_and_submodules():
    model, context, fsdp_config = _build_sharded_model()

    assert isinstance(model, FSDPModule)

    language_layers = model.model.language_model.layers
    vision_blocks = model.model.visual.blocks

    if fsdp_config.shard_language_layers:
        assert all(isinstance(layer, FSDPModule) for layer in language_layers)

    if fsdp_config.shard_vision_blocks:
        assert all(isinstance(block, FSDPModule) for block in vision_blocks)

    assert context.num_language_layers == len(language_layers)
    assert context.num_vision_blocks == len(vision_blocks)
    assert context.mesh.size() == dist.get_world_size()

    print("\nworld_size:", dist.get_world_size())
    print("num_language_layers:", context.num_language_layers)
    print("num_vision_blocks:", context.num_vision_blocks)


def test_apply_fsdp_shards_parameters_as_dtensor():
    model, _, _ = _build_sharded_model()

    sharded_params = [p for p in model.parameters() if isinstance(p, DTensor)]
    assert len(sharded_params) > 0

    sample = sharded_params[0]
    print("\nsample sharded param placements:", sample.placements)
    print("sample sharded param device_mesh:", sample.device_mesh)


if __name__ == "__main__":
    initialize_distributed()
    try:
        test_apply_fsdp_wraps_root_and_submodules()
        test_apply_fsdp_shards_parameters_as_dtensor()
        print("\nAll FSDP smoke tests passed.")
    finally:
        cleanup_distributed()
