from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)

from src.training.schema import FSDPContext
from src.config.schema import FSDPConfig


def apply_fsdp(model: nn.Module, config: FSDPConfig,) -> tuple[nn.Module, FSDPContext]:
    """
    Apply PyTorch FSDP2 sharding to Qwen2.5-VL.
    Sharding strategy:
        1. shard each vision transformer block
        2. shard each language decoder layer
        3. shard the root model

    FSDP must be applied bottom-up so that each transformer layer
    becomes an independent communication group.
    """

    mesh = _build_device_mesh()
    mp_policy = _build_mixed_precision_policy(config)

    fsdp_kwargs = {
        "mesh": mesh,
        "reshard_after_forward": config.reshard_after_forward,
    }

    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy

    if getattr(config, "cpu_offload", False):
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(
            pin_memory=getattr(
                config,
                "cpu_offload_pin_memory",
                True,
            )
        )

    # Sharding vision blocks
    vision_blocks = model.model.visual.blocks

    if config.shard_vision_blocks:
        for block in vision_blocks:
            fully_shard(block, **fsdp_kwargs,)

    # Sharding Language model
    language_layers = model.model.language_model.layers

    if config.shard_language_layers:
        for layer in language_layers:
            fully_shard(layer, **fsdp_kwargs,)

    # shard root model to accomodate sharding of rest of the model components
    fully_shard(model, **fsdp_kwargs,)

    context = FSDPContext(
        mesh=mesh,
        num_language_layers=len(language_layers),
        num_vision_blocks=len(vision_blocks),
    )

    return model, context


def _build_device_mesh() -> DeviceMesh:
    """
    Construct a 1-D data-parallel mesh spanning all distributed ranks.

    Example with 8 GPUs:
        mesh = [0, 1, 2, 3, 4, 5, 6, 7]
    Parameters are sharded across this dimension.
    """

    world_size = dist.get_world_size()

    return init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )


def _build_mixed_precision_policy(config: FSDPConfig,) -> MixedPrecisionPolicy | None:
    """
    Build FSDP2's module-level mixed precision policy.

    Typical configuration:
        parameter compute: BF16
        gradient reduction: FP32
    """

    if not getattr(config, "mixed_precision", True):
        return None


    return MixedPrecisionPolicy(
        param_dtype=config.param_dtype,
        reduce_dtype=config.reduce_dtype,
        output_dtype=config.output_dtype,
        cast_forward_inputs=True,
    )


def set_gradient_sync(model: nn.Module, enabled: bool,) -> None:
    """
    Enable/disable FSDP gradient communication.

    Useful for gradient accumulation.

    For example, with accumulation_steps=4:

        microbatch 0 -> sync=False
        microbatch 1 -> sync=False
        microbatch 2 -> sync=False
        microbatch 3 -> sync=True
    """

    model.set_requires_gradient_sync(enabled, recurse=True,)