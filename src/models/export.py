from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from src.utils.dist_utils import get_rank, is_distributed, barrier
from src.utils.logger import logger


def export_huggingface_model(
    model: nn.Module,
    processor: Any,
    output_dir: str | Path,
    *,
    export_dtype: str | torch.dtype | None = "bfloat16",
    max_shard_size: str = "5GB",
) -> Path:
    """
    Export an FSDP2-sharded Hugging Face model into a normal
    `save_pretrained()` directory.
    All distributed ranks MUST call this function.
    Parameters
    ----------
    model:
        FSDP2-sharded Hugging Face model.

    processor:
        Matching Hugging Face processor.

    output_dir:
        Destination directory.

    export_dtype:
        Floating-point dtype used for exported parameters.

        Useful when training keeps FP32 parameter shards but inference
        should use BF16.

        Set to None to preserve the original parameter dtype.

    max_shard_size:
        Maximum Hugging Face safetensors shard size.

    Returns
    -------
    Path
        Export directory.
    """

    if not is_distributed():
        raise RuntimeError("System is not distributed") 

    output_dir = Path(output_dir)

    rank = get_rank()

    # 1. Synchronize before export
    barrier()

    if rank == 0:
        logger.info("Gathering full FSDP2 model state for Hugging Face export...")

    # Gather the FSDP2 DTensor state dict
    # full_state_dict=True
    #     Convert DTensor/sharded parameters into normal tensors.
    # cpu_offload=True
    #     Move gathered tensors to CPU.
    # Importantly, when both are True, only rank 0 receives the full
    # state dict. Other ranks receive an empty dictionary.
    state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    # Non-zero ranks wait while rank 0 serializes
    if rank != 0:
        barrier()
        return output_dir

    if not state_dict:
        raise RuntimeError("Rank 0 received an empty state dict during export.")

    logger.info("Full model state gathered | tensors=%d", len(state_dict),)

    # Optionally convert export precision
    target_dtype = None if export_dtype is None else export_dtype

    if target_dtype is not None:
        logger.info("Converting exported floating-point tensors to %s", target_dtype,)

        _convert_state_dict_dtype_inplace(
            state_dict,
            target_dtype,
        )

    output_dir.mkdir(parents=True,exist_ok=True,)

    # Update exported model config dtype
    # This is metadata only. The actual dtype is determined by the tensors in state_dict.
    if target_dtype is not None:
        _update_config_dtype(model, target_dtype,)

    # Save Hugging Face model
    logger.info("Saving Hugging Face model to %s", output_dir,)

    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        max_shard_size=max_shard_size,
        is_main_process=True,
    )

    # Save processor
    processor.save_pretrained(output_dir)

    # Free large CPU state dict
    del state_dict
    gc.collect()

    logger.info("Hugging Face export complete: %s", output_dir,)

    # Release remaining ranks
    barrier()

    return output_dir


def _convert_state_dict_dtype_inplace(state_dict: dict[str, Any],dtype: torch.dtype,) -> None:
    """
    Convert floating-point tensors in a state dict to the desired dtype.
    Integer tensors/buffers remain unchanged.
    Conversion is performed one tensor at a time to avoid creating a
    second complete copy of the model in memory.
    """

    for name in list(state_dict.keys()):
        tensor = state_dict[name]

        if not isinstance(tensor, torch.Tensor,):
            continue

        if not tensor.is_floating_point():
            continue

        if tensor.dtype == dtype:
            continue

        state_dict[name] = tensor.to(dtype=dtype)


def _update_config_dtype(model: nn.Module, dtype: torch.dtype,) -> None:
    """
    Update Hugging Face configuration metadata where supported.

    Different Transformers versions expose the dtype field slightly
    differently, so handle both representations conservatively.
    """
    config = getattr(model, "config", None,)

    if config is None:
        return

    # Transformers 4.x commonly exposes torch_dtype.
    if hasattr(config, "torch_dtype",):
        config.torch_dtype = dtype

    # Newer Transformers versions may expose dtype directly.
    if hasattr(config, "dtype",):
        try:
            config.dtype = dtype
        except (AttributeError, TypeError):
            pass
