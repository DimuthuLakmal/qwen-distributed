from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from src.models.schema import QwenVLModelBundle
from src.config.schema import ModelConfig


def build_qwen_vl(
    config: ModelConfig,
) -> QwenVLModelBundle:
    """
    Build a Qwen2.5-VL model and its matching processor.

    This function is responsible only for model construction and
    model-level training configuration.

    FSDP wrapping, optimizer creation, and device placement should be
    handled by the training/distributed modules.
    """

    processor_kwargs = {
        "cache_dir": config.cache_dir,
    }
    # Qwen-VL's visual token count depends on image resolution.
    # These values are optional and should come from model/data config.
    if config.min_pixels != 0:
        processor_kwargs["min_pixels"] = config.min_pixels

    if config.max_pixels != 0:
        processor_kwargs["max_pixels"] = config.max_pixels

    # Load processor
    processor = AutoTokenizer.from_pretrained(
        config.model_id,
        **processor_kwargs,
    )

    if processor.pad_token_id is None:
        processor.pad_token_id = (
            processor.eos_token_id
        )

    # Model Loading
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        cache_dir=config.cache_dir,
        torch_dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        low_cpu_mem_usage=True,

        # Do not use device_map="auto" when training with FSDP.
        device_map=None,
    )

    # KV cache is useful for autoregressive inference but not training.
    model.config.use_cache = False

    # Configure Trainable Modules
    _configure_trainable_modules(
        model=model,
        freeze_vision=getattr(
            config,
            "freeze_vision",
            False,
        ),
        freeze_language_model=getattr(
            config,
            "freeze_language_model",
            False,
        ),
        freeze_vision_merger=getattr(
            config,
            "freeze_vision_merger",
            False,
        ),
    )


    # Gradient Checkpointing
    if getattr(config, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

        # Particularly useful if you later freeze embeddings or use PEFT.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # log parameters
    _print_parameter_summary(model)

    return QwenVLModelBundle(
        model=model,
        processor=processor,
    )


def _configure_trainable_modules(
    model: Qwen2_5_VLForConditionalGeneration,
    freeze_vision: bool,
    freeze_language_model: bool,
    freeze_vision_merger: bool,
) -> None:
    """
    Configure which Qwen-VL components should receive gradients.
    """

    if freeze_vision:
        model.model.visual.requires_grad_(False)

        if not freeze_vision_merger:
            model.model.visual.merger.requires_grad_(True)

    if freeze_language_model:
        model.model.language_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)


def _print_parameter_summary(
    model: torch.nn.Module,
) -> None:
    """
    Print total and trainable parameter counts.
    """

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_params = total_params - trainable_params

    trainable_ratio = (
        100.0 * trainable_params / total_params
        if total_params > 0
        else 0.0
    )

    print(
        "\nModel parameter summary\n"
        "-----------------------\n"
        f"Total parameters     : {total_params:,}\n"
        f"Trainable parameters : {trainable_params:,}\n"
        f"Frozen parameters    : {frozen_params:,}\n"
        f"Trainable percentage : {trainable_ratio:.2f}%\n"
    )