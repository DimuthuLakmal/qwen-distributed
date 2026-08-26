from __future__ import annotations

from typing import Any

import torch


def move_to_device(value: Any, device: torch.device) -> Any:
    """
    Recursively move tensors onto `device`. Keeps non-tensor metadata
    unchanged. Shared by the training loop and the evaluator so both
    move batches to the local GPU the same way.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)

    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}

    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]

    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)

    return value


def count_supervised_tokens(labels: torch.Tensor) -> torch.Tensor:
    """
    Count targets contributing to causal-LM loss.
    Causal LM predicts:
        token[t + 1]
    from:
        token[t]
    so the first label position is not used by the shifted loss.
    """

    if labels.ndim != 2:
        raise ValueError(
            "Expected labels with shape [B, L], "
            f"got {tuple(labels.shape)}."
        )

    return (labels[:, 1:] != -100).sum().double()
