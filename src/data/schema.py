from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class Conversation:
    messages: list[dict[str, Any]]


@dataclass
class ProcessedSample:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None

    pixel_values_videos: torch.Tensor | None
    video_grid_thw: torch.Tensor | None