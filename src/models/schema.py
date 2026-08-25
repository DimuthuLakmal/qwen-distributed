from dataclasses import dataclass
from typing import Any

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import (
    AutoTokenizer,
    AutoProcessor
)
from torch.distributed.device_mesh import DeviceMesh


@dataclass
class QwenVLModelBundle:
    model: Qwen2_5_VLForConditionalGeneration
    processor: AutoProcessor | AutoTokenizer


@dataclass
class FSDPContext:
    """
    Metadata associated with the FSDP configuration.
    """
    mesh: DeviceMesh
    num_language_layers: int
    num_vision_blocks: int