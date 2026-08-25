from torch.distributed.device_mesh import DeviceMesh
from dataclasses import dataclass


@dataclass
class FSDPContext:
    """
    Metadata associated with the FSDP configuration.
    """
    mesh: DeviceMesh
    num_language_layers: int
    num_vision_blocks: int


@dataclass
class TrainState:
    epoch: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    batch_in_epoch: int = 0