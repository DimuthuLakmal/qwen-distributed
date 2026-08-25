import os

import torch
import torch.distributed as dist


def initialize_distributed() -> torch.device:
    dist.init_process_group(backend="nccl",)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return torch.device("cuda", local_rank,)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if (dist.is_available() and dist.is_initialized()):
        return dist.get_rank()

    return 0


def barrier() -> None:
    if (dist.is_available() and dist.is_initialized()):
        dist.barrier()

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()