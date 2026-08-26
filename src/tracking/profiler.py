from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from src.config.schema import ProfilerConfig
from src.utils.dist_utils import get_rank
from src.utils.logger import logger


@contextmanager
def build_profiler(config: ProfilerConfig):
    """
    Yields a torch.profiler.profile tracing CPU/CUDA activity over the
    train/eval step loop, or None when disabled. Traces are written to a
    per-rank subdirectory (viewable in TensorBoard) so concurrent ranks
    don't clobber each other's output.
    """
    if not config.enabled:
        yield None
        return

    if config.profile_ranks is not None and get_rank() not in config.profile_ranks:
        yield None
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    output_dir = os.path.join(config.output_dir, f"rank{get_rank()}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Torch profiler enabled | output_dir=%s", output_dir)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.wait_steps,
            warmup=config.warmup_steps,
            active=config.active_steps,
            repeat=config.repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_stack=config.with_stack,
    ) as profiler:
        yield profiler
