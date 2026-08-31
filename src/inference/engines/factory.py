"""
Single choke point for backend selection.

`import vllm` (transitively, via `src.inference.engines.vllm`) only ever
happens inside the `engine.type == "vllm"` branch, so a Transformers-only
deployment never touches the vLLM package or its CUDA extensions at all.
"""

from __future__ import annotations

from src.config.schema import EngineConfig, InferenceTransformersModelConfig, ProfilerConfig
from src.inference.engines.base import InferenceEngine
from src.inference.engines.transformers import TransformersQwenVLEngine
from src.utils.logger import logger


def build_engine(
    engine_config: EngineConfig,
    model_config: InferenceTransformersModelConfig | None,
    model_version: str,
    profiler_config: ProfilerConfig | None = None,
) -> InferenceEngine:
    if engine_config.type == "transformers":
        return TransformersQwenVLEngine(
            model_config, model_version=model_version, profiler_config=profiler_config
        )

    if engine_config.type == "vllm":
        resolved_dp = _resolve_data_parallel_size(engine_config.vllm.data_parallel_size)
        logger.info(
            "Resolved vLLM data_parallel_size=%d (configured=%r)",
            resolved_dp,
            engine_config.vllm.data_parallel_size,
        )

        from src.inference.engines.vllm import VLLMEngine  # deferred: see module docstring

        # vLLM resolves its own model_id/cache_dir/dtype from engine_config.vllm --
        # it never depends on model_config (the transformers-only config).
        return VLLMEngine(
            engine_config.vllm,
            model_version=model_version,
            data_parallel_size=resolved_dp,
            profiler_config=profiler_config,
        )

    raise ValueError(f"Unknown engine.type {engine_config.type!r}. Expected 'transformers' or 'vllm'.")


def _resolve_data_parallel_size(configured: int | str) -> int:
    if configured != "auto":
        return configured

    from src.utils.dist_utils import visible_gpu_count

    count = visible_gpu_count()
    if count < 1:
        raise RuntimeError("data_parallel_size='auto' but no visible GPUs were detected.")

    return count