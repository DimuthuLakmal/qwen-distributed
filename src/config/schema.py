from dataclasses import dataclass
from pathlib import Path

import torch

@dataclass
class DataConfig:
    path: str | Path
    split: str
    cache_dir: str | Path
    image_root: str | Path | None
    max_length: int
    adapter: str


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    drop_last: bool


@dataclass
class ModelConfig:
    model_id: str
    cache_dir: str | Path
    dtype: torch.dtype
    attn_implementation: str | None
    freeze_vision_tower: bool
    gradient_checkpointing: bool
    min_pixels: int
    max_pixels: int
    freeze_vision_merger: bool
    freeze_vision: bool
    freeze_language_model: bool
    gradient_checkpointing: bool


@dataclass
class FSDPConfig:
    enabled: bool
    reshard_after_forward: bool
    shard_language_layers: bool
    shard_vision_blocks: bool
    mixed_precision: bool
    param_dtype: torch.dtype
    reduce_dtype: torch.dtype
    output_dtype: torch.dtype | None
    cpu_offload: bool
    cpu_offload_pin_memory: bool
    
    
@dataclass
class TrainingConfig:
    num_epochs: int = 3

    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    gradient_accumulation_steps: int = 4

    max_grad_norm: float | None = 1.0

    scheduler_type: str = "cosine"

    warmup_ratio: float = 0.03
    warmup_steps: int | None = None

    log_every_steps: int = 10


@dataclass
class CheckpointConfig:
    enabled: bool = True
    directory: str = "checkpoints"
    save_every_steps: int = 500
    save_at_epoch_end: bool = True
    keep_last_n: int | None = 3
    resume_from: str | None = None


@dataclass
class ExportConfig:
    enabled: bool
    output_dir: str
    dtype: str | None = "bfloat16"
    max_shard_size: str = "5GB"


@dataclass
class EvalConfig:
    # "checkpoint"  -> restore weights from a CheckpointManager-written
    #                  distributed (DCP) training checkpoint.
    # "pretrained"  -> load a self-contained Hugging Face directory
    #                  produced by export_huggingface_model().
    model_source: str = "checkpoint"

    # Used when model_source == "checkpoint".
    checkpoint_dir: str | None = None
    checkpoint_path: str = "latest"

    # Used when model_source == "pretrained".
    pretrained_model_dir: str | None = None

    # Whether to FSDP2-shard the exported model across ranks (needed if it
    # doesn't fit on a single GPU) or replicate the full model per rank and
    # rely on DistributedSampler alone for data-parallel inference. Ignored
    # for model_source == "checkpoint", which always shards (the checkpoint
    # itself was written from a sharded model).
    shard_pretrained_model: bool = True

    compute_loss: bool = True

    compute_generation_metrics: bool = True
    max_new_tokens: int = 128
    num_beams: int = 1
    do_sample: bool = False
    metrics: list[str] | None = None

    output_dir: str = "results/eval"
    save_predictions: bool = True


@dataclass
class TrackingConfig:
    enabled: bool = False

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "qwen-vl-sft"
    mlflow_run_name: str | None = None

    log_perf_metrics: bool = False
    s3_bucket: str | None = None
    s3_prefix: str = "metrics"
    metrics_log_interval: int = 50


@dataclass
class ProfilerConfig:
    enabled: bool = False
    output_dir: str = "results/profiler"
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    record_shapes: bool = False
    profile_memory: bool = True
    with_stack: bool = False

    # Ranks to trace. Every rank under FSDP does near-identical work, so
    # tracing all of them just multiplies multi-GB trace files. None means
    # trace every rank.
    profile_ranks: list[int] | None = None


@dataclass
class InferenceServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_version: str = "unknown"


@dataclass
class InferenceModelConfig:
    model_path: str | Path = "results/final"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str | None = None
    min_pixels: int = 0
    max_pixels: int = 0


@dataclass
class GenerationDefaultsConfig:
    do_sample: bool = False
    max_new_tokens: int = 128
    num_beams: int = 1
    temperature: float = 1.0


@dataclass
class InferenceLimitsConfig:
    max_prompt_chars: int = 4000
    max_images: int = 4
    max_image_bytes: int = 8_000_000
    max_image_pixels: int = 20_000_000
    max_new_tokens: int = 512


@dataclass
class WorkerEndpointConfig:
    worker_id: str
    url: str


@dataclass
class RoutingConfig:
    strategy: str = "round_robin"
    connect_timeout_seconds: float = 2.0
    read_timeout_seconds: float = 30.0
    status_poll_timeout_seconds: float = 2.0