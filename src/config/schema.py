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
class TrackingConfig:
    enabled: bool = False

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "qwen-vl-sft"
    mlflow_run_name: str | None = None

    log_perf_metrics: bool = False
    s3_bucket: str | None = None
    s3_prefix: str = "metrics"
    metrics_log_interval: int = 50