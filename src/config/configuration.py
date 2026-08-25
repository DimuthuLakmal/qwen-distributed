from pathlib import Path

import torch

from src.utils.io_utils import read_yaml, create_directories
from src.config.schema import (
    DataConfig,
    DataLoaderConfig,
    ModelConfig,
    FSDPConfig,
    TrainingConfig,
    CheckpointConfig,
    ExportConfig,
    TrackingConfig,
)

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ConfigurationManager:
    def __init__(self, data_config_path: Path,
                 model_config_path: Path,
                 training_config_path: Path):
        self.data_config = read_yaml(data_config_path)
        self.model_config = read_yaml(model_config_path)
        self.training_config = read_yaml(training_config_path)

    def get_data_config(self) -> DataConfig:
        config = self.data_config.dataset

        create_directories([d for d in (config.cache_dir, config.image_root) if d])

        data_config = DataConfig(
            path=config.path,
            split=config.split,
            cache_dir=config.cache_dir,
            image_root=config.image_root,
            max_length=config.max_length,
            adapter=config.adapter,
        )

        return data_config

    def get_dataloader_config(self) -> DataLoaderConfig:
        config = self.data_config.dataloader

        dataloader_config = DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            drop_last=config.drop_last,
        )

        return dataloader_config

    def get_model_config(self) -> ModelConfig:
        config = self.model_config.model

        create_directories([config.cache_dir] if config.cache_dir else [])

        model_config = ModelConfig(
            model_id=config.model_id,
            cache_dir=config.cache_dir,
            dtype=_TORCH_DTYPES[config.dtype],
            attn_implementation=config.attn_implementation,
            freeze_vision_tower=config.freeze_vision_tower,
            gradient_checkpointing=config.gradient_checkpointing,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            freeze_language_model=config.freeze_language_model,
            freeze_vision=config.freeze_vision,
            freeze_vision_merger=config.freeze_vision_merger,
        )

        return model_config

    def get_fsdp_config(self) -> FSDPConfig:
        config = self.training_config.fsdp

        fsdp_config = FSDPConfig(
            enabled=config.enabled,
            reshard_after_forward=config.reshard_after_forward,
            shard_language_layers=config.shard_language_layers,
            shard_vision_blocks=config.shard_vision_blocks,
            mixed_precision=config.mixed_precision,
            param_dtype=_TORCH_DTYPES[config.param_dtype],
            reduce_dtype=_TORCH_DTYPES[config.reduce_dtype],
            output_dtype=_TORCH_DTYPES[config.output_dtype] if config.output_dtype else None,
            cpu_offload=config.cpu_offload,
            cpu_offload_pin_memory=config.cpu_offload_pin_memory,
        )

        return fsdp_config


    def get_training_config(self) -> TrainingConfig:
        config = self.training_config.training

        training_config = TrainingConfig(
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            adam_epsilon=config.adam_epsilon,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            scheduler_type=config.scheduler_type,
            warmup_ratio=config.warmup_ratio,
            warmup_steps=config.warmup_steps,
            log_every_steps=config.log_every_steps,
        )

        return training_config

    def get_checkpoint_config(self) -> CheckpointConfig:
        config = self.training_config.checkpoint

        if config.enabled:
            create_directories([config.directory])

        checkpoint_config = CheckpointConfig(
            enabled=config.enabled,
            directory=config.directory,
            save_every_steps=config.save_every_steps,
            save_at_epoch_end=config.save_at_epoch_end,
            keep_last_n=config.keep_last_n,
            resume_from=config.resume_from,
        )

        return checkpoint_config


    def get_export_config(self) -> ExportConfig:
        config = self.model_config.export

        if config.enabled:
            create_directories([config.output_dir])

        export_config = ExportConfig(
            enabled=config.enabled,
            output_dir=config.output_dir,
            dtype=config.dtype,
            max_shard_size=config.max_shard_size,
        )

        return export_config

    def get_tracking_config(self) -> TrackingConfig:
        config = self.training_config.tracking

        tracking_config = TrackingConfig(
            enabled=config.enabled,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
            mlflow_experiment_name=config.mlflow_experiment_name,
            mlflow_run_name=config.mlflow_run_name,
            log_perf_metrics=config.log_perf_metrics,
            s3_bucket=config.s3_bucket,
            s3_prefix=config.s3_prefix,
            metrics_log_interval=config.metrics_log_interval,
        )

        return tracking_config
