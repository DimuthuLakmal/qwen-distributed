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
    EvalConfig,
    ProfilerConfig,
    InferenceServiceConfig,
    InferenceModelConfig,
    GenerationDefaultsConfig,
    InferenceLimitsConfig,
)

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ConfigurationManager:
    def __init__(self, data_config_path: Path | None = None,
                 model_config_path: Path | None = None,
                 training_config_path: Path | None = None,
                 tracking_config_path: Path | None = None,
                 eval_config_path: Path | None = None,
                 inference_config_path: Path | None = None):
        self.data_config = read_yaml(data_config_path) if data_config_path else None
        self.model_config = read_yaml(model_config_path) if model_config_path else None
        self.training_config = read_yaml(training_config_path) if training_config_path else None
        self.tracking_config = read_yaml(tracking_config_path) if tracking_config_path else None
        self.eval_config = read_yaml(eval_config_path) if eval_config_path else None
        self.inference_config = read_yaml(inference_config_path) if inference_config_path else None

    @staticmethod
    def _require(config_box, config_name: str):
        if config_box is None:
            raise RuntimeError(
                f"{config_name} was not loaded. Pass a path for it to "
                "ConfigurationManager before calling this getter."
            )

        return config_box

    def get_data_config(self) -> DataConfig:
        config = self._require(self.data_config, "data_config_path").dataset

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
        config = self._require(self.data_config, "data_config_path").dataloader

        dataloader_config = DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            drop_last=config.drop_last,
        )

        return dataloader_config

    def get_model_config(self) -> ModelConfig:
        config = self._require(self.model_config, "model_config_path").model

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
        config = self._require(self.training_config, "training_config_path").fsdp

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
        config = self._require(self.training_config, "training_config_path").training

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
        config = self._require(self.training_config, "training_config_path").checkpoint

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
        config = self._require(self.model_config, "model_config_path").export

        if config.enabled:
            create_directories([config.output_dir])

        export_config = ExportConfig(
            enabled=config.enabled,
            output_dir=config.output_dir,
            dtype=config.dtype,
            max_shard_size=config.max_shard_size,
        )

        return export_config

    def get_eval_data_config(self) -> DataConfig:
        config = self._require(self.eval_config, "eval_config_path").dataset

        create_directories([d for d in (config.cache_dir, config.image_root) if d])

        return DataConfig(
            path=config.path,
            split=config.split,
            cache_dir=config.cache_dir,
            image_root=config.image_root,
            max_length=config.max_length,
            adapter=config.adapter,
        )

    def get_eval_dataloader_config(self) -> DataLoaderConfig:
        config = self._require(self.eval_config, "eval_config_path").dataloader

        return DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            drop_last=config.drop_last,
        )

    def get_eval_config(self) -> EvalConfig:
        config = self._require(self.eval_config, "eval_config_path").evaluation

        create_directories([config.output_dir] if config.output_dir else [])

        return EvalConfig(
            model_source=config.model_source,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_path=config.checkpoint_path,
            pretrained_model_dir=config.pretrained_model_dir,
            shard_pretrained_model=config.shard_pretrained_model,
            compute_loss=config.compute_loss,
            compute_generation_metrics=config.compute_generation_metrics,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            do_sample=config.do_sample,
            metrics=list(config.metrics) if config.metrics else None,
            output_dir=config.output_dir,
            save_predictions=config.save_predictions,
        )

    def get_tracking_config(self) -> TrackingConfig:
        config = self._require(self.tracking_config, "tracking_config_path").tracking

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

    def get_profiler_config(self) -> ProfilerConfig:
        config = self._require(self.tracking_config, "tracking_config_path").profiler

        if config.enabled:
            create_directories([config.output_dir])

        profiler_config = ProfilerConfig(
            enabled=config.enabled,
            output_dir=config.output_dir,
            wait_steps=config.wait_steps,
            warmup_steps=config.warmup_steps,
            active_steps=config.active_steps,
            repeat=config.repeat,
            record_shapes=config.record_shapes,
            profile_memory=config.profile_memory,
            with_stack=config.with_stack,
            profile_ranks=list(config.profile_ranks) if config.profile_ranks else None,
        )

        return profiler_config

    def get_inference_service_config(self) -> InferenceServiceConfig:
        config = self._require(self.inference_config, "inference_config_path").service

        return InferenceServiceConfig(
            host=config.host,
            port=config.port,
            model_version=config.model_version,
        )

    def get_inference_model_config(self) -> InferenceModelConfig:
        config = self._require(self.inference_config, "inference_config_path").model

        return InferenceModelConfig(
            model_path=config.model_path,
            device=config.device,
            dtype=_TORCH_DTYPES[config.dtype],
            attn_implementation=config.attn_implementation,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )

    def get_generation_defaults_config(self) -> GenerationDefaultsConfig:
        inference_config = self._require(self.inference_config, "inference_config_path")
        config = inference_config.generation
        limits = inference_config.limits

        if config.max_new_tokens > limits.max_new_tokens:
            raise ValueError(
                f"generation.max_new_tokens ({config.max_new_tokens}) must not "
                f"exceed limits.max_new_tokens ({limits.max_new_tokens})."
            )

        return GenerationDefaultsConfig(
            do_sample=config.do_sample,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            temperature=config.temperature,
        )

    def get_inference_limits_config(self) -> InferenceLimitsConfig:
        config = self._require(self.inference_config, "inference_config_path").limits

        return InferenceLimitsConfig(
            max_prompt_chars=config.max_prompt_chars,
            max_images=config.max_images,
            max_image_bytes=config.max_image_bytes,
            max_image_pixels=config.max_image_pixels,
            max_new_tokens=config.max_new_tokens,
        )
