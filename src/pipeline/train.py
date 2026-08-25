from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from src.config.configuration import ConfigurationManager
from src.data.dataloader import build_dataloader
from src.data.dataset import SFTDataset
from src.models.qwen_vl import build_qwen_vl
from src.utils.dist_utils import (
    cleanup_distributed,
    initialize_distributed,
)
from src.training.fsdp import apply_fsdp
from src.training.trainer import SFTTrainer
from src.utils.logger import logger
from src.config.schema import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    DataLoaderConfig,
    FSDPConfig,
    CheckpointConfig,
    ExportConfig,
    TrackingConfig,
)
from src.utils.dist_utils import barrier, get_rank
from src.training.checkpoint import CheckpointManager
from src.models.export import export_huggingface_model
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.perf_metrics import PerfMetricsS3Uploader


def run_training(data_config: DataConfig,
                dataloader_config: DataLoaderConfig,
                model_config: ModelConfig,
                fsdp_config: FSDPConfig,
                training_config: TrainingConfig,
                checkpoint_config: CheckpointConfig,
                export_config: ExportConfig,
                tracking_config: TrackingConfig) -> None:
    """
    Main Qwen-VL SFT training pipeline.

    Responsibilities:
        - orchestrate distributed setup
        - create model / processor
        - create dataset / dataloader
        - apply FSDP2
        - create trainer
        - start training
        - checkpointing
    """

    # Initialize distributed environment
    device = initialize_distributed()

    rank = get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    tracking = None
    perf_metrics = None

    try:
        if rank == 0:
            logger.info("Starting distributed Qwen-VL SFT training")

        logger.info(
            "Distributed process initialized | "
            "rank=%d | local_rank=%d | world_size=%d | device=%s",
            rank,
            local_rank,
            world_size,
            device,
        )

        base_seed = getattr(training_config, "seed", 42,)
        _set_seed(base_seed + rank)

        # Experiment tracking
        # MLflow tracks configs and training metrics. GPU utilization is an
        # infra/perf metric rather than an experiment result, so it's
        # written to S3 as CSV instead (see PerfMetricsS3Uploader), for
        # Grafana to read.
        run_id = tracking_config.mlflow_run_name or Path(checkpoint_config.directory).name

        tracking = MLflowTracker(config=tracking_config, checkpoint_dir=checkpoint_config.directory)
        tracking.start()
        tracking.log_params(
            data=data_config,
            dataloader=dataloader_config,
            model=model_config,
            fsdp=fsdp_config,
            training=training_config,
            checkpoint=checkpoint_config,
            export=export_config,
        )

        perf_metrics = PerfMetricsS3Uploader(config=tracking_config, run_id=run_id)
        perf_metrics.start()

        # Build model and processor
        # At this point the model is still the ordinary Hugging Face Qwen2.5-VL model.
        # FSDP2 will place/shard the parameters according to its DeviceMesh.
        if rank == 0:
            logger.info("Loading Qwen-VL model and processor...")

        model_bundle = build_qwen_vl(model_config)
        model = model_bundle.model
        processor = model_bundle.processor

        if rank == 0:
            logger.info("Model and processor loaded successfully.")

        # Build training dataset
        if rank == 0:
            logger.info("Building training dataset...")

        train_dataset = SFTDataset.from_config(data_config=data_config, processor=processor,)

        if rank == 0:
            num_samples = len(train_dataset)
            logger.info("Training dataset ready | samples=%d", num_samples,)


        # Build distributed DataLoader
        train_loader, _ = build_dataloader(train_dataset, dataloader_config, shuffle=True)

        if rank == 0:
            logger.info(
                "Training DataLoader created | "
                "batches_per_rank=%d",
                len(train_loader),
            )

        # Apply FSDP2
        if rank == 0:
            logger.info("Applying FSDP2 sharding...")

        model, fsdp_context = apply_fsdp(model=model, config=fsdp_config)

        checkpoint_manager = None
        if checkpoint_config.enabled:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_config.directory,
                save_every_steps=checkpoint_config.save_every_steps,
                keep_last_n=checkpoint_config.keep_last_n,
            )

        if rank == 0:
            logger.info(
                "FSDP2 applied | "
                "language_layers=%d | "
                "vision_blocks=%d",
                fsdp_context.num_language_layers,
                fsdp_context.num_vision_blocks,
            )

        # Ensure every rank has completed model construction before
        # entering training.
        barrier()

        # Construct trainer
        trainer = SFTTrainer(
            model=model,
            train_loader=train_loader,
            config=training_config,
            device=device,
            checkpoint_manager=checkpoint_manager,
            tracking=tracking,
            perf_metrics=perf_metrics,
        )

        _resume_checkpoint(checkpoint_manager=checkpoint_manager, 
                        checkpoint_config=checkpoint_config, 
                        trainer=trainer)

        # Run training
        if rank == 0:
            logger.info("Beginning training...")

        train_state = trainer.train()

        # Final synchronization
        barrier()

        if export_config.enabled:
            export_huggingface_model(
                model=trainer.model,
                processor=processor,
                output_dir=export_config.output_dir,
                export_dtype=export_config.dtype,
                max_shard_size=export_config.max_shard_size,
            )

        if rank == 0:
            logger.info(
                "Training completed successfully | "
                "epochs=%d | "
                "optimizer_steps=%d | "
                "micro_steps=%d",
                train_state.epoch + 1,
                train_state.optimizer_step,
                train_state.micro_step,
            )

    finally:
        if perf_metrics is not None:
            perf_metrics.shutdown()

        if tracking is not None:
            tracking.finish()

        cleanup_distributed()


def _set_seed(seed: int,) -> None:
    """
    Seed Python, NumPy, CPU PyTorch, and CUDA RNGs.

    A rank-dependent seed should normally be supplied so that
    stochastic operations such as dropout are not identical across
    distributed workers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _resume_checkpoint(checkpoint_manager, checkpoint_config, trainer):
    if (checkpoint_manager is not None and checkpoint_config.resume_from is not None):
        resume_from = checkpoint_config.resume_from

        if resume_from == "latest":
            checkpoint_path = (checkpoint_manager.latest_checkpoint())

            if checkpoint_path is None:
                if get_rank() == 0:
                    logger.info( "No existing checkpoint found. Starting from pretrained model.")

            else:
                checkpoint_manager.load(
                    checkpoint_path=checkpoint_path,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    train_state=trainer.state,
                )

        else:
            checkpoint_manager.load(
                checkpoint_path=resume_from,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                train_state=trainer.state,
            )

# CLI
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Qwen-VL."
        )
    )

    parser.add_argument(
        "--config_dir",
        type=Path,
        required=True,
        help=(
            "Directory to the configuration files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_dir = args.config_dir
    config_manager = ConfigurationManager(
        data_config_path=config_dir / "data.yaml",
        model_config_path=config_dir / "model.yaml",
        training_config_path=config_dir / "training.yaml",
    )

    run_training(
        data_config=config_manager.get_data_config(),
        dataloader_config=config_manager.get_dataloader_config(),
        model_config=config_manager.get_model_config(),
        fsdp_config=config_manager.get_fsdp_config(),
        training_config=config_manager.get_training_config(),
        checkpoint_config=config_manager.get_checkpoint_config(),
        export_config=config_manager.get_export_config(),
        tracking_config=config_manager.get_tracking_config(),
    )


if __name__ == "__main__":
    main()