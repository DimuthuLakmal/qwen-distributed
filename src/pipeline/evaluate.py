from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.config.configuration import ConfigurationManager
from src.config.schema import (
    DataConfig,
    DataLoaderConfig,
    EvalConfig,
    FSDPConfig,
    ModelConfig,
    ProfilerConfig,
    TrackingConfig,
)
from src.data.dataloader import build_dataloader
from src.data.dataset import SFTDataset
from src.data.eval_dataset import SFTEvalDataset
from src.data.eval_collate import GenerationCollator
from src.evaluation.evaluator import Evaluator
from src.models.qwen_vl import build_qwen_vl
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.profiler import build_profiler
from src.training.checkpoint import CheckpointManager
from src.training.fsdp import apply_fsdp
from src.utils.dist_utils import (
    barrier,
    cleanup_distributed,
    get_rank,
    initialize_distributed,
)
from src.utils.logger import logger


def run_evaluation(
    data_config: DataConfig,
    dataloader_config: DataLoaderConfig,
    model_config: ModelConfig,
    fsdp_config: FSDPConfig,
    eval_config: EvalConfig,
    tracking_config: TrackingConfig,
    profiler_config: ProfilerConfig,
) -> dict[str, float]:
    """
    Qwen-VL SFT evaluation pipeline.

    Supports two model sources (eval_config.model_source):
        - "checkpoint": build the base model from model_config and restore
          weights from a CheckpointManager-written distributed (DCP)
          training checkpoint.
        - "pretrained": load a self-contained Hugging Face directory
          produced by export_huggingface_model(), with no checkpoint
          restore step.

    Responsibilities:
        - orchestrate distributed setup
        - create model / processor
        - apply FSDP2 (so a single rank never has to hold the full model)
        - restore model weights (checkpoint) or load them directly (pretrained)
        - compute held-out loss/perplexity and/or generation metrics
        - log to MLflow and persist predictions
    """

    if eval_config.model_source == "pretrained":
        if not eval_config.pretrained_model_dir:
            raise ValueError(
                "eval_config.pretrained_model_dir must be set when "
                "model_source is 'pretrained'."
            )
    elif eval_config.model_source == "checkpoint":
        if not eval_config.checkpoint_dir:
            raise ValueError(
                "eval_config.checkpoint_dir must be set when "
                "model_source is 'checkpoint'."
            )
    else:
        raise ValueError(
            f"Unknown eval_config.model_source: {eval_config.model_source!r}. "
            "Expected 'checkpoint' or 'pretrained'."
        )

    device = initialize_distributed()
    rank = get_rank()

    tracking = None

    try:
        if rank == 0:
            logger.info("Starting distributed Qwen-VL SFT evaluation")

        tracking = MLflowTracker(config=tracking_config, checkpoint_dir=eval_config.output_dir)
        tracking.start()
        tracking.log_params(
            data=data_config, model=model_config, evaluation=eval_config, profiler=profiler_config
        )

        if eval_config.model_source == "pretrained":
            if rank == 0:
                logger.info(
                    "Loading exported Hugging Face model from %s",
                    eval_config.pretrained_model_dir,
                )

            model_config = replace(model_config, model_id=eval_config.pretrained_model_dir)

        model_bundle = build_qwen_vl(model_config)
        model = model_bundle.model
        processor = model_bundle.processor

        shard_model = (
            eval_config.model_source == "checkpoint" or eval_config.shard_pretrained_model
        )

        if shard_model:
            model, _ = apply_fsdp(model=model, config=fsdp_config)
        else:
            model = model.to(device)

        if eval_config.model_source == "checkpoint":
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=eval_config.checkpoint_dir,
                save_every_steps=1,
                keep_last_n=None,
                save_at_epoch_end=False,
            )

            checkpoint_path = (
                checkpoint_manager.latest_checkpoint()
                if eval_config.checkpoint_path == "latest"
                else Path(eval_config.checkpoint_path)
            )

            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoint found under {eval_config.checkpoint_dir}"
                )

            checkpoint_manager.load_model(checkpoint_path=checkpoint_path, model=model)

        barrier()

        all_metrics: dict[str, float] = {}

        with build_profiler(profiler_config) as profiler:
            evaluator = Evaluator(
                model=model,
                processor=processor,
                device=device,
                config=eval_config,
                tracking=tracking,
                profiler=profiler,
            )

            if eval_config.compute_loss:
                if rank == 0:
                    logger.info("Computing held-out loss/perplexity...")

                eval_dataset = SFTDataset.from_config(data_config=data_config, processor=processor)
                eval_loader, _ = build_dataloader(eval_dataset, dataloader_config, shuffle=False)

                all_metrics.update(evaluator.evaluate_loss(eval_loader))

            if eval_config.compute_generation_metrics:
                if rank == 0:
                    logger.info("Computing generation-based task metrics...")

                generation_dataset = SFTEvalDataset.from_config(
                    data_config=data_config, processor=processor
                )
                generation_loader, _ = build_dataloader(
                    generation_dataset,
                    dataloader_config,
                    shuffle=False,
                    collator_cls=GenerationCollator,
                )

                generation_metrics, predictions, references = evaluator.evaluate_generation(
                    generation_loader
                )
                all_metrics.update(generation_metrics)
                evaluator.save_predictions(predictions, references)

            evaluator.log_metrics(all_metrics)

        barrier()

        if rank == 0:
            logger.info("Evaluation complete | metrics=%s", all_metrics)

        return all_metrics

    finally:
        if tracking is not None:
            tracking.finish()

        cleanup_distributed()


# CLI
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Qwen-VL SFT checkpoint."
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
        tracking_config_path=config_dir / "tracking.yaml",
        eval_config_path=config_dir / "eval.yaml",
    )

    run_evaluation(
        data_config=config_manager.get_eval_data_config(),
        dataloader_config=config_manager.get_eval_dataloader_config(),
        model_config=config_manager.get_model_config(),
        fsdp_config=config_manager.get_fsdp_config(),
        eval_config=config_manager.get_eval_config(),
        tracking_config=config_manager.get_tracking_config(),
        profiler_config=config_manager.get_profiler_config(),
    )


if __name__ == "__main__":
    main()
