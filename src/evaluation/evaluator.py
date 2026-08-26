from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from src.config.schema import EvalConfig
from src.evaluation.metrics import compute_metrics
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.dist_utils import get_rank
from src.utils.logger import logger
from src.utils.tensor_utils import count_supervised_tokens, move_to_device


class Evaluator:
    """
    Evaluator for Qwen-VL SFT checkpoints.

    Responsibilities
    ----------------
    - forward-only pass over a held-out split -> loss / perplexity
    - model.generate() over prompt-only samples -> task metrics
      (exact-match, etc.) against reference answers
    - aggregate results across distributed ranks
    - optionally log to MLflow and persist predictions to disk

    Mirrors SFTTrainer's batch handling (move_to_device,
    count_supervised_tokens, distributed all_reduce for loss) but never
    computes gradients or touches the optimizer/scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        device: torch.device,
        config: EvalConfig,
        tracking: MLflowTracker | None = None,
        profiler: torch.profiler.profile | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.config = config
        self.tracking = tracking
        self.profiler = profiler

    @torch.no_grad()
    def evaluate_loss(self, eval_loader: DataLoader) -> dict[str, float]:
        """
        Forward-only pass computing token-weighted NLL over the whole
        loader, aggregated across distributed ranks.
        """

        self.model.eval()

        local_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        local_token_count = torch.zeros((), device=self.device, dtype=torch.float64)

        for batch in eval_loader:
            batch = move_to_device(batch, self.device)

            outputs = self.model(**batch)
            loss = outputs.loss

            supervised_tokens = count_supervised_tokens(batch["labels"])
            local_loss_sum += loss.detach().double() * supervised_tokens
            local_token_count += supervised_tokens

            if self.profiler is not None:
                self.profiler.step()

        totals = torch.stack([local_loss_sum, local_token_count])

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        global_loss_sum, global_token_count = totals[0], totals[1]

        eval_loss = (
            (global_loss_sum / global_token_count).item()
            if global_token_count.item() > 0
            else float("nan")
        )
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

        metrics = {"eval/loss": eval_loss, "eval/perplexity": perplexity}

        if get_rank() == 0:
            logger.info("Eval loss=%.6f | perplexity=%.4f", eval_loss, perplexity)

        return metrics

    @torch.no_grad()
    def evaluate_generation(
        self, generation_loader: DataLoader
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """
        Runs batched generation over prompt-only samples and scores the
        decoded text against reference answers. Predictions/references
        are gathered across ranks before scoring so metrics are computed
        once, identically, on every rank.
        """

        self.model.eval()

        local_predictions: list[str] = []
        local_references: list[str] = []

        for batch in generation_loader:
            references = batch.pop("references")
            batch = move_to_device(batch, self.device)

            prompt_length = batch["input_ids"].shape[1]

            generated_ids = self.model.generate(
                **batch,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                do_sample=self.config.do_sample,
                pad_token_id=self.processor.pad_token_id,
            )

            new_tokens = generated_ids[:, prompt_length:]
            decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)

            local_predictions.extend(text.strip() for text in decoded)
            local_references.extend(references)
            _log_predictions(decoded, references)

            if self.profiler is not None:
                self.profiler.step()

        predictions, references = _gather_generation_results(
            local_predictions, local_references
        )

        metric_names = self.config.metrics or ["exact_match"]
        metric_values = compute_metrics(metric_names, predictions, references)
        metrics = {f"eval/{name}": value for name, value in metric_values.items()}

        if get_rank() == 0:
            logger.info("Eval generation metrics: %s", metrics)

        return metrics, predictions, references

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        if get_rank() != 0:
            return

        if self.tracking is not None:
            self.tracking.log_metrics(metrics, step=step)

    def save_predictions(
        self,
        predictions: list[str],
        references: list[str],
    ) -> Path | None:
        if get_rank() != 0 or not self.config.save_predictions:
            return None

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predictions.jsonl"

        with output_path.open("w") as output_file:
            for prediction, reference in zip(predictions, references):
                output_file.write(
                    json.dumps({"prediction": prediction, "reference": reference}) + "\n"
                )

        logger.info("Saved %d predictions to %s", len(predictions), output_path)

        return output_path


def _gather_generation_results(
    local_predictions: list[str],
    local_references: list[str],
) -> tuple[list[str], list[str]]:
    if not (dist.is_available() and dist.is_initialized()):
        return local_predictions, local_references

    world_size = dist.get_world_size()
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, (local_predictions, local_references))

    predictions: list[str] = []
    references: list[str] = []
    for rank_predictions, rank_references in gathered:
        predictions.extend(rank_predictions)
        references.extend(rank_references)

    return predictions, references


def _log_predictions(
    predictions: list[str],
    references: list[str],
) -> None:
    if get_rank() == 0:
        for index, (prediction, reference) in enumerate(zip(predictions, references)):
            logger.info(
                "Generation sample %d | prediction=%r | reference=%r",
                index,
                prediction,
                reference,
            )
