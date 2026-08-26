from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.training.fsdp import set_gradient_sync
from src.training.schema import TrainState
from src.utils.logger import logger
from src.utils.dist_utils import get_rank
from src.utils.tensor_utils import count_supervised_tokens, move_to_device
from src.training.checkpoint import CheckpointManager
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.perf_metrics import PerfMetricsS3Uploader


class SFTTrainer:
    """
    Trainer for Qwen-VL supervised fine-tuning.

    Responsibilities
    ----------------
    - move batches to the local device
    - forward/backward
    - gradient accumulation
    - FSDP gradient synchronization control
    - gradient clipping
    - optimizer updates
    - LR scheduler updates
    - distributed loss logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: Any,
        device: torch.device,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        checkpoint_manager: CheckpointManager = None,
        tracking: MLflowTracker | None = None,
        perf_metrics: PerfMetricsS3Uploader | None = None,
        profiler: torch.profiler.profile | None = None,
    ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device

        self.state = TrainState()

        self.num_batches = len(train_loader)
        self.steps_per_epoch = math.ceil(
            self.num_batches/ self.config.gradient_accumulation_steps
        )

        self.total_training_steps = self.steps_per_epoch * self.config.num_epochs

        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()
        self.scheduler = scheduler if scheduler is not None else self._build_scheduler()

        self.checkpoint_manager = checkpoint_manager
        self.tracking = tracking
        self.perf_metrics = perf_metrics
        self.profiler = profiler

        self._log_training_configuration()

    def train(self) -> TrainState:
        """
        Execute the full training loop.
        """

        self.model.train()

        self.optimizer.zero_grad(set_to_none=True)

        training_start = time.perf_counter()

        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.num_epochs):
            self.state.epoch = epoch

            # DistributedSampler requires a new epoch seed.
            self._set_sampler_epoch(epoch)
            start_batch = self.state.batch_in_epoch if epoch == start_epoch else 0

            self._train_epoch(epoch=epoch, start_batch=start_batch)

            self.state.epoch = epoch + 1
            self.state.batch_in_epoch = 0

            # Save checkpoint at the end of epoch
            if (
                self.checkpoint_manager is not None
                and self.checkpoint_manager.save_at_epoch_end
            ):
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    train_state=self.state,
                )

        elapsed = time.perf_counter() - training_start

        if get_rank() == 0:
            logger.info(
                "Training finished | "
                "optimizer_steps=%d | "
                "micro_steps=%d | "
                "time=%.2fs",
                self.state.optimizer_step,
                self.state.micro_step,
                elapsed,
            )

        return self.state

    def _train_epoch(self, epoch: int, start_batch: int = 0,) -> None:
        self.model.train()

        # Accumulates token-weighted NLL for logging.
        local_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64,)
        local_token_count = torch.zeros((), device=self.device, dtype=torch.float64,)

        log_start = time.perf_counter()

        for batch_idx, batch in enumerate(self.train_loader):

            self.state.micro_step += 1

            # Determine the accumulation window size
            accumulation_size = self._get_accumulation_window_size(batch_idx)

            # Check if the optimizer needs to be run
            is_optimizer_step = self._is_optimizer_step(batch_idx)

            # FSDP gradient communication
            # Intermediate microbatches:
            #     no reduce-scatter
            # Final microbatch:
            #     synchronize gradients
            if (self.config.gradient_accumulation_steps > 1):
                set_gradient_sync(
                    self.model,
                    enabled=is_optimizer_step,
                )

            # Move batch to local GPU
            batch = move_to_device(batch, self.device)

            # Forward
            outputs = self.model(**batch)

            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss detected on "
                    f"rank={get_rank()}: {loss.detach().item()}"
                )


            # Qwen's LM loss is averaged over supervised tokens.
            supervised_tokens = count_supervised_tokens(batch["labels"])
            local_loss_sum += loss.detach().double() * supervised_tokens
            local_token_count += supervised_tokens

            # Gradient accumulation
            scaled_loss = loss / accumulation_size
            scaled_loss.backward()

            # Nothing further to do until the final microbatch in the accumulation window.
            if not is_optimizer_step:
                continue

            # Gradient clipping
            # FSDP2 uses DTensor gradients, and the normal PyTorch
            # clipping API handles distributed DTensor norms.
            grad_norm = None

            if (self.config.max_grad_norm is not None and self.config.max_grad_norm > 0):
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                )

            # Optimizer
            self.optimizer.step()

            # Scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

            self.state.optimizer_step += 1
            self.state.batch_in_epoch += 1

            if self.perf_metrics is not None:
                self.perf_metrics.maybe_collect(self.state.optimizer_step)

            if self.profiler is not None:
                self.profiler.step()

            # Checkpointing
            if (
                self.checkpoint_manager is not None
                and self.checkpoint_manager.should_save(self.state.optimizer_step)
            ):
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    train_state=self.state,
                )

            # Logging
            if (self.state.optimizer_step % self.config.log_every_steps == 0):
                self._log_step(
                    local_loss_sum=local_loss_sum,
                    local_token_count=local_token_count,
                    grad_norm=grad_norm,
                    start_time=log_start,
                )

                local_loss_sum.zero_()
                local_token_count.zero_()

                log_start = time.perf_counter()

        # Log remaining measurements at end of epoch
        if local_token_count.item() > 0:
            self._log_step(
                local_loss_sum=local_loss_sum,
                local_token_count=local_token_count,
                grad_norm=None,
                start_time=log_start,
            )

        if get_rank() == 0:
            logger.info(
                "Finished epoch %d/%d",
                epoch + 1,
                self.config.num_epochs,
            )

    def _is_optimizer_step(self, batch_idx: int,) -> bool:
        """
        Return True for the final microbatch of each accumulation window.
        Also forces an optimizer step for an incomplete final window.
        """
        accumulation_steps = self.config.gradient_accumulation_steps

        normal_boundary = (batch_idx + 1) % accumulation_steps == 0
        final_batch = (batch_idx + 1 == self.num_batches)

        return normal_boundary or final_batch


    def _get_accumulation_window_size(self, batch_idx: int,) -> int:
        """
        Return the number of microbatches in the current accumulation window.
        """
        accumulation_steps = self.config.gradient_accumulation_steps

        window_start = (batch_idx // accumulation_steps) * accumulation_steps # (9//4)*4 = 8
        window_end = min(window_start + accumulation_steps, self.num_batches,)

        return window_end - window_start


    def _build_optimizer(self,) -> Optimizer:
        """
        Build AdamW.
        Parameters with ndim >= 2 receive weight decay.
        Biases, LayerNorm/RMSNorm parameters, etc. are typically
        one-dimensional and therefore excluded from weight decay.
        """

        decay_params = []
        no_decay_params = []

        for parameter in self.model.parameters():

            if not parameter.requires_grad:
                continue

            if parameter.ndim >= 2:
                decay_params.append(
                    parameter
                )
            else:
                no_decay_params.append(
                    parameter
                )

        optimizer_groups = [
            {
                "params": decay_params,
                "weight_decay": (
                    self.config.weight_decay
                ),
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(
                self.config.adam_beta1,
                self.config.adam_beta2,
            ),
            eps=self.config.adam_epsilon,
        )

    def _build_scheduler(self,) -> LRScheduler:

        if self.config.warmup_steps is not None:
            warmup_steps = self.config.warmup_steps

        else:
            warmup_steps = int(self.total_training_steps * self.config.warmup_ratio)

        return get_scheduler(
            name=self.config.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_training_steps,
        )

    def _log_step(
        self,
        local_loss_sum: torch.Tensor,
        local_token_count: torch.Tensor,
        grad_norm: torch.Tensor | None,
        start_time: float,
    ) -> None:

        metrics = torch.stack([local_loss_sum, local_token_count,])

        # Aggregate across all FSDP ranks
        if (dist.is_available() and dist.is_initialized()):
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM,)

        global_loss_sum = metrics[0]
        global_token_count = metrics[1]

        if global_token_count.item() > 0:
            global_loss = (global_loss_sum / global_token_count).item()

        else:
            global_loss = float("nan")

        elapsed = max(time.perf_counter() - start_time, 1e-8,)
        tokens_per_second = (global_token_count.item() / elapsed)

        learning_rate = (self.optimizer.param_groups[0]["lr"])

        if get_rank() == 0:
            grad_norm_value = None
            if grad_norm is not None:
                # DTensor/scalar Tensor safe conversion.
                try:
                    grad_norm_value = grad_norm.item()
                except RuntimeError:
                    grad_norm_value = float(grad_norm)

            log_message = (
                f"epoch={self.state.epoch + 1} "
                f"step={self.state.optimizer_step}/"
                f"{self.total_training_steps} "
                f"loss={global_loss:.6f} "
                f"lr={learning_rate:.3e} "
                f"tokens/s={tokens_per_second:.0f}"
            )

            if grad_norm_value is not None:
                log_message += (f" grad_norm={grad_norm_value:.4f}")

            logger.info(log_message)

            if self.tracking is not None:
                metrics = {
                    "train/loss": global_loss,
                    "train/learning_rate": learning_rate,
                    "train/tokens_per_second": tokens_per_second,
                }

                if grad_norm_value is not None:
                    metrics["train/grad_norm"] = grad_norm_value

                self.tracking.log_metrics(metrics, step=self.state.optimizer_step)

    def _set_sampler_epoch(self, epoch: int,) -> None:
        sampler = getattr(self.train_loader, "sampler", None,)

        if (sampler is not None and hasattr(sampler, "set_epoch",)):
            sampler.set_epoch(epoch)

    def _log_training_configuration(self,) -> None:
        if get_rank() != 0:
            return

        world_size = (
            dist.get_world_size()
            if (
                dist.is_available()
                and dist.is_initialized()
            )
            else 1
        )

        per_device_batch = self.train_loader.batch_size
        global_batch_size = per_device_batch * world_size * self.config.gradient_accumulation_steps

        logger.info(
            "\n"
            "Training configuration\n"
            "----------------------\n"
            "Epochs                   : %d\n"
            "Batches / epoch          : %d\n"
            "Optimizer steps / epoch  : %d\n"
            "Total optimizer steps    : %d\n"
            "Per-device batch size    : %s\n"
            "World size               : %d\n"
            "Gradient accumulation    : %d\n"
            "Effective global batch   : %s\n"
            "Learning rate            : %.3e\n"
            "Weight decay             : %.3e\n"
            "Scheduler                 : %s\n",
            self.config.num_epochs,
            self.num_batches,
            self.steps_per_epoch,
            self.total_training_steps,
            per_device_batch,
            world_size,
            self.config.gradient_accumulation_steps,
            global_batch_size,
            self.config.learning_rate,
            self.config.weight_decay,
            self.config.scheduler_type,
        )