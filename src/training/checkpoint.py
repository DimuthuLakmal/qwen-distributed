from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from src.utils.dist_utils import get_rank, barrier
from src.utils.logger import logger


class TrainingAppState(Stateful):
    """
    Complete state required to resume SFT training.

    Model and optimizer states are handled through PyTorch distributed
    state-dict APIs, making them compatible with FSDP2 / DTensor.

    Scheduler and trainer progress are replicated state and therefore
    stored normally inside the distributed checkpoint.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_state,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_state = train_state

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model,
            self.optimizer,
        )

        state = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,

            "trainer": {
                "epoch": self.train_state.epoch,
                "batch_in_epoch": self.train_state.batch_in_epoch,
                "micro_step": self.train_state.micro_step,
                "optimizer_step": self.train_state.optimizer_step,
            },
        }

        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict: dict[str, Any],) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )


        if (self.scheduler is not None and "scheduler" in state_dict):
            self.scheduler.load_state_dict(state_dict["scheduler"])

        trainer_state = state_dict["trainer"]
        self.train_state.epoch = trainer_state["epoch"]
        self.train_state.batch_in_epoch = (trainer_state["batch_in_epoch"])
        self.train_state.micro_step = (trainer_state["micro_step"])
        self.train_state.optimizer_step = (trainer_state["optimizer_step"])


class ModelOnlyAppState(Stateful):
    """
    Model-only counterpart to TrainingAppState.

    Used to restore just the model weights from a checkpoint (e.g. for
    evaluation), without needing an optimizer/scheduler to satisfy the
    Stateful contract.
    """

    def __init__(self, model) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return {"model": get_model_state_dict(self.model)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(self.model, model_state_dict=state_dict["model"])


class CheckpointManager:
    """
    Distributed checkpoint manager for FSDP2 training.

    All ranks MUST call save() and load().
    """

    SUCCESS_FILE = "_SUCCESS"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_every_steps: int = 500,
        keep_last_n: int | None = 1,
        save_at_epoch_end: bool = False
    ) -> None:

        self.checkpoint_dir = Path(checkpoint_dir)

        self.save_every_steps = save_every_steps
        self.keep_last_n = keep_last_n
        self.save_at_epoch_end = save_at_epoch_end

        if self.save_every_steps <= 0:
            raise ValueError("save_every_steps must be > 0.")

        if get_rank() == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True,)

        barrier()

    def save(
        self,
        model,
        optimizer,
        scheduler,
        train_state,
    ) -> Path:
        """
        Save one distributed checkpoint.
        Every distributed rank must call this function.
        """

        step = train_state.optimizer_step
        checkpoint_path = (self.checkpoint_dir / f"step_{step:08d}")

        # Prepare directory
        if get_rank() == 0:
            # Remove an incomplete checkpoint from a failed previous
            # attempt at the same step.
            if checkpoint_path.exists():
                success_file = (checkpoint_path / self.SUCCESS_FILE)

                if not success_file.exists():
                    shutil.rmtree(checkpoint_path)
                else:
                    raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

            checkpoint_path.mkdir(parents=True, exist_ok=False,)

        barrier()

        if get_rank() == 0:
            logger.info("Saving distributed checkpoint: %s", checkpoint_path,)

        # Construct state
        app_state = TrainingAppState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=train_state,
        )

        # Distributed save
        # ALL ranks participate.
        # Each rank writes its local DTensor / optimizer shards.
        dcp.save(state_dict={"app": app_state,}, checkpoint_id=str(checkpoint_path),)

        # dcp.save() is synchronous. Once every rank returns, the
        # checkpoint data has been written.

        barrier()

        # Mark checkpoint complete
        if get_rank() == 0:
            (checkpoint_path / self.SUCCESS_FILE).touch()
            self._write_latest_pointer(checkpoint_path)

            logger.info("Checkpoint saved successfully: %s", checkpoint_path,)

            self._prune_old_checkpoints()

        barrier()

        return checkpoint_path

    # Load
    def load(
        self,
        checkpoint_path: str | Path,
        model,
        optimizer,
        scheduler,
        train_state,
    ) -> None:
        """
        Restore a distributed checkpoint.

        The model must already be FSDP2-sharded and the optimizer
        must already have been constructed.
        """

        checkpoint_path = Path(checkpoint_path)
        self._validate_checkpoint(checkpoint_path)

        if get_rank() == 0:
            logger.info("Loading distributed checkpoint: %s", checkpoint_path,)

        app_state = TrainingAppState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=train_state,
        )

        # DCP reads only the shards required by the current rank.
        # The current world size does not necessarily have to equal
        # the world size used when the checkpoint was written.
        dcp.load(state_dict={"app": app_state,}, checkpoint_id=str(checkpoint_path),)

        barrier()

        if get_rank() == 0:
            logger.info(
                "Checkpoint restored | "
                "epoch=%d | "
                "batch=%d | "
                "optimizer_step=%d",
                train_state.epoch,
                train_state.batch_in_epoch,
                train_state.optimizer_step,
            )

    def load_model(
        self,
        checkpoint_path: str | Path,
        model,
    ) -> None:
        """
        Restore only the model weights from a checkpoint, skipping the
        optimizer/scheduler/trainer state. Intended for evaluation,
        where no optimizer exists.

        The model must already be FSDP2-sharded, same as load().
        """

        checkpoint_path = Path(checkpoint_path)
        self._validate_checkpoint(checkpoint_path)

        if get_rank() == 0:
            logger.info("Loading model weights from checkpoint: %s", checkpoint_path,)

        app_state = ModelOnlyAppState(model=model)

        dcp.load(state_dict={"app": app_state,}, checkpoint_id=str(checkpoint_path),)

        barrier()

        if get_rank() == 0:
            logger.info("Model weights restored from: %s", checkpoint_path,)

    def should_save(
        self,
        optimizer_step: int,
    ) -> bool:
        return (optimizer_step > 0 and optimizer_step % self.save_every_steps == 0)

    def latest_checkpoint(self,) -> Path | None:
        """
        Find the latest successfully completed checkpoint.

        Rank 0 performs filesystem discovery and broadcasts the
        resolved path to all ranks.
        """

        checkpoint: str | None = None

        if get_rank() == 0:
            candidates = []

            for path in self.checkpoint_dir.glob("step_*"):
                if not path.is_dir():
                    continue

                if not (path / self.SUCCESS_FILE).exists():
                    continue

                try:
                    step = int(path.name.split("_")[1])
                except (ValueError, IndexError):
                    continue

                candidates.append((step, path))

            if candidates:
                _, latest = max(candidates, key=lambda item: item[0],)
                checkpoint = str(latest)

        # Broadcast rank-0's decision
        if (dist.is_available() and dist.is_initialized()):
            result = [checkpoint]
            dist.broadcast_object_list(result, src=0,)

            checkpoint = result[0]

        if checkpoint is None:
            return None

        return Path(checkpoint)

    def _write_latest_pointer(self, checkpoint_path: Path,) -> None:
        latest_file = (self.checkpoint_dir / "latest")
        latest_file.write_text(checkpoint_path.name)

    def _prune_old_checkpoints(self,) -> None:
        if (self.keep_last_n is None or self.keep_last_n <= 0):
            return

        checkpoints = []

        for path in self.checkpoint_dir.glob(
            "step_*"
        ):
            if not path.is_dir():
                continue

            if not (path / self.SUCCESS_FILE).exists():
                continue

            try:
                step = int(path.name.split("_")[1])
            except (ValueError, IndexError,):
                continue

            checkpoints.append((step, path))

        checkpoints.sort(key=lambda item: item[0],reverse=True,)

        for _, path in checkpoints[self.keep_last_n :]:
            logger.info("Removing old checkpoint: %s", path,)
            shutil.rmtree(path)

    def _validate_checkpoint(self, checkpoint_path: Path,) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        success_file = (checkpoint_path / self.SUCCESS_FILE)

        if not success_file.exists():
            raise RuntimeError(f"Checkpoint is incomplete: {checkpoint_path}")