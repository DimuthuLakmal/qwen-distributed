from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from src.config.schema import TrackingConfig
from src.utils.dist_utils import get_rank
from src.utils.logger import logger
import mlflow

_MAX_PARAM_VALUE_LEN = 250
_PARAMS_PER_BATCH = 100


class MLflowTracker:
    """
    Tracks configs and training metrics (loss, lr, tokens/s, grad_norm) in
    MLflow. Only rank 0 talks to MLflow; every method is a no-op elsewhere
    or when tracking is disabled.

    GPU utilization is an infra/perf metric rather than an experiment
    result, so it's handled separately by PerfMetricsS3Uploader instead of
    being logged here.
    """

    def __init__(self, config: TrackingConfig, checkpoint_dir: str | Path) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self._active = False

    def start(self) -> None:
        if not self.config.enabled or get_rank() != 0:
            return

        try:
            import mlflow
        except ImportError:
            logger.warning("mlflow is not installed; tracking disabled.")
            return

        tracking_uri = self.config.mlflow_tracking_uri or (
            f"file:{os.path.join(str(self.checkpoint_dir), 'mlruns')}"
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        mlflow.start_run(run_name=self.config.mlflow_run_name)

        self._active = True

        logger.info(
            "MLflow tracking started | uri=%s | experiment=%s",
            tracking_uri,
            self.config.mlflow_experiment_name,
        )

    def log_params(self, **named_configs: Any) -> None:
        if not self._active:
            return

        params: dict[str, Any] = {}
        for prefix, config in named_configs.items():
            params.update(_flatten(config, prefix=f"{prefix}."))

        items = list(params.items())
        for i in range(0, len(items), _PARAMS_PER_BATCH):
            try:
                mlflow.log_params(dict(items[i : i + _PARAMS_PER_BATCH]))
            except Exception as exc:
                logger.warning("Failed to log MLflow params batch: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self._active:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as exc:
            logger.warning("Failed to log MLflow metrics at step %d: %s", step, exc)

    def finish(self) -> None:
        if not self._active:
            return

        import mlflow

        mlflow.end_run()
        self._active = False


def _flatten(config: Any, prefix: str = "") -> dict[str, Any]:
    values = asdict(config) if is_dataclass(config) else vars(config)

    params = {}
    for key, value in values.items():
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            value = str(value)
        if isinstance(value, str) and len(value) > _MAX_PARAM_VALUE_LEN:
            value = value[: _MAX_PARAM_VALUE_LEN - 3] + "..."
        params[f"{prefix}{key}"] = value

    return params
