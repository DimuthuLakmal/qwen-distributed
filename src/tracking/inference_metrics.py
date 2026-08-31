"""
Inference-side efficiency-metrics recorder.

Reuses `MLflowTracker` for per-request metrics and composes
`InferenceGPUMetricsSampler` for GPU utilization/memory -- see that module's
docstring for why GPU sampling needs its own background asyncio task instead
of training's per-step `PerfMetricsS3Uploader.maybe_collect()`.

Deliberately tracks efficiency only (latency, throughput, GPU utilization) --
never accuracy/quality, which has no meaning for a served request. "Step" for
`log_metrics` purposes is the running request count, the inference analogue
of a training optimizer step.

No-ops end-to-end when `tracking_config.enabled` is False, so callers never
need to branch on whether tracking is configured.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config.schema import TrackingConfig
from src.tracking.inference_gpu_sampler import InferenceGPUMetricsSampler
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.logger import logger

# Local MLflow file-store fallback when mlflow_tracking_uri isn't set --
# mirrors training's <checkpoint.directory>/mlruns convention, but inference
# has no checkpoint dir, so this is a fixed results/ location instead.
_LOCAL_MLFLOW_DIR = Path("results/inference")


class InferenceMetricsRecorder:
    """
    Records per-request efficiency metrics (latency, token counts,
    tokens/second) plus periodic GPU utilization, gated by
    `tracking_config.enabled`. One instance lives for the lifetime of a
    serving process (single-process vLLM app, or one Transformers worker).
    """

    def __init__(self, tracking_config: TrackingConfig, model_version: str) -> None:
        self.tracking_config = tracking_config
        self.run_id = tracking_config.mlflow_run_name or f"{model_version}-{uuid4().hex[:8]}"
        self._request_count = 0

        self._mlflow = MLflowTracker(config=tracking_config, checkpoint_dir=_LOCAL_MLFLOW_DIR)
        self._gpu_sampler: InferenceGPUMetricsSampler | None = None

    def start(self, is_active: Callable[[], bool], **static_params: Any) -> None:
        """
        `is_active` should report whether at least one request is currently
        in-flight (e.g. `lambda: inference_service.active_request_count > 0`)
        -- passed in rather than taken at construction time since the
        recorder is typically built before the `InferenceService` it reports
        on exists.
        """
        if not self.tracking_config.enabled:
            return

        self._mlflow.start()
        if static_params:
            self._mlflow.log_params(**static_params)

        self._gpu_sampler = InferenceGPUMetricsSampler(
            config=self.tracking_config, run_id=self.run_id, is_active=is_active
        )
        self._gpu_sampler.start()

        logger.info("Inference efficiency-metrics tracking started | run_id=%s", self.run_id)

    def record_request(self, *, input_tokens: int, output_tokens: int, latency_seconds: float) -> None:
        if not self.tracking_config.enabled:
            return

        self._request_count += 1
        tokens_per_second = output_tokens / max(latency_seconds, 1e-8)

        self._mlflow.log_metrics(
            {
                "inference/latency_seconds": latency_seconds,
                "inference/input_tokens": input_tokens,
                "inference/output_tokens": output_tokens,
                "inference/tokens_per_second": tokens_per_second,
            },
            step=self._request_count,
        )

    async def finish(self) -> None:
        if not self.tracking_config.enabled:
            return

        if self._gpu_sampler is not None:
            await self._gpu_sampler.stop()

        self._mlflow.finish()