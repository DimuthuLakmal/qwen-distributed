"""
Background NVML sampler for a long-running inference server.

Training's `PerfMetricsS3Uploader` assumes a tight, synchronous
per-optimizer-step loop -- "collect every Nth step" is a meaningful periodic
sample there because steps happen at a roughly constant cadence. An HTTP
inference server has no such cadence: request completions are bursty (many
can finish within the same millisecond under vLLM's continuous batching) and
can go idle for arbitrary stretches, so gating collection on request
completions would leave long blind spots and isn't safe to call concurrently
from multiple in-flight `generate()` coroutines anyway.

This sampler instead runs as an independent asyncio background task that
polls NVML on a fixed wall-clock interval, and periodically flushes
avg/min/max of the buffered samples -- decoupled entirely from request
traffic and safe under concurrency (a single task owns the buffers; no
request-handling coroutine touches them).

Utilization/power/temperature are only buffered while `is_active()` reports
at least one in-flight request, so long idle stretches (GPU util ~0) don't
drag the reported averages toward zero. Memory usage is buffered on every
poll regardless -- resident model weights and reserved KV-cache blocks keep
it elevated even at idle, and an idle-time memory creep is exactly the kind
of thing worth surfacing.

Each flush uploads its own small CSV to a timestamped S3 key rather than
re-uploading one ever-growing file (training's approach) -- an inference
server can run for days/weeks, and re-uploading the full history every flush
would grow unbounded.
"""

from __future__ import annotations

import asyncio
import csv
import io
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone

from src.config.schema import TrackingConfig
from src.utils.dist_utils import get_rank
from src.utils.logger import logger

CSV_HEADER = ["timestamp", "run_id", "metric_name", "stat", "metric_value", "gpu_index"]


class InferenceGPUMetricsSampler:
    def __init__(
        self,
        config: TrackingConfig,
        run_id: str,
        is_active: Callable[[], bool],
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.is_active = is_active
        self.poll_interval_seconds = poll_interval_seconds
        self.flush_interval_seconds = max(config.metrics_log_interval, 1)

        self._enabled = config.enabled and config.log_perf_metrics and get_rank() == 0
        self._s3 = None
        self._pynvml = None
        self._handles: list = []
        self._task: asyncio.Task | None = None
        self._buffers: dict[tuple[str, int], list[float]] = defaultdict(list)

    def start(self) -> None:
        if not self._enabled:
            return

        try:
            import boto3

            self._s3 = boto3.client("s3")
        except Exception as exc:
            logger.warning("Failed to init boto3 S3 client, disabling GPU metrics sampling: %s", exc)
            self._enabled = False
            return

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception as exc:
            logger.warning("NVML init failed, disabling GPU metrics sampling: %s", exc)
            self._enabled = False
            return

        self._task = asyncio.create_task(self._run())
        logger.info(
            "Background GPU metrics sampler started | poll_interval=%.1fs | flush_interval=%ds",
            self.poll_interval_seconds,
            self.flush_interval_seconds,
        )

    async def stop(self) -> None:
        """Cancel the background task, flushing whatever's buffered first."""
        if self._task is None:
            return

        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    async def _run(self) -> None:
        last_flush = time.monotonic()
        try:
            while True:
                await asyncio.sleep(self.poll_interval_seconds)
                self._poll_once()

                if time.monotonic() - last_flush >= self.flush_interval_seconds:
                    self._flush()
                    last_flush = time.monotonic()
        except asyncio.CancelledError:
            self._flush()
            raise

    def _poll_once(self) -> None:
        pynvml = self._pynvml
        active = self.is_active()

        for idx, handle in enumerate(self._handles):
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._buffers[("gpu.mem_used_gb", idx)].append(mem.used / (1024**3))
            except Exception as exc:
                logger.warning("Failed to read NVML memory for GPU %d: %s", idx, exc)

            # Utilization/power/temperature only mean something while a
            # request is actually being served -- see module docstring.
            if not active:
                continue

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._buffers[("gpu.utilization_pct", idx)].append(float(util.gpu))
                self._buffers[("gpu.mem_utilization_pct", idx)].append(float(util.memory))
            except Exception as exc:
                logger.warning("Failed to read NVML utilization for GPU %d: %s", idx, exc)

            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                self._buffers[("gpu.power_w", idx)].append(power_w)
            except Exception:
                pass

            try:
                temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                self._buffers[("gpu.temperature_c", idx)].append(temp_c)
            except Exception:
                pass

    def _flush(self) -> None:
        if not self._buffers:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        rows = []
        for (metric_name, gpu_index), values in self._buffers.items():
            if not values:
                continue
            rows.append([timestamp, self.run_id, metric_name, "avg", sum(values) / len(values), gpu_index])
            rows.append([timestamp, self.run_id, metric_name, "min", min(values), gpu_index])
            rows.append([timestamp, self.run_id, metric_name, "max", max(values), gpu_index])

        self._buffers.clear()

        if not rows or self._s3 is None:
            return

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

        key = f"{self.config.s3_prefix}/{self.run_id}/gpu_metrics/{timestamp}.csv"
        try:
            self._s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=buffer.getvalue().encode("utf-8"),
                ContentType="text/csv",
            )
        except Exception as exc:
            logger.warning(
                "Failed to upload GPU metrics CSV to s3://%s/%s: %s", self.config.s3_bucket, key, exc
            )