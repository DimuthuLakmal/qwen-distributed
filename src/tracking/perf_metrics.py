from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from src.config.schema import TrackingConfig
from src.utils.dist_utils import get_rank
from src.utils.logger import logger


class PerfMetricsS3Uploader:
    """
    Collects per-GPU NVML utilization/memory/power/temperature every
    `metrics_log_interval` optimizer steps and re-uploads them as one
    long-format CSV to S3, so Grafana (via an S3 CSV/JSON datasource
    plugin) can visualize them.

    These are infra/perf metrics, not experiment results, so they're kept
    out of MLflow. Only rank 0 collects and uploads; assumes a single-node
    job so NVML on rank 0 can see every GPU regardless of
    CUDA_VISIBLE_DEVICES.

    AWS credentials are read by boto3 from the environment
    (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION) --
    never hardcode them here.
    """

    CSV_HEADER = ["timestamp", "run_id", "optimizer_step", "metric_name", "metric_value", "gpu_index"]

    def __init__(self, config: TrackingConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.s3_key = f"{config.s3_prefix}/{run_id}/perf_metrics.csv"

        self._enabled = config.enabled and config.log_perf_metrics and get_rank() == 0
        self._s3 = None
        self._pynvml = None
        self._handles: list = []
        self._rows: list[list] = []

    def start(self) -> None:
        if not self._enabled:
            return

        try:
            import boto3

            self._s3 = boto3.client("s3")
        except Exception as exc:
            logger.warning("Failed to init boto3 S3 client, disabling perf-metrics upload: %s", exc)
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
            logger.warning("NVML init failed, disabling GPU metrics: %s", exc)
            self._enabled = False

    def maybe_collect(self, optimizer_step: int) -> None:
        if not self._enabled:
            return

        if optimizer_step == 0 or optimizer_step % self.config.metrics_log_interval != 0:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        self._collect_gpu_rows(timestamp, optimizer_step)
        self._upload()

    def _collect_gpu_rows(self, timestamp: str, step: int) -> None:
        pynvml = self._pynvml
        for idx, handle in enumerate(self._handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._rows.append([timestamp, self.run_id, step, "gpu.utilization_pct", float(util.gpu), idx])
                self._rows.append([timestamp, self.run_id, step, "gpu.mem_utilization_pct", float(util.memory), idx])
                self._rows.append([timestamp, self.run_id, step, "gpu.mem_used_gb", mem.used / (1024**3), idx])
                self._rows.append([timestamp, self.run_id, step, "gpu.mem_total_gb", mem.total / (1024**3), idx])
            except Exception as exc:
                logger.warning("Failed to read NVML utilization/memory for GPU %d: %s", idx, exc)
                continue
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                self._rows.append([timestamp, self.run_id, step, "gpu.power_w", power_w, idx])
            except Exception:
                pass
            try:
                temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                self._rows.append([timestamp, self.run_id, step, "gpu.temperature_c", temp_c, idx])
            except Exception:
                pass

    def _upload(self) -> None:
        if self._s3 is None or not self._rows:
            return

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.CSV_HEADER)
        writer.writerows(self._rows)

        try:
            self._s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=self.s3_key,
                Body=buffer.getvalue().encode("utf-8"),
                ContentType="text/csv",
            )
        except Exception as exc:
            logger.warning(
                "Failed to upload perf metrics CSV to s3://%s/%s: %s",
                self.config.s3_bucket,
                self.s3_key,
                exc,
            )

    def shutdown(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
