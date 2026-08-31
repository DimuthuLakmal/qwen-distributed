"""
GPU worker entrypoint.

One worker process owns one full Qwen-VL model replica on one physical GPU.
Assign the GPU explicitly via `CUDA_VISIBLE_DEVICES` and use the default
`cuda:0` (or whatever `transformers.device` is configured to) inside the process --
do not rely on multiple worker processes auto-distributing themselves across
GPUs.

    CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.worker \\
        --config configs/inference.yaml --worker-id gpu-0 --port 8100

    CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.worker \\
        --config configs/inference.yaml --worker-id gpu-1 --port 8101

See `scripts/start_workers.sh` for launching several workers at once.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from src.config.configuration import ConfigurationManager
from src.inference.engines.transformers import TransformersQwenVLEngine
from src.inference.transformers.worker_app.app import create_worker_app
from src.tracking.inference_metrics import InferenceMetricsRecorder
from src.utils.logger import logger


def main() -> None:
    args = parse_args()

    config_manager = ConfigurationManager(inference_config_path=args.config)

    service_config = config_manager.get_inference_service_config()
    model_config = config_manager.get_inference_transformers_model_config()
    generation_defaults = config_manager.get_generation_defaults_config()
    limits = config_manager.get_inference_limits_config()
    tracking_config = config_manager.get_inference_tracking_config()
    profiler_config = config_manager.get_inference_profiler_config()

    host = args.host or service_config.host
    port = args.port or int(os.environ.get("WORKER_PORT", 0)) or service_config.port

    def engine_factory() -> TransformersQwenVLEngine:
        return TransformersQwenVLEngine(
            model_config, model_version=service_config.model_version, profiler_config=profiler_config
        )

    # model_version is suffixed with worker_id so each worker process falls
    # back to its own MLflow run/S3 key when tracking.mlflow_run_name isn't
    # explicitly set -- otherwise multiple workers on one host would collide.
    metrics = InferenceMetricsRecorder(
        tracking_config, model_version=f"{service_config.model_version}-{args.worker_id}"
    )

    app = create_worker_app(
        worker_id=args.worker_id,
        engine_factory=engine_factory,
        limits=limits,
        defaults=generation_defaults,
        metrics=metrics,
    )

    logger.info(
        "Starting Qwen-VL worker | worker_id=%s | host=%s | port=%d | model_version=%s",
        args.worker_id,
        host,
        port,
        service_config.model_version,
    )

    uvicorn.run(app, host=host, port=port)


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a single GPU worker's internal Qwen-VL inference API."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to the shared inference configuration file (e.g. configs/inference.yaml). "
            "The worker reads service/transformers/generation/limits and ignores workers/routing."
        ),
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=os.environ.get("WORKER_ID"),
        help="Unique worker identifier (e.g. gpu-0). Falls back to $WORKER_ID.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Overrides service.port. Falls back to $WORKER_PORT, then service.port.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Overrides service.host.",
    )

    args = parser.parse_args()

    if not args.worker_id:
        parser.error("--worker-id is required (or set $WORKER_ID).")

    return args


if __name__ == "__main__":
    main()
