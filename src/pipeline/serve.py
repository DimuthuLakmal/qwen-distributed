"""
Public inference entrypoint. Topology is chosen by `engine.type` in the
shared inference config:

    engine.type: "transformers"
        Starts the public router only. Loads no model weights and never
        imports `TransformersQwenVLEngine` -- it wires a
        `WorkerRegistry`/`WorkerClient`/`ReplicaRouter` from `workers:`/
        `routing:` and forwards public traffic to whichever GPU worker(s)
        (started separately via `src.pipeline.worker`, see
        `scripts/start_workers.sh`) are healthy.

    engine.type: "vllm"
        Starts a single, self-contained process: the public API served
        directly against a `VLLMEngine`, which owns its own internal
        multi-GPU data-parallel replica management. No router, no worker
        processes, no `scripts/start_workers.sh` -- this is the only
        process needed.

    python -m src.pipeline.serve --config configs/inference.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.config.configuration import ConfigurationManager
from src.config.schema import EngineConfig, InferenceServiceConfig
from src.utils.logger import logger


def main() -> None:
    args = parse_args()

    config_manager = ConfigurationManager(inference_config_path=args.config)

    service_config = config_manager.get_inference_service_config()
    engine_config = config_manager.get_engine_config()

    if engine_config.type == "transformers":
        _serve_transformers_router(config_manager, service_config)
    elif engine_config.type == "vllm":
        _serve_vllm(config_manager, service_config, engine_config)
    else:
        raise ValueError(f"Unknown engine.type {engine_config.type!r}.")


def _serve_transformers_router(
    config_manager: ConfigurationManager, service_config: InferenceServiceConfig
) -> None:
    from src.inference.transformers.router.app import create_router_app
    from src.inference.transformers.router.registry import WorkerDescriptor, WorkerRegistry
    from src.inference.transformers.router.replica_router import ReplicaRouter
    from src.inference.transformers.router.service import RouterService
    from src.inference.transformers.router.worker_client import WorkerClient

    limits = config_manager.get_inference_limits_config()
    routing_config = config_manager.get_routing_config()
    worker_endpoints = config_manager.get_worker_endpoints_config()

    registry = WorkerRegistry(
        [
            WorkerDescriptor(worker_id=endpoint.worker_id, base_url=endpoint.url)
            for endpoint in worker_endpoints
        ]
    )
    worker_client = WorkerClient.build(routing_config)
    replica_router = ReplicaRouter(registry, strategy=routing_config.strategy)
    router_service = RouterService(registry, worker_client, replica_router, limits)

    app = create_router_app(router_service)

    logger.info(
        "Starting Qwen-VL public router | host=%s | port=%d | workers=%s | strategy=%s",
        service_config.host,
        service_config.port,
        [endpoint.worker_id for endpoint in worker_endpoints],
        routing_config.strategy,
    )

    uvicorn.run(app, host=service_config.host, port=service_config.port)


def _serve_vllm(
    config_manager: ConfigurationManager,
    service_config: InferenceServiceConfig,
    engine_config: EngineConfig,
) -> None:
    from src.inference.engines.factory import build_engine
    from src.inference.vllm.app import create_vllm_app
    from src.tracking.inference_metrics import InferenceMetricsRecorder

    limits = config_manager.get_inference_limits_config()
    defaults = config_manager.get_generation_defaults_config()
    tracking_config = config_manager.get_inference_tracking_config()
    profiler_config = config_manager.get_inference_profiler_config()

    # engine.type == "vllm" here, so build_engine never touches the
    # transformers-only model config -- vLLM resolves its own model_id/
    # cache_dir/dtype from engine_config.vllm.
    engine_factory = lambda: build_engine(  # noqa: E731
        engine_config,
        model_config=None,
        model_version=service_config.model_version,
        profiler_config=profiler_config,
    )
    metrics = InferenceMetricsRecorder(tracking_config, model_version=service_config.model_version)

    app = create_vllm_app(engine_factory, limits, defaults, metrics=metrics)

    logger.info(
        "Starting Qwen-VL vLLM inference service | host=%s | port=%d | engine=%s",
        service_config.host,
        service_config.port,
        engine_config.type,
    )

    uvicorn.run(app, host=service_config.host, port=service_config.port)


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the Qwen-VL public inference API. Topology (application-managed "
            "router+workers vs. a single direct process) is chosen by engine.type "
            "in the config file."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the shared inference configuration file (e.g. configs/inference.yaml).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()