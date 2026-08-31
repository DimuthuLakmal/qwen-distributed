"""
Public router entrypoint.

Loads no model weights and never imports `TransformersQwenVLEngine` -- it
only wires a `WorkerRegistry`/`WorkerClient`/`ReplicaRouter` from
`workers:`/`routing:` in the shared inference config and forwards public
traffic to whichever GPU worker(s) are healthy. Workers are started
separately via `src.pipeline.worker` (see `scripts/start_workers.sh`).

    python -m src.pipeline.serve --config configs/inference.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.config.configuration import ConfigurationManager
from src.inference.router.app import create_router_app
from src.inference.router.registry import WorkerDescriptor, WorkerRegistry
from src.inference.router.replica_router import ReplicaRouter
from src.inference.router.service import RouterService
from src.inference.router.worker_client import WorkerClient
from src.utils.logger import logger


def main() -> None:
    args = parse_args()

    config_manager = ConfigurationManager(inference_config_path=args.config)

    service_config = config_manager.get_inference_service_config()
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


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the Qwen-VL public router, forwarding requests to configured GPU workers."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to the shared inference configuration file (e.g. configs/inference.yaml). "
            "The router reads service/limits/workers/routing and ignores model/generation."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
