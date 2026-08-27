from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.api.app import create_app
from src.config.configuration import ConfigurationManager
from src.inference.engines.transformers import TransformersQwenVLEngine
from src.utils.logger import logger


def main() -> None:
    args = parse_args()

    config_manager = ConfigurationManager(inference_config_path=args.config)

    service_config = config_manager.get_inference_service_config()
    model_config = config_manager.get_inference_model_config()
    generation_defaults = config_manager.get_generation_defaults_config()
    limits = config_manager.get_inference_limits_config()

    def engine_factory() -> TransformersQwenVLEngine:
        return TransformersQwenVLEngine(model_config, model_version=service_config.model_version)

    app = create_app(engine_factory=engine_factory, limits=limits, defaults=generation_defaults)

    logger.info(
        "Starting Qwen-VL inference service | host=%s | port=%d | model_version=%s",
        service_config.host,
        service_config.port,
        service_config.model_version,
    )

    uvicorn.run(app, host=service_config.host, port=service_config.port)


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the Qwen-VL inference API from an exported Hugging Face artifact."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the inference configuration file (e.g. configs/inference.yaml).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
