"""
GPU smoke test for TransformersQwenVLEngine against the real exported model.

NOTE: This loads the real Hugging Face artifact configured in
configs/inference.yaml (model.model_path) onto a real device and runs a real
generation -- run this on a compute node (srun/sbatch), not the shared login
node. If the exported model directory does not exist yet (export hasn't run),
the test is skipped rather than failed.

Run from the repo root with:
    python -m pytest test/inference/test_transformers_engine_gpu.py -v -s
or directly:
    python test/inference/test_transformers_engine_gpu.py
"""

from pathlib import Path

from PIL import Image

from src.config.configuration import ConfigurationManager
from src.inference.engines.transformers import TransformersQwenVLEngine
from src.inference.schema import GenerationConfig, InferenceRequest, ImageInput

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _load_engine_or_skip() -> TransformersQwenVLEngine:
    import pytest

    config_manager = ConfigurationManager(inference_config_path=CONFIGS_DIR / "inference.yaml")
    model_config = config_manager.get_inference_model_config()
    service_config = config_manager.get_inference_service_config()

    if not Path(model_config.model_path).exists():
        pytest.skip(
            f"Exported model not found at {model_config.model_path!r}; "
            "run the export step first, and run this test on a compute node."
        )

    return TransformersQwenVLEngine(model_config, model_version=service_config.model_version)


def test_real_engine_generates_text_for_image_and_prompt():
    engine = _load_engine_or_skip()

    image = Image.new("RGB", (64, 64), color="green")
    request = InferenceRequest(
        request_id="gpu-smoke-1",
        prompt="What color is this image?",
        images=[ImageInput(image=image)],
        generation=GenerationConfig(max_new_tokens=32, do_sample=False, num_beams=1),
    )

    response = engine.generate(request)

    assert response.generated_text
    assert response.input_token_count > 0
    assert response.output_token_count > 0

    print(f"\nlatency_seconds={response.latency_seconds:.3f}")
    print(f"input_token_count={response.input_token_count}")
    print(f"output_token_count={response.output_token_count}")
    print(f"generated_text={response.generated_text!r}")


if __name__ == "__main__":
    test_real_engine_generates_text_for_image_and_prompt()
    print("\nGPU inference engine smoke test passed.")
