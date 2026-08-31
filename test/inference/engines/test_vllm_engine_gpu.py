"""
GPU smoke test for VLLMEngine against the real exported model, using vLLM's
internal single-GPU path (data_parallel_size=1, tensor_parallel_size=1) --
see todo.md Task 13.

NOTE: Requires the `vllm` package (run in the `vllm` conda env) and a real
GPU. If either `vllm` isn't importable or the exported model directory
doesn't exist yet, the test is skipped rather than failed, so it's safe to
collect from any environment.

Run from the repo root with:
    python -m pytest test/inference/engines/test_vllm_engine_gpu.py -v -s
or directly:
    python test/inference/engines/test_vllm_engine_gpu.py
"""

import asyncio
from pathlib import Path

from PIL import Image

from src.config.configuration import ConfigurationManager
from src.inference.schema import GenerationConfig, ImageInput, InferenceRequest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _load_engine_or_skip():
    import pytest

    try:
        from src.inference.engines.vllm import VLLMEngine
    except ImportError:
        pytest.skip("vllm is not installed in this environment; run in the `vllm` conda env.")

    config_manager = ConfigurationManager(inference_config_path=CONFIGS_DIR / "inference.yaml")
    engine_config = config_manager.get_engine_config()
    service_config = config_manager.get_inference_service_config()

    if engine_config.vllm is None:
        pytest.skip("engine.type is not 'vllm' in configs/inference.yaml.")

    return VLLMEngine(
        engine_config.vllm,
        model_version=service_config.model_version,
        data_parallel_size=1,
    )


def test_real_engine_generates_text_for_image_and_prompt():
    engine = _load_engine_or_skip()

    image = Image.new("RGB", (64, 64), color="green")
    request = InferenceRequest(
        request_id="vllm-gpu-smoke-1",
        prompt="What color is this image?",
        images=[ImageInput(image=image)],
        generation=GenerationConfig(max_new_tokens=32, do_sample=False, num_beams=1),
    )

    response = asyncio.run(engine.generate(request))

    assert response.generated_text
    assert response.input_token_count > 0
    assert response.output_token_count > 0

    print(f"\nlatency_seconds={response.latency_seconds:.3f}")
    print(f"input_token_count={response.input_token_count}")
    print(f"output_token_count={response.output_token_count}")
    print(f"generated_text={response.generated_text!r}")

    asyncio.run(engine.close())


if __name__ == "__main__":
    test_real_engine_generates_text_for_image_and_prompt()
    print("\nvLLM GPU inference engine smoke test passed.")