# test_vllm.py
import os
import sys
import shutil
from vllm import LLM, SamplingParams

from src.utils.dist_utils import get_rank


MODEL_ID = "Qwen/Qwen3.5-9B"

# vLLM / Hugging Face will put the downloaded model files under here.
MODEL_CACHE_DIR = "/home/student.unimelb.edu.au/dkariyawasan/llm_models"


def main():
    # If this script happens to be launched with multiple processes
    # (e.g. via torchrun), only let the main process (rank 0) run the
    # test. Every other process would otherwise print/initialize
    # vLLM redundantly, which makes debugging output very confusing.
    # if get_rank() != 0:
    #     print(f"[rank {get_rank()}] Skipping - only the main process runs this test.")
    #     return

    print("=== Environment check ===")
    print("Python:", sys.executable)
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
    print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    print(
        "FLASHINFER_EXTRA_LDFLAGS:",
        os.environ.get("FLASHINFER_EXTRA_LDFLAGS")
    )
    print("nvcc:", shutil.which("nvcc"))
    print("=========================")

    prompts = [
        """You are an expert in large language model inference systems. Consider a production inference server that hosts a transformer-based language model on multiple NVIDIA GPUs. The system receives many concurrent requests from users, and the prompts often contain a long shared system instruction followed by user-specific questions.
The inference engine uses a KV cache to store the key and value tensors produced by previous tokens so that these tensors do not need to be recomputed during autoregressive generation. The KV cache is divided into fixed-size blocks, and requests refer to these blocks through block tables rather than requiring contiguous GPU memory. The engine also supports continuous batching, where new requests may enter the running batch while other requests are still generating tokens.
Another optimization available in the inference engine is prefix caching. If two requests begin with exactly the same token sequence, the KV cache generated for the shared prefix may be reused by the later request rather than running the transformer layers again for those prefix tokens. The cache lookup depends on the token sequence and the corresponding KV cache blocks associated with that prefix.
Assume that the system instruction above is frequently reused by many requests and that prefix caching is enabled.
Explain how PagedAttention helps reduce KV-cache memory fragmentation in this inference system and why block-based KV-cache allocation is useful when serving many requests with different sequence lengths.""",
        """You are an expert in large language model inference systems. Consider a production inference server that hosts a transformer-based language model on multiple NVIDIA GPUs. The system receives many concurrent requests from users, and the prompts often contain a long shared system instruction followed by user-specific questions.
The inference engine uses a KV cache to store the key and value tensors produced by previous tokens so that these tensors do not need to be recomputed during autoregressive generation. The KV cache is divided into fixed-size blocks, and requests refer to these blocks through block tables rather than requiring contiguous GPU memory. The engine also supports continuous batching, where new requests may enter the running batch while other requests are still generating tokens.
Another optimization available in the inference engine is prefix caching. If two requests begin with exactly the same token sequence, the KV cache generated for the shared prefix may be reused by the later request rather than running the transformer layers again for those prefix tokens. The cache lookup depends on the token sequence and the corresponding KV cache blocks associated with that prefix.
Assume that the system instruction above is frequently reused by many requests and that prefix caching is enabled.
Explain how continuous batching improves GPU utilization in this inference system and how it differs from traditional static batching where all sequences in a batch are required to finish before the next batch can be processed.""",
        # "Introduce yourself briefly.",
        "Explain continuous batching in LLM inference.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=128,
    )

    print("Initializing vLLM engine...")

    llm = LLM(
        model=MODEL_ID,

        # Store downloaded checkpoints here.
        download_dir=MODEL_CACHE_DIR,

        # Online FP8 quantization.
        quantization="fp8_per_tensor",

        # L40S/L4 support BF16.
        dtype="bfloat16",

        # We only need text for this test.
        # Qwen3.5 is multimodal, so don't load the vision components.
        language_model_only=True,

        # Keep the initial experiment modest.
        max_model_len=8192,

        # Allow vLLM to use up to 90% of GPU memory for
        # weights + activations + KV cache.
        gpu_memory_utilization=0.80,

        # We have one GPU for this experiment.
        tensor_parallel_size=1,

        # More than enough for this small batch.
        max_num_seqs=8,
    )

    # Qwen3.5-Instruct is a chat/instruction model.
    # Using llm.chat() lets vLLM apply the model's chat template.
    print("\nRunning inference...")
    print("\n--- Test Results ---")

    for i, prompt in enumerate(prompts, start=1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")

        conversation = [{"role": "user", "content": prompt}]

        outputs = llm.chat(
            [conversation],
            sampling_params=sampling_params,
        )

        generated_text = outputs[0].outputs[0].text
        print(f"[Generated]: {generated_text}")


if __name__ == "__main__":
    main()