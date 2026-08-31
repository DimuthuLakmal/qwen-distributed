# CLAUDE.md

## Purpose

This repository already contains training, evaluation, Hugging Face export, and an inference service for a fine-tuned Qwen-VL model.

The active objective is to support two interchangeable inference backends:

```yaml
engine:
  type: "transformers"
```

or:

```yaml
engine:
  type: "vllm"
```

The public API and application-level inference contract must remain backend-independent.

The key architectural rule is:

> Transformers and vLLM do not need to use the same internal multi-GPU topology.

For Transformers, keep the existing application-managed replica workers and router.

For vLLM, prefer vLLM's own data-parallel replica management and internal request balancing when supported by the installed vLLM version.

---

# Working Rules

1. Inspect the existing repository and inference implementation before changing code.
2. Preserve working training and evaluation behavior.
3. Keep public request/response schemas independent of backend.
4. Keep `InferenceService` independent of concrete runtime classes.
5. Load only the selected backend during startup.
6. Do not use `device_map="auto"` as a substitute for replica serving.
7. Do not use DDP, FSDP, or `torch.nn.DataParallel` for request balancing.
8. Do not reimplement vLLM continuous batching, KV-cache management, or request scheduling.
9. Prefer vLLM data parallelism when the model fits on one GPU:
   - `tensor_parallel_size = 1`
   - `data_parallel_size = number_of_replicas`
10. Do not automatically consume every GPU unless explicitly configured.
11. An optional `data_parallel_size: auto` may resolve to visible GPU count, but explicit values are preferred.
12. Use the exported Hugging Face model artifact for inference.
13. Keep multimodal validation before model execution.
14. Do not log raw images, prompts, or generated responses by default.
15. Inspect the locally installed vLLM API/version before implementation.
16. Stop after the active stage and complete the `Review` section in `todo.md`.

---

# Target Architecture

## Transformers backend

```text
                         Public API
                             |
                             v
                      Replica Router
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
      Worker 0           Worker 1           Worker N
          |                  |                  |
          v                  v                  v
 TransformersEngine   TransformersEngine   TransformersEngine
          |                  |                  |
          v                  v                  v
        GPU 0              GPU 1              GPU N
```

Each process is explicitly assigned one GPU, typically through:

```bash
CUDA_VISIBLE_DEVICES=0 ...
CUDA_VISIBLE_DEVICES=1 ...
```

The existing Transformers routing implementation should remain intact unless a small refactor is required to make backend topology configurable.

## vLLM backend — preferred topology

```text
                  Public API
                      |
                      v
                 VLLM backend
                      |
                      v
        vLLM data-parallel coordinator
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
      DP rank 0   DP rank 1   DP rank N
          |           |           |
      full model  full model  full model
          |           |           |
        GPU 0       GPU 1       GPU N
```

For the current model, if it fits on one GPU:

```text
tensor_parallel_size = 1
data_parallel_size   = N
```

Do not create one application-managed vLLM worker per GPU.

For the vLLM backend, multi-GPU replication and request balancing must be owned by vLLM's internal data-parallel mechanism.

---

# vLLM Integration Modes

Before implementation, inspect the installed vLLM version and select the cleanest supported mode.

## Mode A — preferred

In-process asynchronous vLLM engine with internal data parallelism:

```text
FastAPI / service
      |
      v
VLLMEngine
      |
      v
Async vLLM engine
      |
      +-> DP replica 0 -> GPU 0
      +-> DP replica 1 -> GPU 1
      +-> DP replica N -> GPU N
```

Use this if the installed async Python API supports local `data_parallel_size > 1` cleanly.

## Mode B — fallback

Use a self-contained `vllm serve` backend with internal data parallelism:

```text
Public application API
        |
        v
vLLM backend client
        |
        v
vllm serve --data-parallel-size=N
        |
        v
vLLM internal DP
```

The public application remains the stable API boundary.

Only vLLM-managed internal data parallelism is allowed for the vLLM backend in the active stage.

Do **not** implement externally routed vLLM replicas.

The only acceptable vLLM integration choices are:

```text
Mode A — in-process async vLLM with internal DP
Mode B — application -> vllm serve with internal DP
```

Selection preference:

```text
A > B
```

If neither mode is supported cleanly by the installed vLLM version, stop and document the limitation rather than implementing an application-managed vLLM replica router.

---

# Backend Selection

Backend selection must be configuration-driven:

```yaml
engine:
  type: "transformers"
```

or:

```yaml
engine:
  type: "vllm"
```

Recommended vLLM configuration shape:

```yaml
engine:
  type: "vllm"

  vllm:
    data_parallel_size: 4
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    max_num_seqs: 16

    limit_mm_per_prompt:
      image: 4
      video: 0

    mm_processor_kwargs:
      max_pixels: 1003520
```

Optional:

```yaml
data_parallel_size: "auto"
```

If `auto` is implemented:

- resolve only visible GPUs,
- log the resolved value,
- allow an explicit integer to override it,
- fail clearly when requested parallelism is incompatible with available resources.

---

# Engine Contract

Use the existing engine abstraction where possible.

Conceptually:

```python
class InferenceEngine(ABC):

    async def start(self) -> None:
        ...

    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        ...

    async def health(self) -> bool:
        ...

    async def shutdown(self) -> None:
        ...
```

If the existing implementation already has an equivalent lifecycle, preserve it rather than introducing unnecessary refactoring.

Backend-specific types must remain inside the engine layer.

---

# Concurrency

## Transformers

If the existing Transformers backend serializes model execution for correctness, keep that behavior backend-specific.

## vLLM

Do not place a global application/service lock around vLLM generation.

Multiple requests must be allowed to enter the async vLLM engine so vLLM can perform:

- continuous/in-flight batching,
- request scheduling,
- KV-cache management,
- replica selection within its data-parallel topology.

The application must not duplicate those responsibilities.

---

# Multimodal Input Boundary

Both backends should consume the same semantic request:

```text
InferenceRequest
    |
    v
validated canonical Qwen messages
    |
    +--------------------+
    |                    |
    v                    v
TransformersEngine     VLLMEngine
    |                    |
HF processor           render Qwen prompt
tokenization           process visual inputs
image processing       construct multi_modal_data
    |                    |
torch tensors          vLLM PromptType
```

Do not pass Transformers tensors into vLLM.

For Qwen-VL, inspect the installed integration and use the supported processing path, including `process_vision_info(messages)` if required.

---

# Generation Parameter Parity

Map the canonical generation configuration explicitly.

Examples:

```text
canonical max_new_tokens
  -> Transformers: max_new_tokens
  -> vLLM: SamplingParams.max_tokens

temperature
  -> both backends

top_p
  -> both backends

top_k
  -> both backends when supported
```

Do not allow different hidden backend defaults to change service semantics accidentally.

The service-level generation configuration should remain authoritative.

---

# Health, Cancellation, and Shutdown

The vLLM backend may own background engine-core processes.

The implementation must:

- report not-ready if the engine is unhealthy,
- surface startup failure clearly,
- cancel/abort abandoned requests when necessary,
- shut down background resources cleanly,
- avoid orphan child processes,
- release GPU resources.

---

# Intended Repository Structure

Adapt this to the existing repository rather than forcing duplicate modules.

```text
src/
├── inference/
│   ├── config.py
│   ├── schema.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   ├── service.py
│   └── engines/
│       ├── base.py
│       ├── factory.py
│       ├── transformers.py
│       └── vllm.py
│
├── api/
│   └── ...
│
└── pipeline/
    ├── serve.py
    └── worker.py

configs/
└── inference.yaml

tests/
├── inference/
│   └── engines/
└── api/
```

Do not automatically add one vLLM worker module/process per GPU unless the selected integration mode genuinely requires it.

---

# Active Stage — Stage 2: Pluggable vLLM Backend

## Goal

Add vLLM as an alternative inference backend while preserving the existing Transformers path.

Desired result:

```text
engine.type=transformers
    |
    v
existing application-managed GPU replicas


engine.type=vllm
    |
    v
vLLM-managed data-parallel replicas
```

The public API must remain unchanged.

## Required Outcome

Stage 2 is complete when:

- Transformers still works,
- vLLM works with the exported Qwen-VL artifact,
- backend selection is configuration-driven,
- vLLM multi-GPU DP works exclusively through vLLM's internal data-parallel mechanism,
- concurrent requests reach vLLM asynchronously,
- vLLM internal request balancing is demonstrated,
- health/failure behavior is understood,
- parity tests pass,
- a baseline benchmark compares both backends.

The detailed tasks are in `todo.md`.

---

# Out of Scope for This Stage

Do not implement unless explicitly requested:

- public token streaming,
- SSE/WebSockets,
- tensor parallelism,
- pipeline parallelism,
- speculative decoding,
- quantized serving,
- LoRA hot loading,
- custom continuous batching,
- custom KV-cache management,
- token-aware cross-node routing,
- aggressive prefix-cache tuning,
- multimodal cache optimization,
- Kubernetes/autoscaling,
- canary model routing,
- full Prometheus/OpenTelemetry integration.

---

# Stage Completion Definition

The active stage is complete only when:

1. all applicable tasks in `todo.md` are complete,
2. Transformers regression tests pass,
3. vLLM works on real GPU hardware,
4. multi-GPU vLLM DP behavior is verified,
5. backend switching requires configuration only,
6. no application-managed per-GPU vLLM worker/router layer was introduced,
7. known limitations are documented,
8. the `Review` section in `todo.md` is complete,
9. work stops for review before the next stage.
