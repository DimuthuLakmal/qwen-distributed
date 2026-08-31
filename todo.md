# Stage 2 TODO — vLLM Backend with Internal Multi-GPU Data Parallelism

## Goal

Add **vLLM** as a selectable inference backend alongside the existing Transformers backend.

The main architectural requirement is:

> Do not automatically create one application-managed vLLM worker per GPU.

For vLLM, prefer vLLM's own data-parallel replica management and internal request balancing.

Target behavior:

```yaml
engine:
  type: "transformers"
```

uses the existing application-managed replica topology.

```yaml
engine:
  type: "vllm"
```

uses:

```text
one logical vLLM backend
  -> data_parallel_size = N
  -> N full model replicas
  -> vLLM internal request balancing
  -> one replica per GPU
```

Assume the current model fits on one GPU:

```text
tensor_parallel_size = 1
data_parallel_size   = N
```

Do not implement tensor parallelism in this stage.

---

# Task 1 — Inspect the Existing Inference Stack

- [ ] Inspect the current `InferenceEngine` contract.
- [ ] Inspect `TransformersEngine`.
- [ ] Inspect `InferenceService`.
- [ ] Inspect the public API and current worker/router topology.
- [ ] Identify which routing components are Transformers-specific.
- [ ] Identify any code that incorrectly assumes every backend uses application-managed workers.
- [ ] Inspect FastAPI startup/lifespan behavior.
- [ ] Inspect multimodal request validation.
- [ ] Inspect canonical Qwen message construction.
- [ ] Inspect generation configuration.
- [ ] Inspect token accounting.
- [ ] Inspect existing tests.

### Completion criteria

Document which components remain shared and which topology becomes backend-specific.

---

# Task 2 — Inspect the Installed vLLM Version and Select the DP Integration Mode

Before implementation:

- [ ] Record `vllm.__version__`.
- [ ] Inspect locally installed async engine APIs.
- [ ] Verify `AsyncLLM`, `AsyncLLMEngine`, `AsyncEngineArgs`, or installed equivalents.
- [ ] Verify whether the installed in-process async API supports local `data_parallel_size > 1`.
- [ ] Verify how DP ranks are launched/coordinated.
- [ ] Verify Qwen-VL support.
- [ ] Verify multimodal input format.
- [ ] Verify health API.
- [ ] Verify request abort/cancellation API.
- [ ] Verify shutdown API.
- [ ] Verify `qwen_vl_utils` / `process_vision_info` if required.
- [ ] Run a minimal import/API smoke test.

Select one integration mode:

```text
Mode A — in-process async vLLM + internal DP
Mode B — application -> vllm serve + internal DP
```

Selection preference:

```text
A > B
```

Externally routed vLLM replicas are **not allowed** in this stage.

If neither Mode A nor Mode B works with the installed vLLM version, stop and document the limitation instead of implementing a custom vLLM replica router.

### Completion criteria

Record the chosen mode and the reason in the Review section.

---

# Task 3 — Add Typed Backend and Parallelism Configuration

Extend the existing config system.

Required:

```yaml
engine:
  type: "vllm"
```

Suggested vLLM section:

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

Validation:

- [ ] `engine.type` supports `transformers` and `vllm`.
- [ ] `tensor_parallel_size` defaults to `1`.
- [ ] `data_parallel_size` accepts a positive integer.
- [ ] Optional `auto` resolves only visible GPUs.
- [ ] Resolved DP size is logged.
- [ ] Explicit DP size is validated against visible GPU resources.
- [ ] `gpu_memory_utilization` is validated.
- [ ] `max_model_len > 0`.
- [ ] `max_num_seqs > 0`.
- [ ] multimodal limits align with public service limits.
- [ ] only selected-backend config is required.

### Completion criteria

Multi-GPU vLLM behavior is configuration-driven.

---

# Task 4 — Add or Refine the Backend Factory

- [ ] Create/refine a single backend factory.
- [ ] Preserve `TransformersEngine`.
- [ ] Add `VLLMEngine`.
- [ ] Avoid loading both runtimes.
- [ ] Keep routes backend-independent.
- [ ] Keep `InferenceService` backend-independent.
- [ ] fail fast on unknown backend.
- [ ] Add factory tests.

### Completion criteria

Switching backend requires configuration only.

---

# Task 5 — Make Replica Management Backend-Specific

Refactor only where needed so these topologies can coexist:

```text
Transformers:
public API -> application router -> per-GPU Transformers workers

vLLM:
public API -> vLLM backend -> vLLM internal DP
```

- [ ] Identify universal worker assumptions.
- [ ] Remove unnecessary worker/router coupling from the vLLM path.
- [ ] Keep existing Transformers routing unchanged.
- [ ] Keep public API unchanged.
- [ ] Keep request/response schemas unchanged.
- [ ] Keep request ID behavior unchanged.
- [ ] Add topology-level tests with fakes.

### Completion criteria

The application no longer assumes every GPU replica is an application worker.

---

# Task 6 — Preserve Shared Semantic Multimodal Preprocessing

Shared:

```text
request validation
canonical InferenceRequest
canonical Qwen messages
```

Backend-specific:

```text
Transformers -> processor/tensors
vLLM        -> prompt + multi_modal_data
```

Tasks:

- [ ] reuse existing validators,
- [ ] reuse canonical Qwen messages,
- [ ] keep API/service free of Transformers tensors,
- [ ] preserve image order,
- [ ] preserve system prompt behavior,
- [ ] preserve media limits,
- [ ] avoid unnecessary broad refactoring.

---

# Task 7 — Implement `VLLMEngine` Initialization

According to the selected internal-DP integration mode (A or B only):

- [ ] load exported fine-tuned Qwen-VL artifact,
- [ ] configure dtype,
- [ ] configure `data_parallel_size`,
- [ ] configure `tensor_parallel_size=1`,
- [ ] configure `gpu_memory_utilization`,
- [ ] configure `max_model_len`,
- [ ] configure `max_num_seqs`,
- [ ] configure multimodal limits,
- [ ] configure processor kwargs where applicable,
- [ ] do not initialize a second Transformers model,
- [ ] do not call `model.to(...)` for vLLM,
- [ ] surface startup failures clearly,
- [ ] add initialization tests.

### Completion criteria

Start successfully with:

```text
data_parallel_size = 1
tensor_parallel_size = 1
```

before enabling multi-GPU DP.

---

# Task 8 — Implement Qwen-VL Multimodal Conversion for vLLM

- [ ] render Qwen chat template with `add_generation_prompt=True`,
- [ ] ensure no target assistant answer is included,
- [ ] process visual inputs using the installed supported path,
- [ ] use `process_vision_info(messages)` if required,
- [ ] construct the installed vLLM version's multimodal prompt type,
- [ ] preserve multi-image order,
- [ ] keep public media limits enforced,
- [ ] leave video disabled unless already supported,
- [ ] add conversion tests.

Conceptually:

```python
{
    "prompt": rendered_prompt,
    "multi_modal_data": {
        "image": image_or_images,
    },
}
```

---

# Task 9 — Map Canonical Generation Config to vLLM

- [ ] `max_new_tokens` -> `SamplingParams.max_tokens`,
- [ ] map `temperature`,
- [ ] map `top_p`,
- [ ] map `top_k` if exposed,
- [ ] map stop/repetition settings only if already part of the service contract,
- [ ] preserve greedy behavior,
- [ ] avoid hidden backend defaults,
- [ ] add mapping tests.

### Completion criteria

Both backends have the same intended generation semantics.

---

# Task 10 — Implement Async vLLM Generation

- [ ] use a unique vLLM request ID,
- [ ] preserve the service request ID,
- [ ] call the installed asynchronous generation API,
- [ ] iterate until final output for the current non-streaming API,
- [ ] return generated continuation text,
- [ ] compute input token count,
- [ ] compute output token count,
- [ ] measure latency,
- [ ] preserve model/version metadata,
- [ ] preserve the existing response schema,
- [ ] sanitize backend errors,
- [ ] add fake async-engine tests.

Do not:

- [ ] expose vLLM internal response objects,
- [ ] add public streaming,
- [ ] add a custom batching queue.

---

# Task 11 — Implement vLLM Lifecycle, Health, and Cancellation

- [ ] clean startup,
- [ ] health/readiness integration,
- [ ] engine-death handling,
- [ ] request cancellation,
- [ ] abort abandoned requests if required,
- [ ] clean shutdown,
- [ ] no orphan engine-core/DP processes,
- [ ] lifecycle tests.

### Completion criteria

Start, serve, health-check, cancel, and stop predictably.

---

# Task 12 — Ensure the Application Does Not Serialize vLLM Requests

- [ ] inspect service-level locks/semaphores,
- [ ] retain Transformers-specific serialization only if needed,
- [ ] do not globally serialize vLLM,
- [ ] allow overlapping `VLLMEngine.generate()` calls,
- [ ] keep active-request accounting correct,
- [ ] add concurrency tests.

### Completion criteria

Multiple concurrent requests can enter vLLM so internal continuous batching is possible.

---

# Task 13 — Verify Single-GPU vLLM

Use:

```text
data_parallel_size = 1
tensor_parallel_size = 1
```

- [ ] start backend,
- [ ] verify model load,
- [ ] verify image+text generation,
- [ ] verify readiness,
- [ ] verify public API path,
- [ ] verify response fields,
- [ ] verify shutdown,
- [ ] record GPU memory,
- [ ] record baseline latency.

---

# Task 14 — Enable vLLM Internal Data Parallelism

Test with at least two GPUs:

```text
data_parallel_size = 2
tensor_parallel_size = 1
```

- [ ] expose two GPUs to the vLLM backend,
- [ ] start one logical vLLM backend,
- [ ] confirm vLLM creates two full model replicas,
- [ ] confirm each replica owns one GPU,
- [ ] confirm no application-managed per-GPU vLLM workers or replica router were added,
- [ ] confirm one logical endpoint serves both replicas,
- [ ] send concurrent requests,
- [ ] verify requests are distributed across DP ranks,
- [ ] verify each DP rank schedules its own work,
- [ ] record memory usage per GPU,
- [ ] document how the installed version exposes DP-rank activity.

### Completion criteria

vLLM itself owns replica creation and request balancing.

---

# Task 15 — Verify Optional `data_parallel_size: auto`

If auto mode is implemented:

- [ ] test one visible GPU -> DP=1,
- [ ] test two visible GPUs -> DP=2,
- [ ] log the resolved value,
- [ ] verify explicit integer overrides auto,
- [ ] fail clearly on inconsistent resources.

Do not make auto mandatory.

---

# Task 16 — Verify Backend-Specific Routing Behavior

Transformers:

- [ ] existing application router still works,
- [ ] one-process-per-GPU workers still work.

vLLM:

- [ ] application does not create per-GPU vLLM workers,
- [ ] application does not implement a vLLM replica router,
- [ ] vLLM DP coordinator owns replica selection,
- [ ] public API is unchanged.

### Completion criteria

```text
Transformers -> application-managed replica routing
vLLM         -> vLLM-managed replica routing
```

---

# Task 17 — Backend Contract and Golden-Set Parity

Create a small reusable golden set containing:

- [ ] one image + text,
- [ ] multiple images if supported,
- [ ] optional system prompt,
- [ ] deterministic generation,
- [ ] bounded output length.

Verify for both:

- [ ] identical public schema,
- [ ] request ID preserved,
- [ ] model version present,
- [ ] generated text non-empty,
- [ ] token counts sensible,
- [ ] identical validation behavior.

If deterministic output differs:

- [ ] verify prompt rendering,
- [ ] verify image order,
- [ ] verify chat template,
- [ ] verify generation defaults,
- [ ] document accepted differences only after those checks.

---

# Task 18 — Transformers Regression

- [ ] run existing Transformers tests,
- [ ] verify worker startup,
- [ ] verify replica router,
- [ ] verify single/multi-GPU behavior,
- [ ] verify health,
- [ ] verify shutdown,
- [ ] verify vLLM changes did not break Transformers.

---

# Task 19 — Verify vLLM DP Failure and Health Behavior

- [ ] determine behavior when one DP rank fails,
- [ ] determine health propagation,
- [ ] verify public readiness semantics,
- [ ] verify failed backend returns sanitized errors,
- [ ] ensure requests do not hang indefinitely,
- [ ] document whether replica recovery is automatic or requires backend restart.

Do not build custom DP-replica recovery unless necessary.

---

# Task 20 — Benchmark Transformers vs vLLM

Use the same:

- model artifact,
- GPU type,
- prompts,
- images,
- generation settings,
- warmup procedure.

Measure:

- [ ] startup time,
- [ ] GPU memory per replica,
- [ ] single-request latency,
- [ ] concurrent throughput,
- [ ] output tokens/sec where comparable,
- [ ] requests/sec,
- [ ] error count.

Suggested concurrency:

```text
1
2
4
8
```

For vLLM, compare:

```text
DP=1
DP=2
```

where hardware permits.

Do not turn this into a tuning project.

---

# Task 21 — Update Documentation

Document:

- [ ] backend selection,
- [ ] installed vLLM version,
- [ ] chosen integration mode,
- [ ] `data_parallel_size`,
- [ ] `tensor_parallel_size`,
- [ ] single-GPU vLLM launch,
- [ ] multi-GPU vLLM DP launch,
- [ ] optional auto mode,
- [ ] Transformers topology,
- [ ] vLLM topology,
- [ ] how to switch backends,
- [ ] known limitations.

Explicitly explain:

```text
Transformers:
application creates/routes replicas

vLLM:
vLLM creates/routes replicas
```

---

# Task 22 — Full End-to-End Verification

- [ ] run vLLM unit tests,
- [ ] run factory tests,
- [ ] run public API tests,
- [ ] run Transformers regression tests,
- [ ] run lint/type/static checks,
- [ ] run real single-GPU vLLM test,
- [ ] run real multi-GPU vLLM DP test,
- [ ] run concurrency test,
- [ ] run golden-set parity test,
- [ ] run baseline benchmark,
- [ ] confirm only selected backend is loaded,
- [ ] confirm no per-GPU vLLM application workers or external vLLM replica router exist,
- [ ] confirm no raw prompts/images are logged,
- [ ] confirm no orphan vLLM processes remain after shutdown.

---

# Out of Scope

Do not implement in this stage:

- [ ] externally routed vLLM replicas,
- [ ] application-managed per-GPU vLLM workers,
- [ ] custom vLLM replica load balancer,

- [ ] public token streaming,
- [ ] SSE/WebSockets,
- [ ] tensor parallelism,
- [ ] pipeline parallelism,
- [ ] speculative decoding,
- [ ] quantized serving,
- [ ] LoRA hot swapping,
- [ ] custom continuous batching,
- [ ] custom KV-cache management,
- [ ] token-aware cross-node routing,
- [ ] aggressive prefix-cache tuning,
- [ ] multimodal cache optimization,
- [ ] Kubernetes/autoscaling,
- [ ] canary deployment,
- [ ] full Prometheus/OpenTelemetry integration.

---

# Review

Complete this section after Stage 2 is finished.

## Files Added/Changed

_To be completed by Claude Code._

## Existing Architecture Findings

_To be completed by Claude Code._

## Installed vLLM Compatibility

```text
vLLM version:
async API:
Qwen-VL support:
multimodal input format:
health API:
shutdown API:
data-parallel API:
```

_To be completed by Claude Code._

## Selected vLLM Integration Mode

```text
A — in-process async vLLM + internal DP
B — application -> vllm serve + internal DP
```

Externally routed vLLM replicas are intentionally out of scope.

Selection:

Reason:

_To be completed by Claude Code._

## Final Backend Topology

### Transformers

_To be completed by Claude Code._

### vLLM

_To be completed by Claude Code._

## Single-GPU vLLM Verification

_To be completed by Claude Code._

## Multi-GPU vLLM DP Verification

```text
visible GPUs:
data_parallel_size:
tensor_parallel_size:
replicas observed:
GPU memory per GPU:
evidence of request distribution:
```

_To be completed by Claude Code._

## Concurrency Verification

_To be completed by Claude Code._

## Failure / Health Behavior

_To be completed by Claude Code._

## Transformers Regression

_To be completed by Claude Code._

## Backend Parity

_To be completed by Claude Code._

## Benchmark

| Metric | Transformers | vLLM DP=1 | vLLM DP=N |
|---|---:|---:|---:|
| Startup time | | | |
| GPU memory / replica | | | |
| Single-request latency | | | |
| Throughput @ concurrency 2 | | | |
| Throughput @ concurrency 4 | | | |
| Throughput @ concurrency 8 | | | |

_To be completed by Claude Code._

## Tests Executed

_To be completed by Claude Code._

## Known Limitations

_To be completed by Claude Code._

## Security / Privacy Review

Confirm:

- raw image bytes are not logged,
- prompts/responses are not logged by default,
- existing input/token limits remain enforced,
- backend errors are sanitized,
- no arbitrary remote media fetching was introduced,
- no secrets were added.

_To be completed by Claude Code._

## Recommended Next Stage

Possible future work:

```text
streaming
TTFT/inter-token latency
prefix caching
multimodal cache tuning
tensor parallelism for larger models
quantization
production observability
Kubernetes deployment
```

Do not start until this stage is reviewed.
