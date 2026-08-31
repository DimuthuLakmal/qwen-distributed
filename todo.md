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

Added:

- `src/inference/engines/vllm.py` -- `VLLMEngine` (Tasks 7-11).
- `src/inference/engines/factory.py` -- `build_engine()` backend dispatch (Task 4).
- `src/inference/schema.py` -- gained a "Public HTTP contract" section (`GenerateRequestSchema`/`GenerateResponseSchema`/`GenerationOptionsSchema`), extracted from the router's inference route so both the router and the new vLLM app use one definition, alongside the module's other inference schemas.
- `src/inference/http_utils.py` -- `read_bounded_upload()`, extracted alongside the schemas above but kept in its own module so `schema.py` stays schema definitions only.
- `src/inference/vllm/{app,dependencies}.py`, `src/inference/vllm/routes/{inference,health}.py` -- the single-process app used by `engine.type: vllm` (Task 5).
- `requirements-vllm.txt` -- pins `vllm==0.28.0`, separate from `requirements-inference.txt` so a Transformers-only deployment never installs it.
- `test/inference/engines/test_vllm_engine.py`, `test_vllm_engine_gpu.py`, `test_factory.py` -- new engine/factory tests.
- `test/inference/vllm/test_vllm_app.py` -- topology-level tests for the new vLLM app (fakes only).
- `test/inference/worker_app/test_worker_app.py` -- previously-missing coverage for the existing worker-internal app (`create_worker_app`), added so the Transformers regression baseline (Task 18) is real.

Changed:

- `src/inference/engines/base.py`, `service.py`, `engines/fake.py`, `engines/transformers.py`, `api/routes/internal_{generate,health,status}.py` -- converted `InferenceEngine.generate/health` and `InferenceService.generate/is_ready` from sync to `async def` (prerequisite for Task 12; see "Existing Architecture Findings" below). Transformers' actual blocking work is unchanged, just moved one layer down via `asyncio.to_thread` inside the engine instead of at the route boundary.
- `src/config/schema.py`, `src/config/configuration.py` -- added `EngineConfig`/`VLLMConfig` and `get_engine_config()` (Task 3).
- `src/utils/dist_utils.py` -- added `visible_gpu_count()` for `data_parallel_size: "auto"` resolution.
- `configs/inference.yaml` -- added the `engine:`/`engine.vllm:` block.
- `src/inference/transformers/router/routes/inference.py` -- now imports its request/response schemas from `schema.py` and its upload helper from `http_utils.py` instead of defining them locally (behavior unchanged).
- `src/pipeline/serve.py` -- now branches on `engine.type`: `transformers` keeps the existing router+worker wiring verbatim (moved into `_serve_transformers_router`); `vllm` builds and serves the new vLLM app (`_serve_vllm`), routing engine construction through `build_engine()` so `vllm` is never imported for a Transformers deployment.
- `docs/serving.md` -- added backend-selection, vLLM launch, and known-limitations sections.
- Existing tests (`test_service.py`, `test_fake_engine.py`, `test_transformers_engine.py`, `test_transformers_engine_gpu.py`, `test_config.py`) updated for the async engine contract and the new `get_engine_config()` cases.

Removed:

- `test/api/test_generate.py`, `test/api/test_health.py` -- imported `src.api.app`, deleted in an earlier commit (`85eab1d`) that replaced the single-process app with the router/worker split; these were dead/broken at collection time. Their case matrix now lives in `test/inference/vllm/test_vllm_app.py` against the new vLLM app, which serves the same role the old single-process app did.

## Existing Architecture Findings

- **The Transformers router/worker topology (Task 1) is real and HTTP-based**: a CPU-only public router process (`src/inference/transformers/router/`) forwards `/v1/generate` to per-GPU worker processes (`src/pipeline/worker.py` + `src/inference/transformers/worker_app/`) over `httpx`, selecting a worker via `ReplicaRouter` (`round_robin`/`least_loaded`) and re-encoding images as base64 JSON for that hop. At the start of this session, `src/pipeline/serve.py` imported this router package but the package itself didn't exist yet in the working tree (an earlier commit, `85eab1d`, had deleted the previous single-process app and rewritten `serve.py` to point at router/worker modules that were never added) -- the user pulled the missing files from the remote mid-session, after which the topology was confirmed complete and working.
- **Shared, backend-agnostic layer (Task 1/6), reused unchanged for vLLM**: `src/inference/schema.py` domain dataclasses (`InferenceRequest`/`InferenceResponse`/`GenerationConfig`/`ImageInput`), `src/inference/preprocessing.py` (`validate_and_build_request`, `build_conversation_messages`), `src/inference/postprocessing.py`, `src/inference/exception.py`'s core exception taxonomy, and `src/inference/service.py` (`InferenceService`, which only ever depended on the `InferenceEngine` ABC, never a concrete class).
- **Transformers-specific, kept as-is**: `src/inference/engines/transformers.py`, `InferenceModelConfig`, the router/worker HTTP split itself (`src/inference/transformers/router/`, `src/inference/transformers/worker_app/`, `src/pipeline/worker.py`, `scripts/start_workers.sh`) -- none of it was touched, only made reachable behind the new `engine.type` branch in `serve.py`.
- **Sync/async mismatch (not in todo.md's task list, but a required prerequisite)**: `InferenceEngine.generate()`/`health()` and `InferenceService.generate()`/`is_ready()` were plain sync methods, offloaded to a thread at the FastAPI route boundary via `asyncio.to_thread`. vLLM's `AsyncLLM.generate()` is fundamentally an async generator; forcing it through a sync ABC would mean bridging into an event loop from a worker thread per request, fighting vLLM's own scheduler and reintroducing exactly the kind of serialization Task 12 forbids. Converted the ABC and its immediate callers to `async def` (Transformers moved its thread-offload one layer down, into the engine itself, via a `_generate_sync` helper) -- mechanical, behavior-preserving for Transformers, and makes Task 12 hold by construction for vLLM (no lock/thread-pool around `engine.generate()` at the service layer any more).
- **No `engine.type`/backend discriminator existed anywhere** before this stage -- `configs/inference.yaml` had no `engine:` section, and `InferenceModelConfig` was Transformers-specific by construction. Added from scratch (Task 3).

## Installed vLLM Compatibility

```text
vLLM version:        0.28.0 (installed in a separate `vllm` conda env, not the repo's normal env)
async API:            vllm.v1.engine.async_llm.AsyncLLM -- AsyncLLM.from_engine_args(AsyncEngineArgs) is sync;
                       generate(prompt, sampling_params, request_id) is an async generator yielding RequestOutput;
                       abort(request_id) and check_health() are coroutines; shutdown(timeout=None) is sync.
Qwen-VL support:       Confirmed -- vllm.model_executor.models.registry lists Qwen2VLForConditionalGeneration,
                       Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration.
                       This repo's fine-tuned model is Qwen/Qwen3-VL-8B-Instruct (configs/model.yaml) -> Qwen3VLForConditionalGeneration, directly supported.
multimodal input format: vllm.inputs.llm.TextPrompt is a TypedDict (plain dict at runtime): {"prompt": str, "multi_modal_data": {"image": PIL.Image | list[PIL.Image]}}.
                       PIL images pass directly; qwen_vl_utils.process_vision_info was not needed (and is not installed in the vllm env) --
                       image-token placeholder handling is vLLM's own job once multi_modal_data is attached.
health API:            AsyncLLM.check_health() -- coroutine, raises (via self.dead_error) if the engine has died; no separate boolean-returning method.
shutdown API:          AsyncLLM.shutdown(timeout=None) -- sync, safe to call directly from InferenceEngine.close().
data-parallel API:     AsyncEngineArgs(data_parallel_size=N, data_parallel_backend="mp", tensor_parallel_size=1, ...) -- data_parallel_size > 1
                       is handled by AsyncLLM spawning its own engine-core subprocesses internally via multiprocessing, all from one calling
                       Python process. This is exactly Mode A ("in-process async vLLM + internal DP") -- confirmed viable without an external
                       `vllm serve` launcher.
```

Verified via live introspection in the `vllm` conda env (not from documentation) during planning, and exercised in `test/inference/engines/test_vllm_engine.py` against a mocked `AsyncLLM` matching this exact surface.

## Selected vLLM Integration Mode

```text
A — in-process async vLLM + internal DP
B — application -> vllm serve + internal DP
```

Externally routed vLLM replicas are intentionally out of scope.

Selection: **A**

Reason: The installed vLLM 0.28.0's `AsyncEngineArgs`/`AsyncLLM` support `data_parallel_size > 1` directly via an in-process multiprocessing backend (`data_parallel_backend="mp"`) -- `AsyncLLM.from_engine_args(...)` spawns and coordinates the DP engine-core subprocesses itself from a single calling Python process, with no need for an external `vllm serve` launcher or an application-managed replica router in front of it. Since Mode A is directly supported and matches the "one logical vLLM backend, vLLM-owned internal DP" architecture required by `claude.md`, and the selection preference is `A > B`, Mode B (an application-managed `vllm serve` subprocess wrapper) was not needed.

## Final Backend Topology

### Transformers

Unchanged: `Public API -> Replica Router (src/inference/transformers/router/) -> per-GPU Worker process (src/pipeline/worker.py, src/inference/transformers/worker_app/) -> TransformersEngine -> GPU`. Multi-process, HTTP-based, application-owned replica selection (`round_robin`/`least_loaded`). `src/pipeline/serve.py`'s `_serve_transformers_router()` starts only the router; workers are started separately via `scripts/start_workers.sh`, exactly as before this stage.

### vLLM

New: `Public API -> InferenceService(VLLMEngine) -> vLLM's internal AsyncLLM DP coordinator -> GPU`. Single process, no HTTP hop, no application-managed replica selection. `src/pipeline/serve.py`'s `_serve_vllm()` builds one `VLLMEngine` (via `src/inference/engines/factory.py:build_engine()`) and serves it directly through `src/inference/vllm/app.py:create_vllm_app()` -- the same public routes/schemas as the Transformers router, but wired straight to the engine in-process instead of forwarding over HTTP. vLLM itself creates and schedules across `data_parallel_size` replicas; no per-GPU application process or replica router was added for this backend.

## Single-GPU vLLM Verification

**Not executed against real weights** -- this dev machine has no exported model artifact at `results/final` yet (the export step has not been run here), so `test/inference/engines/test_vllm_engine_gpu.py` self-skips (confirmed: `1 skipped`), matching the existing `test_transformers_engine_gpu.py`'s skip-if-missing-artifact pattern. What *was* verified on this machine:

- `VLLMEngine` construction, prompt/multimodal rendering, sampling-param mapping, response assembly, health-check, cancellation/abort, and shutdown -- all against a mocked `AsyncLLM` matching the real confirmed API surface (`test/inference/engines/test_vllm_engine.py`, 10/10 passing).
- `src/pipeline/serve.py`'s `engine.type: vllm` branch end-to-end (config load -> `build_engine` -> `create_vllm_app` -> `uvicorn.run`) with `VLLMEngine` mocked out, confirming the process wiring is correct without needing real weights or a GPU.
- The public HTTP contract (multipart `/v1/generate`, `/health/live`, `/health/ready`, error mapping, request-ID propagation) against the vLLM app with `FakeInferenceEngine` (`test/inference/vllm/test_vllm_app.py`, 13/13 passing).

Once an exported artifact exists, run `test/inference/engines/test_vllm_engine_gpu.py` on a compute node in the `vllm` conda env (sourcing `.env` for `CUDA_HOME`/`CUDA_PATH`/`LD_LIBRARY_PATH`/`FLASHINFER_EXTRA_LDFLAGS`) to complete this verification with `data_parallel_size=1, tensor_parallel_size=1` and record GPU memory/baseline latency.

## Multi-GPU vLLM DP Verification

```text
visible GPUs:                     1 (this dev machine -- confirmed via `nvidia-smi -L`)
data_parallel_size:                not exercised on real hardware (see below)
tensor_parallel_size:              1
replicas observed:                 not exercised on real hardware
GPU memory per GPU:                not measured
evidence of request distribution:  not observed on real hardware
```

**Not executed** -- this dev machine has exactly one visible GPU, so `data_parallel_size >= 2` cannot be verified here regardless of whether an exported artifact is available. `AsyncEngineArgs(data_parallel_size=N, data_parallel_backend="mp")`'s field-level wiring is implemented and covered by the mocked `test_vllm_engine.py::test_engine_args_built_from_config` (confirms `data_parallel_size`/`tensor_parallel_size`/`data_parallel_backend` are passed through exactly as configured), but real multi-replica behavior, per-GPU memory, and request-distribution evidence require running on a multi-GPU node -- this is a known limitation, not a skipped step (see "Known Limitations").

## Concurrency Verification

`InferenceService`'s `asyncio.Lock` guards only the `_active_requests` counter, never the `await self.engine.generate(request)` call itself -- verified by `test/inference/test_service.py::test_generate_allows_concurrent_requests_to_overlap_in_the_engine`, which runs two `InferenceService.generate()` calls concurrently via `asyncio.gather` against a fake engine that sleeps mid-`generate()` and records the maximum observed in-flight overlap; the test asserts an observed overlap of 2 (i.e. both requests were inside `engine.generate()` simultaneously, not serialized). This holds for any engine (Transformers or vLLM) since the lock lives in the shared service layer, not per-backend. For vLLM specifically, `VLLMEngine.generate()` has no internal locking either -- concurrent calls each drive their own `async for` loop over `AsyncLLM.generate(...)`, letting vLLM's own scheduler interleave/batch them.

## Failure / Health Behavior

- `VLLMEngine.health()` returns `False` immediately if the engine never finished loading (`self._ready`), otherwise calls `await self._engine.check_health()` and returns `False` if it raises (verified against both a healthy and a raising mocked `AsyncLLM` in `test_vllm_engine.py`). `InferenceService.generate()` (shared, backend-agnostic) already fails fast with `EngineNotReadyError` (HTTP 503) whenever `is_ready()` is false, so a dead vLLM engine surfaces as 503 through the same path Transformers uses today -- no vLLM-specific readiness plumbing was needed at the API layer.
- Startup failures (bad model path, engine construction error) are wrapped in a clear `RuntimeError` naming the failing artifact/component (mirrors `TransformersQwenVLEngine._load`'s existing pattern) rather than propagating a raw vLLM exception.
- Backend errors are sanitized at one shared point: `VLLMEngine.generate()` does not locally catch/rewrap generation-time exceptions (matching `TransformersQwenVLEngine`); any exception propagates to `InferenceService.generate()`, which logs only `type(exc).__name__` (never the raw message) and raises `InferenceExecutionError("Inference engine failed to generate a response.")` -- the same sanitized 500 response shape as the Transformers path.
- Cancellation: `VLLMEngine.generate()` catches `asyncio.CancelledError` mid-`async for`, calls `await self._engine.abort(vllm_request_id)` (the vLLM-internal request id, not the service-level `request_id`), then re-raises -- verified in `test_vllm_engine.py::test_generate_aborts_engine_request_on_cancellation`.
- **Not verified on real hardware**: actual DP-rank failure/recovery behavior (todo.md Task 19's "determine behavior when one DP rank fails... document whether replica recovery is automatic or requires backend restart") requires a real multi-GPU `AsyncLLM` instance and is out of reach on this single-GPU dev machine. Per `claude.md`/todo.md's explicit instruction, no custom DP-replica recovery was built -- if a rank dies, `check_health()`/`errored` surfacing it as engine-not-ready (-> 503, -> process restart) is the extent of the built-in handling, matching "do not build custom DP-replica recovery unless necessary."

## Transformers Regression

All existing Transformers-path tests pass after the async ABC conversion (behavior-preserving; see "Existing Architecture Findings"): `test_transformers_engine.py` (mocked model/processor), `test_transformers_engine_gpu.py` (skips cleanly, no artifact on this machine), `test_service.py`, `test_fake_engine.py`, `test_config.py`. Additionally added `test/inference/worker_app/test_worker_app.py` (9 tests) -- there was previously **zero** automated coverage for `create_worker_app`/the worker-internal API despite it being the actual per-GPU serving surface, so this stage closes that gap rather than merely asserting "unchanged." `src/pipeline/serve.py`'s `_serve_transformers_router()` is a verbatim extraction of the pre-existing `main()` body (confirmed via a mocked dry-run: router app builds, `uvicorn.run` is called, `vllm` is never imported) -- no behavioral change to the router/worker HTTP topology itself. Real multi-GPU Transformers behavior (worker startup, replica routing across >1 GPU) was not re-verified on hardware in this session since it is unaffected by this stage's changes and this machine only has one GPU.

## Backend Parity

**Not executed.** A golden-set parity comparison (Task 17) needs a real exported model artifact and both conda environments (Transformers-capable + the separate `vllm` env) available at once, which the current environment split makes impractical as a single automated test on this machine. Recommend implementing it as a standalone manual script (`scripts/verify_backend_parity.py`, not yet written) run on a compute node with both environments installed and the exported artifact present, comparing: identical public schema, request ID preserved, model version present, non-empty generated text, sane token counts, identical validation behavior for both backends against the same prompts/images. `num_beams > 1` is a known, documented parity exception (vLLM backend rejects it outright; see "Known Limitations").

## Benchmark

| Metric | Transformers | vLLM DP=1 | vLLM DP=N |
|---|---:|---:|---:|
| Startup time | | | |
| GPU memory / replica | | | |
| Single-request latency | | | |
| Throughput @ concurrency 2 | | | |
| Throughput @ concurrency 4 | | | |
| Throughput @ concurrency 8 | | | |

**Not executed.** Benchmarking requires the exported model artifact (absent on this dev machine) and real GPU time; per `todo.md`'s own instruction ("do not turn this into a tuning project"), no benchmark numbers are fabricated here. Run once `results/final` exists, following Task 20's procedure (same model artifact/GPU/prompts/images/generation settings/warmup for both backends; concurrency 1/2/4/8; DP=1 vs DP=2 for vLLM where hardware permits).

## Tests Executed

`conda run -n vllm python -m pytest test/inference/ test/api/ -v` (in the `vllm` conda env, with `.env`'s `CUDA_HOME`/`CUDA_PATH`/`LD_LIBRARY_PATH`/`FLASHINFER_EXTRA_LDFLAGS` sourced): **98 collected, 96 passed, 2 skipped** (the two real-GPU tests, both skipping cleanly due to no exported artifact). This covers config validation (`test_config.py`, including all new `get_engine_config()` cases), the backend factory (`test_factory.py`, including the `sys.modules` isolation regression), the mocked `VLLMEngine` (`test_vllm_engine.py`), the vLLM app (`test_vllm_app.py`), the worker-internal app (`test_worker_app.py`, previously uncovered), and the full pre-existing `InferenceService`/`TransformersQwenVLEngine`/preprocessing/postprocessing/schema suites (updated for the async engine contract). Training-pipeline tests (`test_data_pipeline.py`, `test_fsdp.py`, `test_qwen_vl.py`) are out of scope for this stage and were not run.

`src/pipeline/serve.py` was also exercised directly (not via pytest) with `uvicorn.run` mocked: the default (`engine.type: transformers`) config runs `main()` end-to-end and confirms `'vllm' not in sys.modules` afterward; a temporary `engine.type: vllm` config runs `main()` end-to-end with `VLLMEngine` mocked, confirming the vLLM-app wiring is correct.

Not run (require the `vllm` conda env's exported artifact and/or real multi-GPU hardware, neither available on this dev machine): `test_vllm_engine_gpu.py`'s real generation, any real multi-GPU DP run, backend parity, benchmarking.

## Known Limitations

- No exported model artifact existed on this dev machine at implementation time -- all real-weight verification (Tasks 13, 15, 17, 19, 20) is blocked on running the export step first.
- This dev machine has exactly one visible GPU -- multi-GPU DP (Task 14) can never be verified here; needs a multi-GPU node.
- `num_beams > 1` is not supported by the vLLM backend (no beam-search equivalent in the installed `SamplingParams`) -- `VLLMEngine.generate()` raises `InvalidGenerationOptionsError` (HTTP 400) rather than silently changing behavior.
- Backend parity (Task 17) has no automated test given the two-conda-env split on this machine (Transformers env vs. the separate `vllm` env); a manual comparison script is recommended instead (see "Backend Parity" above).
- `data_parallel_size: "auto"` resolves via `torch.cuda.device_count()` at the moment the engine is constructed (inside `build_engine`); it has only been tested with a mocked GPU count, not real multi-GPU hardware.
- vLLM's DP-rank failure/recovery behavior (Task 19) is unverified on real hardware; only the health-check/error-sanitization paths were exercised with mocks.
- The `vllm` package requires its own conda environment on this machine (separate from the repo's normal training/eval env) with `CUDA_HOME`/`CUDA_PATH`/`LD_LIBRARY_PATH`/`FLASHINFER_EXTRA_LDFLAGS` set (see `.env`) for real (non-mocked) initialization -- this is an environment-setup constraint, not a code limitation, but worth calling out for anyone reproducing this work on a different machine.

## Security / Privacy Review

Confirm:

- raw image bytes are not logged,
- prompts/responses are not logged by default,
- existing input/token limits remain enforced,
- backend errors are sanitized,
- no arbitrary remote media fetching was introduced,
- no secrets were added.

Confirmed:

- **Raw image bytes are not logged**: `VLLMEngine`/`InferenceService` only log counts, latency, and identifiers -- never image content -- for both backends.
- **Prompts/responses are not logged by default**: unchanged, still covered by `test_service.py::test_generate_success_logs_never_include_raw_prompt_or_response_text` and the failure-path equivalent, both of which exercise the shared `InferenceService` layer used by every backend.
- **Existing input/token limits remain enforced**: `validate_and_build_request`/`_build_generation_config` (unchanged) run before any engine sees a request, for both the router->worker path and the new direct-to-`VLLMEngine` path; the `engine.vllm.limit_mm_per_prompt.image` config value is cross-validated against `limits.max_images` at load time so vLLM can never be configured to silently reject requests the public API already accepts.
- **Backend errors are sanitized**: `VLLMEngine` raises no internal vLLM types past its own boundary; `InferenceService` logs only `type(exc).__name__` and returns a generic `InferenceExecutionError` message, identically for both backends, through the same shared error-handler mapping (`register_error_handlers`).
- **No arbitrary remote media fetching was introduced**: `VLLMEngine` only ever receives already-validated `PIL.Image` objects built from uploaded bytes (`ImageInput`/`build_conversation_messages`) -- no URL/remote-fetch path was added.
- **No secrets were added**: `requirements-vllm.txt` and the new `engine:`/`engine.vllm:` config block contain no credentials. Incidentally noticed `.env` (local `CUDA_HOME`/`CUDA_PATH`/`LD_LIBRARY_PATH`/`FLASHINFER_EXTRA_LDFLAGS` paths, no secrets, pre-existing/user-provided) was untracked but not gitignored -- added `.env` to `.gitignore` as a precaution against a future accidental commit of environment-specific or credential data through that file.

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
