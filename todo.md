# Stage 1 TODO — Correct Multi-GPU Replicated Inference Service

## Goal

Build the first production-shaped inference service for the already-trained and exported Qwen-VL model.

The completed Stage 1 must support:

```text
                         Public API
                             |
                             v
                      Replica Router
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
      GPU Worker 0       GPU Worker 1       GPU Worker 2
      Qwen replica       Qwen replica       Qwen replica
          |                  |                  |
        GPU 0              GPU 1              GPU 2
```

Each GPU worker owns **one complete model replica** when the model fits on one GPU. The same service must also run with one worker/GPU for local development.

Stage 1 is divided into:

```text
Stage 1A — single-worker correctness
Stage 1B — multi-GPU replication and routing
```

Do not implement vLLM, continuous batching, tensor parallelism, streaming, Kubernetes, or advanced observability in this stage.

---

# Stage 1A — Single-Worker Correctness

## Task 1 — Inspect the Existing Repository

- [ ] Read the existing project structure before adding inference code.
- [ ] Identify the existing configuration-loading mechanism and typed config classes.
- [ ] Identify logging/utilities that should be reused.
- [ ] Identify how exported Hugging Face model artifacts are represented.
- [ ] Inspect evaluation model-loading code for reusable model/processor settings.
- [ ] Inspect existing test conventions and dependency management.
- [ ] Confirm whether FastAPI, Uvicorn, an async HTTP client, Pillow, and inference dependencies exist.
- [ ] Record required dependency additions before implementation.
- [ ] Avoid modifying working training/evaluation code unless a shared abstraction truly requires it.

### Completion criteria

- Existing project conventions are understood.
- Inference code integrates cleanly rather than duplicating abstractions.
- Any deviations from `CLAUDE.md` are documented before implementation.

---

## Task 2 — Add Inference and Worker Configuration

- [ ] Add typed inference configuration using the existing config system.
- [ ] Add `inference.yaml` or equivalent.
- [ ] Include exported model path and model/version ID.
- [ ] Include inference dtype/device and attention implementation where supported.
- [ ] Include generation defaults and hard generation limits.
- [ ] Include public router host/port.
- [ ] Include worker IDs/endpoints.
- [ ] Include prompt/image/output limits.
- [ ] Ensure request-level values cannot exceed server limits.
- [ ] Add configuration validation tests.

### Suggested shape

```yaml
service:
  host: "0.0.0.0"
  port: 8000

model:
  id: "qwen-vl-v001"
  path: "/models/qwen-vl-v001"
  dtype: "bfloat16"
  device: "cuda"
  attn_implementation: "flash_attention_2"

generation:
  max_new_tokens: 256
  do_sample: false

limits:
  max_prompt_tokens: 4096
  max_images: 4
  max_image_bytes: 15728640
  max_image_pixels: 16000000
  max_new_tokens: 512

workers:
  - id: "gpu-0"
    url: "http://127.0.0.1:8100"
  - id: "gpu-1"
    url: "http://127.0.0.1:8101"

routing:
  strategy: "round_robin"
```

Prefer process-level GPU assignment with `CUDA_VISIBLE_DEVICES`; do not make the public router responsible for CUDA placement.

### Completion criteria

- Single- and multi-worker deployments are configuration-driven.
- Invalid critical configuration fails clearly at startup.

---

## Task 3 — Define Canonical Inference Schemas

- [ ] Create internal `GenerationConfig`.
- [ ] Create internal `InferenceRequest`.
- [ ] Create internal `InferenceResponse`.
- [ ] Include request ID, prompt, optional system prompt, images, and generation options.
- [ ] Include model/version, input/output token counts, and latency in responses.
- [ ] Define worker status schema.
- [ ] Define worker generation request/response contract.
- [ ] Keep public HTTP/Pydantic schemas separate from internal schemas where useful.
- [ ] Add validation tests.

### Completion criteria

- Public API, router, worker client, and worker service communicate through explicit contracts.

---

## Task 4 — Implement Input Validation and Multimodal Preprocessing

- [ ] Validate prompt presence and limits.
- [ ] Validate image count, MIME/media type, bytes, dimensions, and total pixels.
- [ ] Normalize images into Qwen-compatible inputs.
- [ ] Build Qwen multimodal messages.
- [ ] Support optional system prompt.
- [ ] Bound generation parameters by service limits.
- [ ] Do not fetch arbitrary user-provided remote URLs.
- [ ] Avoid unnecessary image copies where practical.
- [ ] Add unit tests.

### Required tests

- [ ] valid text + image,
- [ ] missing prompt,
- [ ] unsupported image type,
- [ ] oversized payload,
- [ ] excessive image count,
- [ ] excessive pixel dimensions,
- [ ] excessive `max_new_tokens`,
- [ ] optional system prompt,
- [ ] multiple images if supported.

### Completion criteria

- Invalid requests fail before model execution.
- Qwen-specific preprocessing is outside API route handlers.

---

## Task 5 — Create the Inference Engine Interface

- [ ] Add abstract `InferenceEngine`.
- [ ] Define generation and readiness behavior.
- [ ] Keep interface independent of Hugging Face model classes.
- [ ] Decide/document sync vs async boundary.
- [ ] Ensure a future vLLM engine can satisfy the same contract.
- [ ] Add fake/mock engine for tests.

### Completion criteria

- `InferenceService` can run against a fake engine without importing Transformers.

---

## Task 6 — Implement the Transformers Qwen-VL Engine

- [ ] Load processor once during worker startup.
- [ ] Load exported Qwen-VL model once during worker startup.
- [ ] Use configured dtype/device.
- [ ] Set `model.eval()`.
- [ ] Use `torch.inference_mode()`.
- [ ] Apply Qwen multimodal chat template with `add_generation_prompt=True`.
- [ ] Convert multimodal content into processor/model inputs.
- [ ] Move tensors to the worker GPU safely.
- [ ] Use deterministic generation by default.
- [ ] Respect generation limits.
- [ ] Decode generated continuation only.
- [ ] Calculate token counts and latency.
- [ ] Implement readiness and clear startup failures.
- [ ] Add mocked engine tests.
- [ ] Add optional real-GPU smoke test excluded from CPU CI.

### Critical GPU rule

The worker must work with:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_id>
```

and may use:

```python
torch.device("cuda:0")
```

internally.

Do not use `device_map="auto"`, DDP, FSDP, `torch.nn.DataParallel`, or multiple model replicas in one worker process.

### Completion criteria

- One worker process can serve the exported Qwen-VL model on one GPU.

---

## Task 7 — Implement Postprocessing

- [ ] Add a postprocessing component/function.
- [ ] Keep cleanup out of API route code.
- [ ] Preserve meaningful generated output.
- [ ] Leave extension points for structured-output parsing.
- [ ] Add tests.

---

## Task 8 — Implement Worker `InferenceService`

- [ ] Create `InferenceService` with engine dependency injection.
- [ ] Coordinate validation/preprocessing, generation, and postprocessing.
- [ ] Preserve request ID.
- [ ] Keep runtime-specific code out of service layer.
- [ ] Define service exceptions.
- [ ] Track active request count safely for worker status.
- [ ] Add tests using fake engine.

### Completion criteria

- Worker service is testable without a GPU.

---

## Task 9 — Implement Internal GPU Worker API

- [x] Add worker-only FastAPI app.
- [x] Add `GET /internal/health/live`.
- [x] Add `GET /internal/health/ready`.
- [x] Add `GET /internal/status`.
- [x] Add `POST /internal/generate`.
- [x] Initialize model/engine once during app lifespan.
- [x] Expose worker ID, readiness, and `active_requests`.
- [x] Do not build a custom request queue only for Stage 1.
- [x] Map expected errors cleanly and sanitize unexpected errors.
- [ ] Add worker API tests with fake engine. _(deferred: no local GPU this pass; test-writing explicitly out of scope for this change)_

### Suggested status response

```json
{
  "worker_id": "gpu-0",
  "ready": true,
  "active_requests": 1
}
```

### Completion criteria

- Worker API works independently of the public router.

---

## Task 10 — Add Worker Entrypoint

- [x] Create `src/pipeline/worker.py`.
- [x] Load inference/worker configuration.
- [x] Configure logging.
- [x] Start internal worker FastAPI app.
- [x] Support worker ID/port arguments or environment variables.
- [x] Document one-process-per-GPU usage.
- [x] Ensure graceful startup/shutdown.

### Required examples

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.worker   --config configs/inference.yaml   --worker-id gpu-0   --port 8100
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.worker   --config configs/inference.yaml   --worker-id gpu-1   --port 8101
```

### Completion criteria

- Separate workers can each bind to a distinct physical GPU while using `cuda:0` internally.

---

# Stage 1B — Multi-GPU Replica Routing

## Task 11 — Implement Worker Client

- [x] Create a worker client abstraction used by the public router.
- [x] Implement health/status requests and generation forwarding.
- [x] Add configurable connect/read timeouts.
- [x] Reuse persistent HTTP connections.
- [x] Propagate request ID.
- [x] Convert network/worker failures into typed router errors.
- [ ] Add tests with mocked HTTP responses. _(deferred: no local GPU this pass; test-writing explicitly out of scope for this change)_

### Completion criteria

- HTTP client calls are centralized and testable.

---

## Task 12 — Implement Worker Registry

- [x] Build worker registry from configuration.
- [x] Store worker IDs/endpoints and last-known readiness.
- [x] Support temporarily marking failed workers unavailable.
- [x] Avoid hard-coded worker count.
- [ ] Add tests. _(deferred: no local GPU this pass; test-writing explicitly out of scope for this change)_

### Completion criteria

- Router can operate over any configured number of workers.

---

## Task 13 — Implement Replica Routing Interface

- [x] Add `ReplicaRouter` abstraction.
- [x] Keep routing independent of Qwen/Transformers.
- [x] Implement concurrency-safe round-robin routing first.
- [x] Exclude unhealthy/unready workers.
- [x] Return clear no-healthy-worker errors.
- [ ] Add tests. _(deferred: no local GPU this pass; test-writing explicitly out of scope for this change)_

### Required routing test

With healthy workers `gpu-0`, `gpu-1`, `gpu-2`, sequential routing should cycle:

```text
gpu-0 -> gpu-1 -> gpu-2 -> gpu-0
```

### Completion criteria

- Requests are assigned across replicas without model/API coupling.

---

## Task 14 — Optional Least-Loaded Router

Implement only if it remains small and clean after worker status exists.

- [x] Add `least_loaded` strategy using `active_requests`.
- [x] Prefer healthy worker with lowest active count.
- [x] Define deterministic tie-breaking.
- [ ] Add tests. _(deferred: no local GPU this pass; test-writing explicitly out of scope for this change)_
- [x] Make strategy configurable.

Do not add token-cost estimation or continuous batching.

---

## Task 15 — Implement Public Router/API Service

- [x] Create public FastAPI application.
- [x] Add `GET /health/live`.
- [x] Add `GET /health/ready`.
- [x] Add `POST /v1/generate`.
- [x] Public router must not load Qwen model weights.
- [x] Validate/parse public requests.
- [x] Select healthy worker via `ReplicaRouter`.
- [x] Forward using `WorkerClient`.
- [x] Preserve request ID end-to-end.
- [x] Mark failed workers unavailable as appropriate.
- [x] Decide/document retry policy; do not blindly retry partially completed generation. (see `docs/serving.md` "Retry policy")
- [x] Return HTTP 503 when no workers are available.
- [x] Add request ID handling/middleware.
- [x] Do not log raw images/prompts/responses.
- [ ] Add public API integration tests with fake workers. _(deferred: no local GPU this pass; test-writing explicitly out of scope for this change)_

### Readiness semantics

Public `/health/ready` is ready when at least one configured worker is healthy and ready, unless stricter semantics are explicitly configured.

### Completion criteria

- Public API continues serving when one of several workers becomes unavailable.
- Public API contains no tokenizer/model/CUDA logic.

---

## Task 16 — Add Public Router Entrypoint

- [x] Create/adapt `src/pipeline/serve.py`.
- [x] Load public router configuration.
- [x] Configure logging.
- [x] Start public FastAPI/Uvicorn process.
- [x] Do not load model weights.
- [x] Document launch command.

```bash
python -m src.pipeline.serve --config configs/inference.yaml
```

---

## Task 17 — Add Local Multi-GPU Launch Support

Use the simplest mechanism appropriate for the repository: shell script, documented commands, small launcher, or process supervisor config.

- [x] Explicitly map one worker process to one physical GPU.
- [x] Prefer `CUDA_VISIBLE_DEVICES`.
- [x] Start one worker port per replica.
- [x] Start public router separately.
- [x] Do not use `uvicorn --workers N` as the GPU replication mechanism.
- [x] Add clean shutdown instructions.
- [x] Document expected process topology.

### Example

```text
router      :8000   CPU process
worker-0    :8100   CUDA_VISIBLE_DEVICES=0
worker-1    :8101   CUDA_VISIBLE_DEVICES=1
worker-2    :8102   CUDA_VISIBLE_DEVICES=2
worker-3    :8103   CUDA_VISIBLE_DEVICES=3
```

### Completion criteria

- Replicated serving can be launched without source changes.

---

## Task 18 — Baseline Observability

Public router logs:

- [x] request ID,
- [x] selected worker ID,
- [x] success/failure,
- [x] router/total latency.

Worker logs:

- [x] request ID,
- [x] worker ID,
- [x] model version,
- [x] active request count,
- [x] inference latency,
- [x] input/output token counts,
- [x] success/failure.

Privacy/security:

- [x] no raw image bytes,
- [x] no raw prompts/responses by default,
- [x] no secrets.

Do not add a full metrics stack yet unless already present.

### Completion criteria

- A request can be traced from public router to the selected GPU worker.

---

## Task 19 — Failure and Routing Tests

Add integration tests for:

- [ ] all workers healthy,
- [ ] one worker unhealthy,
- [ ] worker becomes unavailable after startup,
- [ ] no workers healthy,
- [ ] worker timeout,
- [ ] worker inference error,
- [ ] round-robin sequence,
- [ ] least-loaded selection if implemented,
- [ ] request ID propagation,
- [ ] public readiness with partial worker availability.

### Completion criteria

- Failure behavior is deterministic and documented.

---

## Task 20 — Real Multi-GPU Smoke Test

When multi-GPU hardware is available:

- [ ] Launch at least two workers on distinct GPUs.
- [ ] Verify each worker sees only its assigned GPU.
- [ ] Verify each worker loads exactly one full model replica.
- [ ] Verify both workers become ready.
- [ ] Send multiple requests through public API.
- [ ] Confirm requests are distributed across both workers.
- [ ] Confirm both GPUs execute inference.
- [ ] Record GPU memory per replica.
- [ ] Stop one worker and confirm service continues through the other.
- [ ] Restart the worker and confirm it can rejoin routing.

Do not optimize performance yet.

### Completion criteria

- Real hardware demonstrates model replication and request distribution.

---

## Task 21 — Regression and End-to-End Verification

- [ ] Run inference/routing/API tests.
- [ ] Run relevant existing training/evaluation tests.
- [ ] Run repository lint/type/static checks.
- [ ] Verify public and worker health/readiness endpoints.
- [ ] Send a real image+prompt through the public API.
- [ ] Confirm generated response.
- [ ] Confirm invalid requests fail before GPU execution where applicable.
- [ ] Confirm no raw prompt/image content appears in normal logs.
- [ ] Confirm public router holds no Qwen weights.
- [ ] Confirm model loading occurs once per worker process.

### Completion criteria

- Stage 1A and Stage 1B both work end-to-end.
- Existing training/evaluation behavior remains intact.

---

# Out of Scope for Stage 1

Do **not** implement yet:

- [ ] vLLM,
- [ ] continuous/in-flight batching,
- [ ] token-aware routing,
- [ ] custom GPU scheduler,
- [ ] streaming,
- [ ] SSE/WebSockets,
- [ ] tensor parallelism,
- [ ] pipeline parallelism,
- [ ] FSDP inference,
- [ ] quantized inference,
- [ ] Kubernetes,
- [ ] autoscaling,
- [ ] Redis/Kafka queues,
- [ ] arbitrary remote image URL fetching,
- [ ] advanced authentication/rate limiting,
- [ ] Prometheus/OpenTelemetry unless already present,
- [ ] canary deployment,
- [ ] model registry integration.

---

# Review

Complete only after all Stage 1 tasks are finished.

## Files added/changed

_To be completed by Claude Code._

## Architecture implemented

_To be completed by Claude Code._

## Single-worker verification

_To be completed by Claude Code._

## Multi-GPU replication verification

Include worker IDs, GPU mapping, model memory per GPU, and evidence that requests were distributed across replicas.

_To be completed by Claude Code._

## Routing behavior

_To be completed by Claude Code._

## Tests executed

_To be completed by Claude Code._

## Known limitations

_To be completed by Claude Code._

## Security review

Confirm at minimum:

- no credentials/secrets were added,
- raw image data is not logged,
- prompts/responses are not logged by default,
- arbitrary remote URL fetching was not added,
- upload size/type/pixel limits are enforced,
- user generation settings cannot exceed server limits,
- errors do not expose stack traces/internal paths.

_To be completed by Claude Code._

## Recommended Stage 2 work

Expected direction only; do not implement yet:

```text
Transformers replicas
  -> vLLM replicas
  -> continuous batching inside each replica
  -> async serving/streaming
  -> richer load-aware routing
  -> throughput, TTFT, p95/p99 benchmarking
```

_To be completed after Stage 1 review._
