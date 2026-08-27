# CLAUDE.md

## Purpose

This repository already contains the model training and evaluation pipelines for a fine-tuned Qwen-VL model. The next objective is to build an industry-grade inference subsystem around the exported Hugging Face model artifact.

This file defines the implementation rules, target architecture, and staged delivery plan for Claude Code.

The inference implementation must remain clearly separated from training and evaluation. Do not move inference-specific concerns into the training code, and do not make the API layer depend directly on Hugging Face model classes.

---

## Working Rules

1. **Read before changing**
   - Inspect the existing repository structure, configuration system, model-loading utilities, logging utilities, and tests before writing code.
   - Reuse existing abstractions where they are appropriate.
   - Do not introduce duplicate configuration or utility systems without a clear reason.

2. **Implement only the active stage**
   - The active stage is defined by `todo.md`.
   - Do not start work from later stages until the current stage is complete and explicitly approved.
   - Later stages described in this file are architectural direction, not permission to implement them now.

3. **Keep changes small and reviewable**
   - Prefer incremental additions over broad refactors.
   - Do not modify working training or evaluation behavior unless an inference requirement genuinely needs a shared abstraction.
   - If a shared abstraction must change, keep backward compatibility where practical and add tests.

4. **Preserve separation of concerns**
   - API routes handle HTTP concerns only.
   - `InferenceService` coordinates application logic.
   - Engine implementations own model/runtime-specific execution.
   - Preprocessing owns request-to-model-input conversion.
   - Postprocessing owns model-output-to-domain-response conversion.
   - Configuration remains external and typed.
   - Deployment concerns do not belong in model or service code.

5. **Use the exported Hugging Face artifact**
   - Production-style inference must load the exported Hugging Face model/processor, not the PyTorch Distributed Checkpoint training format.
   - DCP remains a training-resume/checkpoint-selection format.

6. **Load the model once**
   - Never call `from_pretrained()` per request.
   - Load the processor and model during application/engine startup.
   - The API must reuse the initialized engine for all requests.

7. **Keep the inference runtime replaceable**
   - The API and service layers must not depend on `AutoModelForImageTextToText`.
   - Define an engine interface.
   - Stage 1 uses a Transformers-backed engine for correctness.
   - A later stage may add vLLM or another production runtime without changing API contracts.

8. **Do not prematurely optimize**
   - Stage 1 prioritizes correctness, interfaces, testability, and a working end-to-end path.
   - Do not implement continuous batching, custom CUDA scheduling, tensor parallelism, or complex GPU queues in Stage 1.
   - Measure before optimizing.

9. **Multimodal inputs must be bounded**
   - Treat images and videos as untrusted input.
   - Validate count, type, size, dimensions, and configured limits.
   - Do not introduce arbitrary remote URL fetching in Stage 1.
   - Prefer uploaded content or trusted internal references.
   - Avoid logging raw images, prompts, or generated responses by default.

10. **Model input logic belongs below the API**
    - Routes must not call `apply_chat_template()`, tokenize, build `pixel_values`, or call `model.generate()`.
    - Qwen-specific processing belongs in the inference preprocessing/engine layer.

11. **Use inference-safe PyTorch behavior**
    - Put the model in `eval()` mode.
    - Use `torch.inference_mode()` for generation.
    - Do not create gradients.
    - Do not use FSDP for the initial inference implementation if the exported model fits on one GPU.

12. **Configuration-driven behavior**
    - Model path, dtype, attention backend, device, generation defaults, request limits, and server settings must come from configuration.
    - Avoid hard-coded filesystem paths and GPU IDs.

13. **Type and validate public boundaries**
    - Use explicit request/response schemas.
    - Use typed internal request/response objects where useful.
    - Validate all user-controlled parameters before calling the model.

14. **Testing is part of implementation**
    - Every Stage 1 component should have unit or integration coverage appropriate to its responsibility.
    - API tests should use a mock/fake inference engine rather than requiring a GPU.
    - GPU-backed end-to-end tests should be separable from normal CI tests.

15. **Document behavior, not obvious syntax**
    - Add docstrings where interfaces or lifecycle behavior are non-obvious.
    - Avoid excessive comments that simply restate Python code.

16. **Do not hide errors**
    - Convert expected input errors into clear client-facing responses.
    - Log unexpected failures with request IDs and structured context.
    - Do not silently fall back to a different model/device/runtime.

17. **No secrets in source**
    - Never commit credentials, tokens, API keys, private URLs, or environment-specific secrets.
    - Configuration may reference environment variables where needed.

18. **Finish each task cleanly**
    - Run relevant tests and static checks after each logical change.
    - Mark completed items in `todo.md`.
    - At the end of the active stage, add a `Review` section to `todo.md` describing:
      - files added/changed,
      - important design decisions,
      - tests run,
      - known limitations,
      - recommendations for the next stage.

---

# Target Inference Architecture

The intended dependency direction is:

```text
Client
  |
  v
HTTP API
  |
  v
InferenceService
  |
  +--------------------+
  |                    |
  v                    v
Validation        Pre/Post-processing
  |
  v
InferenceEngine interface
  |
  +-----------------------------+
  |                             |
  v                             v
Transformers engine          Future engine
(Stage 1)                    (e.g. vLLM)
  |
  v
Exported Qwen-VL Hugging Face model
  |
  v
GPU
```

A request should conceptually move through:

```text
HTTP request
  -> request validation
  -> canonical inference request
  -> multimodal preprocessing
  -> inference engine
  -> generation
  -> decode/postprocess
  -> canonical inference response
  -> HTTP response
```

---

# Intended Repository Structure

Adapt this to the existing repository rather than forcing it mechanically.

```text
src/
├── inference/
│   ├── __init__.py
│   ├── config.py
│   ├── schema.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   ├── service.py
│   └── engines/
│       ├── __init__.py
│       ├── base.py
│       ├── transformers.py
│       └── vllm.py              # later stage
│
├── api/
│   ├── __init__.py
│   ├── app.py
│   ├── dependencies.py
│   ├── routes/
│   │   ├── health.py
│   │   └── inference.py
│   └── middleware/
│       ├── request_id.py
│       ├── error_handler.py
│       └── logging.py
│
└── pipeline/
    └── serve.py

configs/
└── inference.yaml

tests/
├── inference/
└── api/
```

If the existing project already has equivalent packages, integrate into them rather than creating redundant modules.

---

# Core Interfaces

## Canonical request

The inference layer should normalize external inputs into a model-independent request concept containing, at minimum:

- request ID,
- user prompt,
- zero or more images,
- optional system prompt,
- generation parameters.

Video support may be represented in the schema now if it is easy to do cleanly, but full video API support is not required in Stage 1 unless the existing project already needs it.

## Canonical response

The internal response should be able to carry:

- request ID,
- generated text,
- input token count,
- output token count,
- total latency,
- model/version identifier.

Do not expose internal tensor/model details through the API.

## Engine interface

The engine must expose a small stable interface such as:

```python
class InferenceEngine(ABC):
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        ...

    async def health(self) -> bool:
        ...
```

The exact implementation may differ if synchronous internals are more appropriate, but API/service code must remain independent of the concrete Transformers model.

---

# Model Serving Principles

## Stage 1 execution model

The first implementation should:

- load the exported Qwen-VL Hugging Face artifact,
- load its matching processor,
- place the model on the configured device once,
- set `model.eval()`,
- use `torch.inference_mode()`,
- use Qwen's multimodal chat template,
- use deterministic generation by default,
- return only generated continuation tokens rather than re-decoding the prompt,
- support a single image plus text reliably,
- support multiple images only if it follows naturally from the processor/API implementation.

Do not add FSDP to Stage 1 inference.

## Generation defaults

Use conservative deterministic defaults unless configuration overrides them:

- `do_sample = false`,
- bounded `max_new_tokens`,
- `temperature` only used when sampling is enabled,
- `num_beams = 1` unless explicitly required.

Never allow a request to bypass configured maximum generation limits.

---

# API Principles

Recommended initial endpoints:

```text
GET  /health/live
GET  /health/ready
POST /v1/generate
```

`/health/live` answers whether the server process is alive.

`/health/ready` answers whether the model/engine is initialized and able to receive traffic.

`/v1/generate` performs non-streaming multimodal generation.

Stage 1 does not need streaming.

The API must remain thin:

```text
request parsing -> validation -> InferenceService -> response mapping
```

No model-specific code in route handlers.

---

# Error Handling

Distinguish at least:

- invalid request,
- unsupported media type,
- payload too large,
- invalid generation options,
- model not ready,
- inference timeout if implemented,
- internal inference error.

Do not return raw stack traces to clients.

Unexpected server errors should retain a request ID in logs so failures can be traced.

---

# Observability Baseline

Stage 1 only needs a clean foundation, not a full production observability stack.

At minimum log structured or consistently formatted fields for:

- request ID,
- model version,
- success/failure,
- total latency,
- input token count,
- output token count.

Do not log raw images or prompts by default.

Prometheus/OpenTelemetry/Grafana integration belongs to a later stage.

---

# Implementation Stages

## Stage 1 — Correct End-to-End Inference Service

### Goal

Build a clean, testable, non-streaming inference service that loads the exported Qwen-VL Hugging Face model once and exposes a validated HTTP generation endpoint.

### Required outcome

At the end of Stage 1, the project should support:

```text
exported Qwen-VL model
  -> Transformers inference engine
  -> InferenceService
  -> FastAPI
  -> image + prompt request
  -> generated response
```

with health endpoints, configuration, tests, and no training/evaluation regression.

### Stage 1 boundaries

Do implement:

- canonical inference schemas,
- inference engine abstraction,
- Transformers Qwen-VL engine,
- multimodal preprocessing,
- postprocessing,
- inference service,
- FastAPI application,
- liveness/readiness endpoints,
- non-streaming generation endpoint,
- request validation and initial safety limits,
- configuration,
- service startup/shutdown lifecycle,
- unit/API tests,
- optional GPU smoke test,
- `pipeline/serve.py`.

Do not implement yet:

- vLLM,
- continuous/in-flight batching,
- custom request scheduler,
- streaming tokens,
- WebSocket/SSE,
- tensor parallelism,
- autoscaling,
- Kubernetes manifests,
- Prometheus/Grafana,
- Redis queues,
- model registry integration,
- canary routing.

The detailed Stage 1 task list lives in `todo.md`.

---

## Stage 2 — Production Inference Runtime and Concurrency

### Goal

Replace or complement the baseline Transformers engine with a production-oriented runtime capable of higher concurrency and efficient batching.

Likely work:

- implement a vLLM-backed engine if compatible with the target Qwen-VL model,
- continuous/in-flight batching,
- asynchronous request handling,
- streaming generation,
- time-to-first-token metrics,
- queue/backpressure behavior,
- engine-level cancellation,
- benchmark Transformers vs production runtime.

Do not begin Stage 2 until Stage 1 has been reviewed.

---

## Stage 3 — Service Hardening and Observability

### Goal

Make the inference service safe and observable under real application traffic.

Likely work:

- hardened media validation,
- decompression/image-bomb protection,
- strict pixel/token budgets,
- timeout policies,
- admission control,
- trusted-object-storage input option if required,
- authentication hooks,
- rate limiting,
- structured logging,
- Prometheus/OpenTelemetry metrics,
- latency and throughput dashboards,
- GPU utilization and memory metrics,
- failure classification.

---

## Stage 4 — Containerization and Deployment

### Goal

Package and deploy the service reproducibly on GPU infrastructure.

Likely work:

- production Docker image,
- non-root runtime where practical,
- GPU runtime configuration,
- Kubernetes deployment/service manifests,
- readiness/liveness probes,
- resource requests/limits,
- one-model-replica-per-GPU deployment strategy where appropriate,
- rolling deployment,
- graceful shutdown/draining,
- configuration/secrets integration.

---

## Stage 5 — Performance, Reliability, and Model Operations

### Goal

Turn the service into a measurable, versioned production inference platform.

Likely work:

- load testing,
- throughput/latency SLOs,
- batching and token-budget tuning,
- quantized serving where justified,
- model version metadata,
- model artifact validation,
- golden-set regression tests,
- canary model rollout,
- rollback strategy,
- benchmark tracking,
- cost/GPU-efficiency analysis.

---

# Stage Completion Definition

A stage is complete only when:

1. every task in the active `todo.md` is checked,
2. relevant tests pass,
3. the implementation follows the architecture above,
4. known limitations are explicitly documented,
5. no later-stage feature was accidentally mixed into the implementation,
6. `todo.md` contains a final `Review` section,
7. work stops for review before the next stage is added.

