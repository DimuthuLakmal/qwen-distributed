# Stage 1 TODO — Correct End-to-End Inference Service

## Goal

Build the first production-shaped inference path for the already-trained and exported Qwen-VL model.

The completed Stage 1 must support:

```text
exported Hugging Face Qwen-VL artifact
  -> model loaded once
  -> Transformers inference engine
  -> InferenceService
  -> FastAPI endpoint
  -> image + prompt
  -> generated response
```

The focus is **correctness, clean interfaces, testability, and lifecycle management**.

Do not implement vLLM, continuous batching, streaming, Kubernetes, or advanced observability in this stage.

---

## Task 1 — Inspect the Existing Repository

- [x] Read the existing project structure before adding inference code.
- [x] Identify the existing configuration-loading mechanism and config dataclasses/schemas.
- [x] Identify existing logging and utility modules that should be reused.
- [x] Identify how the exported Hugging Face model path/version is currently represented.
- [x] Inspect the training/evaluation model-loading code for reusable model/processor settings.
- [x] Inspect existing test conventions and dependency management.
- [x] Confirm whether FastAPI, Uvicorn, Pillow, and required inference dependencies already exist.
- [x] Record any required dependency additions before implementing service code.
- [x] Do not change training/evaluation code during this task unless a concrete shared-interface problem is identified.

### Task 1 completion criteria

- Existing conventions are understood.
- Planned inference files fit the repository instead of duplicating existing abstractions.
- Any necessary structural deviations from `CLAUDE.md` are documented in this TODO before implementation.

**Structural deviations from `CLAUDE.md`'s literal file tree (intentional, per its own "avoid duplicate
config systems" rule):**
- No `src/inference/config.py`. Inference config dataclasses live in `src/config/schema.py`
  (`InferenceServiceConfig`, `InferenceModelConfig`, `GenerationDefaultsConfig`,
  `InferenceLimitsConfig`) and are loaded via new `get_inference_*_config()` methods on the
  existing `ConfigurationManager`, exactly like every other config area in this repo.
- `ConfigurationManager.__init__`'s five yaml-path arguments are now all optional (previously four
  were mandatory), so it can be constructed from `inference.yaml` alone. Backward compatible --
  existing `train.py`/`evaluate.py` call sites are unaffected (see `test/inference/test_config.py::test_existing_training_config_paths_still_work`).
- The Transformers engine (`src/inference/engines/transformers.py`) does not call
  `src/models/qwen_vl.py::build_qwen_vl` -- that builder is training-oriented (freezes modules,
  enables gradient checkpointing, forces `use_cache=False`, leaves the model off-device for FSDP).
  The engine loads the processor/model itself with inference-appropriate settings instead
  (`eval()`, `use_cache=True`, placed on its configured device). `build_qwen_vl` itself is
  unchanged.
- Tests live under `test/inference/` and `test/api/` (the repo uses a flat `test/` directory, not
  `tests/`, so this only adds subdirectories, matching existing convention).

---

## Task 2 — Add Inference Configuration

- [x] Add typed inference configuration using the project's existing config system.
- [x] Add an `inference.yaml` or equivalent configuration file.
- [x] Include model artifact path.
- [x] Include model/version identifier.
- [x] Include device selection.
- [x] Include inference dtype.
- [x] Include attention implementation if supported.
- [x] Include default generation settings.
- [x] Include server host and port.
- [x] Include request limits for prompt length, image count, image size, and maximum output tokens.
- [x] Ensure request-level generation values can only reduce/stay within server limits, not bypass them.
- [x] Add configuration validation tests.

### Minimum configuration areas

```text
service
model
generation
limits
```

### Task 2 completion criteria

- Service behavior can be changed without editing source constants.
- Invalid critical configuration fails during startup with a clear error.

---

## Task 3 — Define Canonical Inference Schemas

- [x] Create internal `GenerationConfig`.
- [x] Create internal `InferenceRequest`.
- [x] Create internal `InferenceResponse`.
- [x] Include request ID.
- [x] Include prompt.
- [x] Include optional system prompt.
- [x] Include image inputs in a form appropriate for the service layer.
- [x] Include generation options.
- [x] Include response model/version.
- [x] Include input token count.
- [x] Include output token count.
- [x] Include total inference latency.
- [x] Keep HTTP/Pydantic-specific schemas separate from internal domain schemas where that improves boundaries.
- [x] Add unit tests for schema validation.

### Task 3 completion criteria

- The service and engine communicate using stable internal objects rather than HTTP request objects.

---

## Task 4 — Implement Input Validation and Multimodal Preprocessing

- [x] Create an inference preprocessor/validator.
- [x] Validate prompt presence and configured prompt limits.
- [x] Validate image count.
- [x] Validate uploaded MIME/media type.
- [x] Validate configured maximum image byte size.
- [x] Validate decoded image dimensions/pixel limits.
- [x] Normalize images to a format accepted by the Qwen processor.
- [x] Build Qwen-compatible multimodal conversation messages.
- [x] Support optional system prompt.
- [x] Ensure generation parameters are bounded by server configuration.
- [x] Do not fetch arbitrary user-provided remote URLs.
- [x] Avoid unnecessary image copies where practical.
- [x] Add unit tests for valid and invalid inputs.

### Required test cases

- [x] text + valid image,
- [x] missing prompt,
- [x] unsupported image type,
- [x] oversized image payload,
- [x] excessive image count,
- [x] excessive pixel dimensions,
- [x] excessive `max_new_tokens`,
- [x] system prompt present,
- [x] multiple images if Stage 1 implementation supports them.

### Task 4 completion criteria

- Invalid requests fail before GPU/model execution.
- Qwen-specific message construction is not implemented inside API routes.

---

## Task 5 — Create the Inference Engine Interface

- [x] Add an abstract `InferenceEngine`.
- [x] Define generation behavior.
- [x] Define readiness/health behavior.
- [x] Keep the interface independent of Hugging Face model classes.
- [x] Decide and document whether the interface is async or sync internally.
- [x] Ensure the interface can later support a vLLM implementation without changing the service/API contract.
- [x] Add a fake/mock engine for tests.

### Task 5 completion criteria

- `InferenceService` can operate against the fake engine without importing Transformers.

---

## Task 6 — Implement the Transformers Qwen-VL Engine

- [x] Load the exported Hugging Face processor once during engine initialization.
- [x] Load the exported Qwen-VL model once during engine initialization.
- [x] Use configured dtype/device.
- [x] Set the model to `eval()`.
- [x] Ensure no gradient state is created during generation.
- [x] Apply the Qwen multimodal chat template correctly.
- [x] Use `add_generation_prompt=True`.
- [x] Convert messages/images into processor/model inputs.
- [x] Move tensor inputs to the configured device safely.
- [x] Run generation inside `torch.inference_mode()`.
- [x] Use deterministic generation by default.
- [x] Respect validated generation limits.
- [x] Remove prompt tokens before decoding generated output.
- [x] Decode only the generated continuation.
- [x] Calculate input/output token counts.
- [x] Measure inference latency.
- [x] Return a canonical `InferenceResponse`.
- [x] Implement engine readiness.
- [x] Produce clear startup errors when model loading fails.
- [x] Add engine unit tests with mocked model/processor where appropriate.
- [x] Add an optional GPU smoke test that is excluded from normal CPU-only CI.

### Stage 1 generation constraints

- No streaming.
- No continuous batching.
- No custom scheduler.
- No FSDP.
- No per-request model loading.

### Task 6 completion criteria

- A known exported model can perform a real image+text generation through the engine.

---

## Task 7 — Implement Postprocessing

- [x] Create a postprocessing component/function rather than embedding cleanup in API code.
- [x] Normalize generated text only where required.
- [x] Do not remove meaningful model output aggressively.
- [x] Keep task-specific structured-output parsing extensible for later downstream requirements.
- [x] Add unit tests.

### Task 7 completion criteria

- Engine output can be postprocessed independently from HTTP handling.

---

## Task 8 — Implement `InferenceService`

- [x] Create `InferenceService`.
- [x] Accept the engine through dependency injection.
- [x] Coordinate validation/preprocessing, engine execution, and postprocessing.
- [x] Keep model/runtime-specific code out of the service.
- [x] Define clear exceptions for expected inference/service failures.
- [x] Ensure request ID is preserved end-to-end.
- [x] Add unit tests using the fake engine.

### Task 8 completion criteria

- The complete application path can be tested without a GPU or real Qwen model.

---

## Task 9 — Implement the FastAPI Application

- [x] Create the FastAPI app factory or equivalent startup module.
- [x] Load/create the inference service during application startup/lifespan.
- [x] Ensure model loading occurs once per serving process.
- [x] Add dependency injection for the initialized service.
- [x] Add `GET /health/live`.
- [x] Add `GET /health/ready`.
- [x] Add `POST /v1/generate`.
- [x] Use clear request/response Pydantic schemas.
- [x] Support image upload in a safe initial API contract.
- [x] Return correct HTTP status codes for validation failures.
- [x] Map unexpected inference errors to sanitized server errors.
- [x] Do not expose stack traces.
- [x] Add request ID middleware or equivalent request ID handling.
- [x] Do not log raw image bytes.
- [x] Do not log raw prompts/responses by default.
- [x] Add API tests using FastAPI's test client and the fake engine.

### Health behavior

`/health/live`:
- succeeds when the process/API is alive.

`/health/ready`:
- succeeds only when the engine/model is initialized and ready.

### Task 9 completion criteria

- API integration tests run without requiring a GPU.
- Route handlers contain no Qwen/tokenizer/generation implementation logic.

---

## Task 10 — Add Service Entrypoint

- [x] Create `src/pipeline/serve.py` or adapt to existing pipeline conventions.
- [x] Load inference configuration.
- [x] Configure logging.
- [x] Start the FastAPI/Uvicorn service.
- [x] Ensure graceful cleanup of engine resources where applicable.
- [x] Keep CLI/server bootstrap concerns out of `api/app.py`.
- [x] Add an example launch command to documentation.

### Example desired usage

```bash
python -m src.pipeline.serve --config configs/inference.yaml
```

or the repository's equivalent standard.

### Task 10 completion criteria

- A developer can start the service from a documented command using an exported model path.

---

## Task 11 — Add Baseline Observability

- [x] Log request ID.
- [x] Log model/version.
- [x] Log request success/failure.
- [x] Log total latency.
- [x] Log input token count.
- [x] Log output token count.
- [x] Ensure sensitive request content is not logged by default.
- [x] Ensure duplicate logs are not produced unnecessarily.
- [x] Keep metrics integration extensible for Stage 3.

### Task 11 completion criteria

- A failed request can be traced using its request ID without exposing user content.

---

## Task 12 — End-to-End Verification

- [~] Run all existing training/evaluation tests relevant to regression safety. (test collection verified clean -- see Known limitations)
- [x] Run all new inference unit tests.
- [x] Run all API integration tests.
- [~] Run lint/type/static checks used by the repository. (none configured beyond pytest -- see Known limitations)
- [ ] Start the service locally/within the target environment. (deferred to a compute node -- see Known limitations)
- [~] Verify `/health/live`. (verified via FastAPI TestClient, not a live process)
- [~] Verify `/health/ready`. (verified via FastAPI TestClient, not a live process)
- [ ] Send at least one real image+prompt request to `/v1/generate`. (deferred to a compute node)
- [ ] Confirm a valid generated response. (deferred to a compute node)
- [~] Confirm the model was loaded only once. (verified by design/tests -- lifespan calls engine_factory exactly once; not observed on a live GPU process)
- [x] Confirm invalid requests are rejected before model execution.
- [x] Confirm no raw image/prompt content appears in normal logs.
- [ ] Record basic latency and GPU memory usage from the smoke test for reference only; do not optimize yet. (deferred to a compute node)

### Task 12 completion criteria

- The baseline end-to-end inference path works with the exported Qwen-VL artifact.
- Existing training/evaluation behavior remains intact.

---

## Out of Scope for Stage 1

Do **not** implement these items yet:

- [ ] vLLM engine,
- [ ] continuous/in-flight batching,
- [ ] request scheduler/queue,
- [ ] token streaming,
- [ ] SSE/WebSockets,
- [ ] tensor parallelism,
- [ ] FSDP inference,
- [ ] multi-GPU serving,
- [ ] quantized inference,
- [ ] Redis/Kafka queues,
- [ ] arbitrary remote image URLs,
- [ ] authentication system,
- [ ] advanced rate limiting,
- [ ] Prometheus/OpenTelemetry integration,
- [ ] Kubernetes,
- [ ] autoscaling,
- [ ] canary deployment,
- [ ] model registry integration.

These belong to later stages after Stage 1 has been reviewed.

---

## Review

### Files added/changed

**New:**
- `configs/inference.yaml`
- `src/inference/__init__.py`, `schema.py`, `preprocessing.py`, `postprocessing.py`, `service.py`
- `src/inference/engines/__init__.py`, `base.py`, `fake.py`, `transformers.py`
- `src/api/__init__.py`, `app.py`, `dependencies.py`
- `src/api/routes/__init__.py`, `health.py`, `inference.py`
- `src/api/middleware/__init__.py`, `request_id.py`, `logging.py`, `error_handler.py`
- `src/pipeline/serve.py`
- `requirements-inference.txt`
- `test/inference/test_config.py`, `test_schema.py`, `test_preprocessing.py`, `test_fake_engine.py`,
  `test_transformers_engine.py`, `test_transformers_engine_gpu.py`, `test_postprocessing.py`,
  `test_service.py`
- `test/api/test_health.py`, `test_generate.py`

**Changed:**
- `src/config/schema.py` -- added `InferenceServiceConfig`, `InferenceModelConfig`,
  `GenerationDefaultsConfig`, `InferenceLimitsConfig`.
- `src/config/configuration.py` -- made all `ConfigurationManager.__init__` yaml-path arguments
  optional (backward compatible), added a `_require()` guard used by every getter, added
  `get_inference_service_config()`, `get_inference_model_config()`,
  `get_generation_defaults_config()` (also validates `generation.max_new_tokens <=
  limits.max_new_tokens` at load time), `get_inference_limits_config()`.

No training/evaluation behavior was changed; `build_qwen_vl`, `export_huggingface_model`, the
trainer, evaluator, and data pipeline are untouched.

### Architecture implemented

```
Client -> POST /v1/generate (multipart: JSON "request" field + image files)
  -> src/api/routes/inference.py            (parsing only; bounded image read)
  -> src/inference/service.py InferenceService.generate()  (via asyncio.to_thread)
       -> src/inference/preprocessing.py     (validation + limits + Qwen message building)
       -> InferenceEngine.generate()         (interface; sync)
            -> src/inference/engines/transformers.py TransformersQwenVLEngine
                 -> processor.apply_chat_template -> model.generate() -> decode
       -> src/inference/postprocessing.py    (text normalization)
  -> GenerateResponseSchema -> HTTP response
```

- `InferenceEngine` (`src/inference/engines/base.py`) is a small sync interface
  (`generate`/`health`); `FakeInferenceEngine` fully replaces it in service/API tests with zero
  Transformers/torch imports. A future vLLM engine implements the same interface.
- `InferenceRequest`/`InferenceResponse`/`GenerationConfig`/`ImageInput` (`src/inference/schema.py`)
  are the model-agnostic domain objects `InferenceService` and engines exchange.
- Config, request-parsing/validation, engine execution, and text postprocessing are in separate
  modules per `claude.md`'s dependency diagram; route handlers contain no Qwen/tokenizer/generation
  code (verified by tests exercising the API purely through `FakeInferenceEngine`).
- `src/pipeline/serve.py` is the only place that wires a real `TransformersQwenVLEngine` into
  `create_app()` and calls `uvicorn.run()`.

### Tests executed

Environment: `module load Anaconda3/2022.10` + `conda activate llm` (existing project env; `pytest`
was additionally `pip install`ed into it with user approval).

```
python -m pytest test/inference test/api -v
```
Result: **60 passed, 1 skipped** (the GPU smoke test, which skips itself because no exported model
exists at `results/final` yet -- see Known limitations).

Existing training/eval test files (`test/test_data_pipeline.py`, `test/test_fsdp.py`,
`test/test_qwen_vl.py`) were `--collect-only`'d to confirm the `ConfigurationManager` signature
change doesn't break their imports (7 tests collected cleanly). They were not executed end-to-end
here: they download the real 8B model and/or require GPU + `torch.distributed`, and are documented
in their own docstrings as compute-node-only, matching this session's HPC constraints (no GPU work
on the shared login node).

### End-to-end verification

Everything reachable without a GPU or the exported model was verified via automated tests
(config loading/validation, schema, preprocessing's full required-case matrix, fake/mocked engine
behavior, `InferenceService` orchestration and exceptions, request-ID propagation, structured
logging content, and the full FastAPI app via `TestClient` + `FakeInferenceEngine` for both health
endpoints and `/v1/generate` including every error-status mapping).

Not executed in this session (see Known limitations): starting `src/pipeline/serve.py` as a live
process, a real image+prompt request against the real exported Qwen-VL model, and GPU
latency/memory measurements.

### Known limitations

- **No GPU smoke test run.** `results/final` (the exported HF artifact) doesn't exist yet in this
  environment, and the shared login node this session ran on cannot do GPU work per HPC policy.
  `test/inference/test_transformers_engine_gpu.py` is written and skips cleanly when the model is
  missing; run it on a compute node once an export exists:
  `python -m pytest test/inference/test_transformers_engine_gpu.py -v -s`. Likewise,
  `python -m src.pipeline.serve --config configs/inference.yaml` plus a real multipart request was
  not attempted here.
- **No lint/type-checking tooling exists in this repo** (no ruff/mypy/flake8 config found) --
  "run lint/type/static checks" in Task 12 reduces to the test suite itself for now.
- Only `max_new_tokens` has an explicit configured ceiling (`limits.max_new_tokens`). `num_beams`,
  `do_sample`, and `temperature` are type/range-validated but not separately capped by a limits
  field; a request could still ask for a large `num_beams` within Stage 1. Worth a `max_num_beams`
  limit if beam search cost becomes a concern.
- No request timeout is enforced around `model.generate()`; a pathological request (e.g. large
  `max_new_tokens` with beam search) can hold the engine for a while. Stage 1 explicitly excludes a
  scheduler/timeout policy (that's Stage 3 scope), but it's worth flagging here.
- `TransformersQwenVLEngine` processes exactly one request per `generate()` call (no batching), per
  Stage 1 scope.

### Security review

- No credentials/secrets were added (confirmed by review of all new files).
- Raw image bytes are never logged (checked in `test/inference/test_service.py`'s logging tests and
  by code review of `middleware/logging.py`, `service.py`, `error_handler.py`).
- Prompts/responses are not logged by default (same tests; logs only carry request ID, model
  version, success/failure, latency, and token counts).
- No arbitrary remote URL fetching was added; images only arrive as uploaded multipart file parts.
- Upload size/type/pixel limits are enforced pre-model, at two layers: a bounded streaming read in
  the route (`_read_bounded`, aborts mid-upload past `limits.max_image_bytes`) and full
  format/dimension validation in `preprocessing.py` (`test_preprocessing.py`,
  `test_generate.py::test_generate_oversized_image_returns_413` /
  `test_generate_unsupported_image_type_returns_415`).
- User generation settings cannot exceed server limits: `max_new_tokens` is clamped/rejected against
  `limits.max_new_tokens`; unknown override keys are rejected outright
  (`test_generate_excessive_max_new_tokens_returns_400`,
  `test_unknown_generation_override_key_is_rejected`).
- Errors do not expose stack traces or internal filesystem details: unexpected exceptions are
  caught by a catch-all handler that logs full detail server-side (with request ID) but returns a
  generic sanitized message to the client
  (`test_generate_unexpected_engine_error_returns_500_without_stack_trace`).

### Recommended Stage 2 work

Do not implement it yet.

- Run the deferred GPU smoke test and a live `serve.py` + real request once an exported model is
  available, and record baseline latency/GPU memory as this stage's task intended.
- Benchmark the Transformers engine's real throughput/latency before deciding whether/how urgently
  a vLLM engine is needed.
- Consider a `max_num_beams` (or similar) limit if beam search is expected to be used with
  user-controlled `num_beams`.
- Design streaming/continuous batching only after Stage 1 is reviewed and approved, per `claude.md`.
