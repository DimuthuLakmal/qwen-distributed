# Serving the Qwen-VL Inference Service

## Backend selection (Stage 2)

`configs/inference.yaml`'s `engine.type` selects one of two backends. The
public API (`POST /v1/generate`, `GET /health/live|ready`, request/response
schemas) is identical either way -- only the internal multi-GPU topology
differs:

```yaml
engine:
  type: "transformers"   # or "vllm"
```

```text
engine.type: "transformers"
    Public API -> application-managed Replica Router -> per-GPU Worker
    processes -> TransformersEngine -> GPU. The application creates and
    routes replicas itself (see "Transformers backend" below).

engine.type: "vllm"
    Public API -> VLLMEngine -> vLLM's own internal data-parallel
    coordinator -> GPU. vLLM creates and routes its own replicas; the
    application never spawns per-GPU worker processes or a replica router
    for this backend (see "vLLM backend" below).
```

Switching backends requires only editing `engine.type` (and, for `vllm`,
the nested `engine.vllm:` section) -- no code changes, and
`python -m src.pipeline.serve --config configs/inference.yaml` is the single
entrypoint for both.

Installed vLLM version: **0.28.0**. Integration mode: **Mode A** --
in-process `AsyncLLM` (`vllm.v1.engine.async_llm.AsyncLLM`) with vLLM's own
internal `data_parallel_size` (spawned via its `data_parallel_backend="mp"`
multiprocessing path), not an external `vllm serve` subprocess. Chosen
because the installed version's async Python API supports local
`data_parallel_size > 1` directly, so Mode A was viable without falling back
to Mode B.

## Transformers backend

Application-managed topology: one CPU-only public router in front of one or
more GPU worker processes, each holding one full model replica.

```text
router      :8000   CPU process, no model weights
worker-0    :8100   CUDA_VISIBLE_DEVICES=0
worker-1    :8101   CUDA_VISIBLE_DEVICES=1
worker-2    :8102   CUDA_VISIBLE_DEVICES=2
worker-3    :8103   CUDA_VISIBLE_DEVICES=3
```

The router and every worker read the same `configs/inference.yaml`: the
router uses `service`/`limits`/`workers`/`routing`, each worker uses
`service`/`model`/`generation`/`limits`. `configs/inference.yaml`'s
`workers:` list must match the worker processes you actually start (worker
IDs and ports).

### Launching locally

1. Start one worker process per GPU (this does **not** start the router):

   ```bash
   scripts/start_workers.sh 2   # starts gpu-0 on :8100, gpu-1 on :8101
   ```

   Equivalent to running by hand:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.worker \
       --config configs/inference.yaml --worker-id gpu-0 --port 8100
   CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.worker \
       --config configs/inference.yaml --worker-id gpu-1 --port 8101
   ```

2. Start the public router, in a separate terminal/process:

   ```bash
   python -m src.pipeline.serve --config configs/inference.yaml
   ```

3. Send requests to the router (`http://127.0.0.1:8000`), never directly to
   a worker's `/internal/*` API.

For a single GPU, run one worker (`scripts/start_workers.sh 1`, or the
`CUDA_VISIBLE_DEVICES=0 ...` command above) plus the router -- the public
API contract is identical whether one or many replicas are behind it.

### Retry policy

The router does **not** retry a `/v1/generate` call after forwarding it to a
worker. A worker request that fails partway through generation is not safe
to silently replay (side effects on the worker's active-request count,
partial GPU work, ambiguous idempotency for the caller) -- `RouterService.generate`
marks the worker unavailable on a connection/unavailable failure and
propagates the error to the client as a single failed request. The client
decides whether to retry, at which point `ReplicaRouter` will naturally route
the new attempt to a different healthy worker if the failed one has already
been marked unavailable. The one exception is worker *selection*: if no
worker is currently marked ready, the router fails fast with 503 rather than
guessing -- it never blocks retrying against a worker it hasn't already
confirmed is unreachable.

### What NOT to do (Transformers)

- Do **not** use `uvicorn --workers N` to add GPU replicas -- that
  spawns multiple event loops in the same process/host without any GPU
  assignment, not independent model replicas.
- Do **not** point external clients at a worker's port directly; only the
  router's `/v1/generate` and `/health/*` are public-facing.
- Do **not** run two worker processes with the same `CUDA_VISIBLE_DEVICES`
  value -- that loads two full model replicas onto the same physical GPU.

### Shutdown

- `scripts/start_workers.sh` traps `EXIT`/`INT`/`TERM`: `Ctrl-C` in that
  terminal stops every worker it started.
- The router is a separate foreground process; stop it with `Ctrl-C` (or
  send `SIGTERM`) independently. Router shutdown closes its pooled HTTP
  connections to workers but does not stop the workers themselves.

## vLLM backend

Single-process topology: no router, no per-GPU worker processes.
`VLLMEngine` owns one in-process `AsyncLLM`, which owns its own internal
data-parallel replicas.

```text
serve   :8000   one process, owns the GPU(s) directly
            -> VLLMEngine -> AsyncLLM
                -> DP rank 0 -> GPU 0
                -> DP rank 1 -> GPU 1   (if data_parallel_size > 1)
```

Only `service`/`engine`/`model`/`generation`/`limits` in
`configs/inference.yaml` are read for this backend; `workers:`/`routing:`
are ignored (those are Transformers-only).

### Config

```yaml
engine:
  type: "vllm"

  vllm:
    data_parallel_size: 1        # positive int, or "auto" (-> visible GPU count)
    tensor_parallel_size: 1      # must be 1 in this stage (no tensor parallelism)
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    max_num_seqs: 16

    limit_mm_per_prompt:
      image: 4
      video: 0

    mm_processor_kwargs:
      max_pixels: 1003520
```

Assumes the model fits on one GPU: `tensor_parallel_size = 1`,
`data_parallel_size = N` full replicas. `data_parallel_size: "auto"` resolves
to `torch.cuda.device_count()` in the process that starts the engine (via
`src/inference/engines/factory.py`), and the resolved value is logged at
startup; an explicit integer always overrides `"auto"`.

### Launching locally

Single-GPU (`data_parallel_size: 1`):

```bash
python -m src.pipeline.serve --config configs/inference.yaml
```

Multi-GPU internal DP (`data_parallel_size: N`, N ≤ visible GPUs): set
`engine.vllm.data_parallel_size: N` in the config and run the exact same
command -- vLLM creates and schedules across all `N` replicas internally
(via its `data_parallel_backend="mp"` multiprocessing path); the application
does not set `CUDA_VISIBLE_DEVICES` per replica or start additional
processes itself.

```bash
python -m src.pipeline.serve --config configs/inference.yaml
```

### What NOT to do (vLLM)

- Do **not** start one `VLLMEngine`/process per GPU and front them with a
  router or load balancer -- that duplicates vLLM's own internal DP
  scheduling and defeats the point of this backend.
- Do **not** place an application-level lock/semaphore around
  `VLLMEngine.generate()` -- concurrent requests must be allowed to enter
  the async engine so vLLM's own continuous batching can interleave them.
- Do **not** run `scripts/start_workers.sh` or `src.pipeline.worker` for
  this backend; they are Transformers-only and are never invoked by the
  vLLM path.

### Shutdown

`python -m src.pipeline.serve` is the only process; `Ctrl-C`/`SIGTERM`
triggers `create_vllm_app`'s lifespan shutdown, which calls
`VLLMEngine.close()` -> `AsyncLLM.shutdown()`, releasing GPU memory and
terminating any engine-core subprocesses vLLM spawned for
`data_parallel_size > 1`. Verify with `nvidia-smi`/`ps` after shutdown that
no orphan engine-core processes remain.

## Known limitations

- **Multi-GPU DP is implemented but not verified on this dev machine** --
  it has exactly one visible GPU, so `data_parallel_size >= 2` has only been
  exercised with a mocked engine (`test/inference/engines/test_vllm_engine.py`),
  not real hardware. Verify on a multi-GPU node before relying on it in
  production.
- **No exported model artifact existed on this dev machine at implementation
  time** (`results/final` was absent) -- the real single-GPU vLLM path
  (`test/inference/engines/test_vllm_engine_gpu.py`) self-skips until the
  export step has been run.
- **`num_beams > 1` is not supported by the vLLM backend** (no beam-search
  equivalent in the installed `SamplingParams`); `VLLMEngine.generate()`
  raises `InvalidGenerationOptionsError` (HTTP 400) rather than silently
  falling back to greedy/sampling decoding.
- **Backend parity (todo.md Task 17) requires both conda environments at
  once** (a Transformers-capable env and the separate `vllm` env on this
  machine), which the current environment split makes impractical as a
  single automated test -- run it as a manual comparison on a compute node
  with both installed rather than expecting CI coverage for it.
