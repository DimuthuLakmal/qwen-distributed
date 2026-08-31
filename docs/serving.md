# Serving the Qwen-VL Inference Service

Stage 1B topology: one CPU-only public router in front of one or more
GPU worker processes, each holding one full model replica.

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

## Launching locally

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

## Retry policy

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

## What NOT to do

- Do **not** use `uvicorn --workers N` to add GPU replicas -- that
  spawns multiple event loops in the same process/host without any GPU
  assignment, not independent model replicas.
- Do **not** point external clients at a worker's port directly; only the
  router's `/v1/generate` and `/health/*` are public-facing.
- Do **not** run two worker processes with the same `CUDA_VISIBLE_DEVICES`
  value -- that loads two full model replicas onto the same physical GPU.

## Shutdown

- `scripts/start_workers.sh` traps `EXIT`/`INT`/`TERM`: `Ctrl-C` in that
  terminal stops every worker it started.
- The router is a separate foreground process; stop it with `Ctrl-C` (or
  send `SIGTERM`) independently. Router shutdown closes its pooled HTTP
  connections to workers but does not stop the workers themselves.
