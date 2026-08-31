#!/usr/bin/env bash
#
# Start one GPU worker process per physical GPU for local multi-GPU serving.
# Each worker gets CUDA_VISIBLE_DEVICES=<i>, worker-id gpu-<i>, port 810<i>.
# Does NOT start the public router -- start it separately with:
#   python -m src.pipeline.serve --config configs/inference.yaml
#
# Usage: scripts/start_workers.sh <num_gpus> [config_path]
# Ctrl-C (or any TERM/INT) stops all workers started by this script.

set -euo pipefail

NUM_GPUS="${1:?usage: scripts/start_workers.sh <num_gpus> [config_path]}"
CONFIG="${2:-configs/inference.yaml}"

pids=()

cleanup() {
  echo "Stopping workers: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for ((i = 0; i < NUM_GPUS; i++)); do
  port=$((8100 + i))
  echo "Starting worker gpu-$i on port $port (CUDA_VISIBLE_DEVICES=$i)"
  CUDA_VISIBLE_DEVICES="$i" python -m src.pipeline.worker \
    --config "$CONFIG" --worker-id "gpu-$i" --port "$port" &
  pids+=($!)
done

wait
