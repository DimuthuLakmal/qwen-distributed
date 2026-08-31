"""
In-memory registry of configured GPU workers and their last-known health.

The registry never talks to workers itself -- `RouterService` probes workers
via `WorkerClient` and reports results back here. Deliberately holds no
Qwen/Transformers-related state or imports so it stays trivially reusable
by both routing strategies and, later, by tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerDescriptor:
    worker_id: str
    base_url: str


@dataclass
class WorkerRecord:
    descriptor: WorkerDescriptor
    # Unknown/unproven until the first successful status probe.
    ready: bool = False
    active_requests: int = 0


class WorkerRegistry:
    def __init__(self, workers: list[WorkerDescriptor]) -> None:
        if not workers:
            raise ValueError("WorkerRegistry requires at least one configured worker.")

        self._records: dict[str, WorkerRecord] = {
            worker.worker_id: WorkerRecord(descriptor=worker) for worker in workers
        }

    def all_workers(self) -> list[WorkerRecord]:
        return list(self._records.values())

    def healthy_workers(self) -> list[WorkerRecord]:
        return [record for record in self._records.values() if record.ready]

    def mark_ready(self, worker_id: str, active_requests: int) -> None:
        record = self._records[worker_id]
        record.ready = True
        record.active_requests = active_requests

    def mark_unavailable(self, worker_id: str) -> None:
        record = self._records[worker_id]
        record.ready = False
