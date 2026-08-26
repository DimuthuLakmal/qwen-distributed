"""
Inference engine interface.

`InferenceService`/the API layer depend only on this interface, never on a
concrete runtime (Transformers, and later vLLM). This is what keeps the
runtime replaceable.

Sync-by-design
--------------
`generate()` and `health()` are plain synchronous methods, not `async def`.
The real work behind `generate()` (`model.generate()`) is a blocking
GPU/CPU-bound call; wrapping it in `async def` would not make it
non-blocking, it would just hide a blocking call inside a coroutine and risk
stalling the event loop if awaited directly. The FastAPI layer is the single
async boundary: it runs `InferenceService.generate(...)` (itself sync, for
the same reason and for trivial unit testing with `FakeInferenceEngine`) via
`asyncio.to_thread(...)`. `health()` is expected to be a cheap, non-blocking
readiness flag check, so it is called directly from the async route without
offloading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.inference.schema import InferenceRequest, InferenceResponse


class InferenceEngine(ABC):
    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run generation for a single validated request."""

    @abstractmethod
    def health(self) -> bool:
        """Return True once the engine has finished loading and can serve requests."""
