"""
Inference engine interface.

`InferenceService`/the API layer depend only on this interface, never on a
concrete runtime (Transformers, and later vLLM). This is what keeps the
runtime replaceable.

Async-by-design
---------------
`generate()` and `health()` are `async def`. This is what lets a vLLM-backed
engine drive `AsyncLLM`'s async-generator `generate()` API directly on the
event loop, so multiple in-flight requests can be interleaved by vLLM's own
scheduler instead of being serialized behind an application-level lock or
thread pool. A synchronous, blocking engine (Transformers) still fits this
contract cleanly: it offloads its own blocking call internally via
`asyncio.to_thread(...)` (see `TransformersQwenVLEngine.generate`), so the
thread-offload boundary lives inside the engine rather than at the service/
route layer. `health()` is expected to be cheap; a synchronous engine can
implement it as a trivial `async def` with no `await` inside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.inference.schema import InferenceRequest, InferenceResponse


class InferenceEngine(ABC):
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run generation for a single validated request."""

    @abstractmethod
    async def health(self) -> bool:
        """Return True once the engine has finished loading and can serve requests."""
