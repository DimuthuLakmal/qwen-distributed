"""
In-memory fake engine for service/API tests.

Deliberately has no Transformers/torch import so `InferenceService` and the
FastAPI app can be fully tested without a GPU or the exported model.
"""

from __future__ import annotations

from src.inference.engines.base import InferenceEngine
from src.inference.schema import InferenceRequest, InferenceResponse


class FakeInferenceEngine(InferenceEngine):
    def __init__(
        self,
        generated_text: str = "a generated response",
        model_version: str = "fake-model-v0",
        ready: bool = True,
        latency_seconds: float = 0.01,
    ) -> None:
        self.generated_text = generated_text
        self.model_version = model_version
        self.ready = ready
        self.latency_seconds = latency_seconds
        self.received_requests: list[InferenceRequest] = []

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.received_requests.append(request)

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=self.generated_text,
            model_version=self.model_version,
            input_token_count=len(request.prompt.split()),
            output_token_count=len(self.generated_text.split()),
            latency_seconds=self.latency_seconds,
        )

    def health(self) -> bool:
        return self.ready
