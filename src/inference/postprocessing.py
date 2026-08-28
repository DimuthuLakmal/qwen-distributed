"""
Postprocessing of engine output into the final canonical response.

Kept intentionally minimal for Stage 1: only whitespace normalization.
Structured/task-specific output parsing (e.g. extracting JSON, stripping a
model-specific stop phrase) can be added here later without touching the
engine or the API layer.
"""

from __future__ import annotations

from dataclasses import replace

from src.inference.schema import InferenceResponse


def postprocess_response(response: InferenceResponse) -> InferenceResponse:
    normalized_text = _normalize_text(response.generated_text)
    if normalized_text == response.generated_text:
        return response

    return replace(response, generated_text=normalized_text)


def _normalize_text(text: str) -> str:
    return text.strip()
