from __future__ import annotations

import re
import string
from typing import Callable


def _normalize(text: str) -> str:
    """
    Standard SQuAD/VQA-style answer normalization: lowercase, strip
    punctuation/articles, collapse whitespace.
    """

    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def exact_match(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0

    matches = sum(
        _normalize(prediction) == _normalize(reference)
        for prediction, reference in zip(predictions, references)
    )

    return matches / len(predictions)


METRIC_REGISTRY: dict[str, Callable[[list[str], list[str]], float]] = {
    "exact_match": exact_match,
}


def compute_metrics(
    metric_names: list[str],
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have the same length: "
            f"{len(predictions)} != {len(references)}"
        )

    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric: {name}. Available metrics: {list(METRIC_REGISTRY)}"
            )

        results[name] = METRIC_REGISTRY[name](predictions, references)

    return results
