from typing import Any

import torch

from src.data.collate import attach_vision_fields


class GenerationCollator:
    """
    Collates a list of unbatched samples produced by
    SFTEvalDataset._process into a batch ready for model.generate().

    Each input sample is a dict with (at minimum):
        input_ids: [seq_len]
        reference: str

    Unlike SFTCollator, padding is applied on the LEFT. Hugging Face's
    generate() appends new tokens on the right, so every sample in the
    batch must already end at the same position for batched generation
    to produce correctly aligned outputs.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Received an empty batch.")

        batch_size = len(samples)
        max_length = max(sample["input_ids"].shape[0] for sample in samples)

        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        for i, sample in enumerate(samples):
            ids = sample["input_ids"]
            seq_len = ids.shape[0]

            input_ids[i, max_length - seq_len:] = ids
            attention_mask[i, max_length - seq_len:] = 1

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "references": [sample["reference"] for sample in samples],
        }

        attach_vision_fields(samples, batch)

        return batch
