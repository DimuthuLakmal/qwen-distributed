from typing import Any

import torch


IGNORE_INDEX = -100


class SFTCollator:
    """
    Collates a list of unbatched samples produced by SFTDataset._process
    into a single padded batch ready for a Qwen-VL forward pass.

    Each input sample is a dict with (at minimum):
        input_ids: [seq_len]
        loss_mask: [seq_len]  (bool)

    and optionally:
        mm_token_type_ids, pixel_values, image_grid_thw,
        pixel_values_videos, video_grid_thw, second_per_grid_ts

    Responsibilities:
        - right-pad input_ids / loss_mask / mm_token_type_ids to the
          batch's longest sequence
        - build the attention_mask
        - turn loss_mask into labels:
              labels = input_ids.clone()
              labels[~loss_mask] = IGNORE_INDEX
        - concatenate per-sample vision tensors along dim 0, since
          Qwen-VL consumes flattened patches segmented by *_grid_thw
          rather than a fixed per-sample image shape
    """

    def __init__(self, pad_token_id: int, ignore_index: int = IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

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
        labels = torch.full(
            (batch_size, max_length),
            fill_value=self.ignore_index,
            dtype=torch.long,
        )

        has_mm_token_type_ids = all(
            "mm_token_type_ids" in sample for sample in samples
        )
        mm_token_type_ids = (
            torch.zeros((batch_size, max_length), dtype=torch.long)
            if has_mm_token_type_ids
            else None
        )

        for i, sample in enumerate(samples):
            ids = sample["input_ids"]
            loss_mask = sample["loss_mask"]
            seq_len = ids.shape[0]

            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1

            sample_labels = ids.clone()
            sample_labels[~loss_mask] = self.ignore_index
            labels[i, :seq_len] = sample_labels

            if mm_token_type_ids is not None:
                mm_token_type_ids[i, :seq_len] = sample["mm_token_type_ids"]

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if mm_token_type_ids is not None:
            batch["mm_token_type_ids"] = mm_token_type_ids

        attach_vision_fields(samples, batch)

        return batch


def attach_vision_fields(samples: list[dict[str, Any]], batch: dict[str, Any]) -> None:
    """
    Merge per-sample vision tensors into `batch`, in place.

    Vision tensors are flattened patches / grid metadata, not
    fixed-shape per-sample images, so they concatenate on dim 0 instead
    of stacking on a new batch dimension. Shared by SFTCollator
    (training) and GenerationCollator (eval) so both batch multimodal
    inputs the same way.
    """

    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        values = [sample[key] for sample in samples if key in sample]
        if values:
            batch[key] = torch.cat(values, dim=0)

    second_per_grid_ts = _merge_optional(samples, "second_per_grid_ts")
    if second_per_grid_ts is not None:
        batch["second_per_grid_ts"] = second_per_grid_ts


def _merge_optional(samples: list[dict[str, Any]], key: str):
    values = [sample[key] for sample in samples if key in sample]

    if not values:
        return None

    if isinstance(values[0], torch.Tensor):
        return torch.cat(values, dim=0)

    merged: list[Any] = []
    for value in values:
        merged.extend(value)

    return merged
