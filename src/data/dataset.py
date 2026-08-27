from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Any

from src.data.schema import Conversation, ProcessedSample
from src.data.adapters.base import DatasetAdapter
from src.data.adapters.registry import build_adapter
from src.config.schema import DataConfig


class SFTDataset:

    def __init__(
        self,
        dataset: Dataset,
        adapter: DatasetAdapter,
        processor: Any,
        max_length: int,
        supervise_im_end: bool = True,
    ):
        self.dataset = dataset
        self.adapter = adapter
        self.processor = processor
        self.max_length = max_length
        self.supervise_im_end = supervise_im_end

        self.tokenizer = processor
        # Qwen chat-template markers.
        #
        # We compute the IDs from the tokenizer rather than hard-coding
        # model-specific numeric token IDs.
        self.assistant_prefix_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )

        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        )

        if self.im_end_token_id is None:
            raise ValueError(
                "Tokenizer does not contain the Qwen <|im_end|> token."
            )

    @classmethod
    def from_config(
        cls,
        data_config: DataConfig,
        processor,
    ) -> SFTDataset:

        raw_dataset = load_dataset(
            data_config.path,
            split=data_config.split,
            cache_dir=data_config.cache_dir,
        )

        adapter = build_adapter(
            data_config.adapter,
        )

        return cls(
            dataset=raw_dataset,
            adapter=adapter,
            processor=processor,
            max_length=data_config.max_length,
        )

    def __len__(self):
        return len(self.dataset)

    def _process(
        self,
        conversation: Conversation,
    ) -> dict[str, Any]:
        """
        Convert a canonical multimodal conversation into one unbatched
        Qwen-VL training sample.

        Responsibilities:
            - apply Qwen chat template
            - tokenize text
            - preprocess images/videos
            - identify assistant tokens that contribute to SFT loss

        Does NOT:
            - pad
            - construct the final attention mask
            - construct labels
            - stack samples into a batch
        """

        messages = conversation.messages

        if not messages:
            raise ValueError("Received an empty conversation.")

        # -------------------------------------------------------------
        # 1. Chat templating + text/vision preprocessing
        # -------------------------------------------------------------
        #
        # Because the messages already contain PIL images / videos in
        # their multimodal content, a recent Transformers processor can
        # process everything directly.
        #
        # IMPORTANT:
        # add_generation_prompt=False because the assistant response is
        # already present in an SFT conversation.
        #
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        if "input_ids" not in encoded:
            raise RuntimeError(
                "Processor output does not contain input_ids."
            )

        # One dataset item -> processor returns [1, seq_len].
        # Dataset should return [seq_len]; batching happens later.
        input_ids = encoded["input_ids"].squeeze(0).to(torch.long)

        # -------------------------------------------------------------
        # 2. Determine which tokens should contribute to SFT loss
        # -------------------------------------------------------------
        #
        # Dataset understands the semantic structure of the conversation.
        # Therefore it determines the assistant-token mask.
        #
        # The collator will later convert this into labels:
        #
        #     labels = input_ids.clone()
        #     labels[~loss_mask] = -100
        #
        loss_mask = self._build_assistant_loss_mask(input_ids)

        if not torch.any(loss_mask):
            raise ValueError(
                "No assistant tokens were found in the processed sample. "
                "Check the dataset adapter and Qwen chat template."
            )

        # -------------------------------------------------------------
        # 3. Build the unbatched training sample
        # -------------------------------------------------------------

        sample: dict[str, Any] = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }

        return self._attach_multimodal_fields(encoded, sample)

    def _attach_multimodal_fields(
        self,
        encoded: dict[str, Any],
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Copy modality-ID / image / video tensors produced by the processor
        onto an unbatched sample dict. Shared by SFTDataset (full
        conversation, training) and SFTEvalDataset (prompt-only,
        generation) so both stay consistent with the processor's output.
        """

        # Newer Qwen processors can provide token-level modality IDs.
        # Shape returned by processor is normally [1, seq_len].
        if (
            "mm_token_type_ids" in encoded
            and encoded["mm_token_type_ids"] is not None
        ):
            sample["mm_token_type_ids"] = (
                encoded["mm_token_type_ids"]
                .squeeze(0)
                .to(torch.long)
            )

        # -------------------------------------------------------------
        # Image tensors
        # -------------------------------------------------------------
        #
        # DO NOT squeeze these tensors.
        #
        # pixel_values:
        #     represents all visual patches belonging to the sample
        #
        # image_grid_thw:
        #     [num_images, 3]
        #
        if (
            "pixel_values" in encoded
            and encoded["pixel_values"] is not None
        ):
            sample["pixel_values"] = encoded["pixel_values"]

        if (
            "image_grid_thw" in encoded
            and encoded["image_grid_thw"] is not None
        ):
            sample["image_grid_thw"] = encoded["image_grid_thw"]

        # -------------------------------------------------------------
        # Video tensors
        # -------------------------------------------------------------

        if (
            "pixel_values_videos" in encoded
            and encoded["pixel_values_videos"] is not None
        ):
            sample["pixel_values_videos"] = (
                encoded["pixel_values_videos"]
            )

        if (
            "video_grid_thw" in encoded
            and encoded["video_grid_thw"] is not None
        ):
            sample["video_grid_thw"] = encoded["video_grid_thw"]

        # Used by newer Qwen2.5-VL implementations for video temporal
        # positioning.
        if (
            "second_per_grid_ts" in encoded
            and encoded["second_per_grid_ts"] is not None
        ):
            sample["second_per_grid_ts"] = (
                encoded["second_per_grid_ts"]
            )

        return sample

    def _build_assistant_loss_mask(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct a boolean mask where True means that the token should
        contribute to the SFT language-modeling loss.

        For Qwen:

            <|im_start|>user
            ...
            <|im_end|>
            <|im_start|>assistant
            ANSWER
            <|im_end|>

        becomes:

            user/system/header tokens -> False
            assistant answer tokens   -> True

        By default <|im_end|> for assistant turns is also supervised so
        that the model learns when to terminate its response.
        """

        loss_mask = torch.zeros(
            input_ids.shape,
            dtype=torch.bool,
        )

        ids = input_ids.tolist()
        prefix = self.assistant_prefix_ids

        search_start = 0
        found_assistant = False

        while search_start < len(ids):

            assistant_start = self._find_subsequence(
                sequence=ids,
                subsequence=prefix,
                start=search_start,
            )

            if assistant_start == -1:
                break

            found_assistant = True

            # Skip:
            #
            # <|im_start|>assistant\n
            #
            # We only supervise the actual assistant response.
            content_start = assistant_start + len(prefix)

            try:
                im_end_position = ids.index(
                    self.im_end_token_id,
                    content_start,
                )
            except ValueError as exc:
                raise ValueError(
                    "Found an assistant turn without a matching "
                    "<|im_end|> token."
                ) from exc

            if self.supervise_im_end:
                content_end = im_end_position + 1
            else:
                content_end = im_end_position

            loss_mask[content_start:content_end] = True

            search_start = im_end_position + 1

        if not found_assistant:
            return loss_mask

        # Defensive masking. These should normally occur only in user
        # content, but they should never be LM targets.
        vision_token_ids = self._get_vision_special_token_ids()

        for token_id in vision_token_ids:
            loss_mask[input_ids == token_id] = False

        return loss_mask

    @staticmethod
    def _find_subsequence(
        sequence: list[int],
        subsequence: list[int],
        start: int = 0,
    ) -> int:
        """
        Find the first occurrence of `subsequence` in `sequence`.

        Returns -1 when not found.
        """

        if not subsequence:
            return -1

        last_start = len(sequence) - len(subsequence)

        for i in range(start, last_start + 1):
            if sequence[i : i + len(subsequence)] == subsequence:
                return i

        return -1

    def _get_vision_special_token_ids(self) -> set[int]:
        """
        Resolve Qwen vision special-token IDs from the tokenizer instead
        of relying on hard-coded numeric IDs.
        """

        token_names = (
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        )

        token_ids: set[int] = set()

        unk_token_id = self.tokenizer.unk_token_id

        for token in token_names:
            token_id = self.tokenizer.convert_tokens_to_ids(token)

            if token_id is None:
                continue

            if (
                unk_token_id is not None
                and token_id == unk_token_id
            ):
                continue

            token_ids.add(token_id)

        return token_ids

    def __getitem__(self, index):
        raw_sample = self.dataset[index]
        conversation = self.adapter.convert(raw_sample)

        processed_sample = self._process(conversation)
        return processed_sample