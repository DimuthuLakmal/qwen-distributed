from __future__ import annotations

from typing import Any

import torch

from src.data.dataset import SFTDataset
from src.data.schema import Conversation


class SFTEvalDataset(SFTDataset):
    """
    Generation-eval variant of SFTDataset.

    Reuses SFTDataset's adapter/processor/multimodal-field plumbing but
    processes only the conversation *up to* the last assistant turn
    (add_generation_prompt=True), so the sample can be fed straight into
    model.generate(). The ground-truth assistant response is kept as
    plain text under "reference" for scoring instead of being tokenized
    into loss_mask/labels.
    """

    def _process(
        self,
        conversation: Conversation,
    ) -> dict[str, Any]:

        messages = conversation.messages

        if not messages:
            raise ValueError("Received an empty conversation.")

        assistant_index = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i]["role"] == "assistant"
            ),
            None,
        )

        if assistant_index is None:
            raise ValueError(
                "No assistant turn found to use as the reference answer."
            )

        reference = self._extract_text(messages[assistant_index])
        prompt_messages = messages[:assistant_index]

        encoded = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if "input_ids" not in encoded:
            raise RuntimeError(
                "Processor output does not contain input_ids."
            )

        input_ids = encoded["input_ids"].squeeze(0).to(torch.long)

        sample: dict[str, Any] = {
            "input_ids": input_ids,
            "reference": reference,
        }

        return self._attach_multimodal_fields(encoded, sample)

    @staticmethod
    def _extract_text(message: dict[str, Any]) -> str:
        content = message["content"]

        if isinstance(content, str):
            return content.strip()

        return " ".join(
            part["text"].strip()
            for part in content
            if part.get("type") == "text"
        ).strip()
