from src.data.adapters.base import DatasetAdapter
from src.data.schema import Conversation
from typing import Any


class LLaVAInstructAdapter(DatasetAdapter):

    """
    Adapter for:

        trl-lib/llava-instruct-mix

    Expected raw sample structure:

        {
            "images": [PIL.Image.Image, ...],
            "prompt": [
                {
                    "role": "user",
                    "content": "..."
                },
                ...
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "..."
                }
            ]
        }

    The adapter converts this into Qwen-style multimodal messages:

        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": PIL.Image.Image(...)
                    },
                    {
                        "type": "text",
                        "text": "..."
                    }
                ]
            },
            ...
        ]
    """
    def convert(self, sample: dict[str, Any]) -> Conversation:

        images = sample["images"] or []
        prompt = sample["prompt"]
        completion = sample["completion"]

        # prompt may contain multiple conversation turns.
        raw_messages = [*prompt, *completion]

        messages: list[dict[str, Any]] = []

        # The LLaVA Instruct Mix image belongs to the conversation.
        # Attach it to the first user turn.
        image_attached = False

        for raw_message in raw_messages:
            role = raw_message["role"]
            text = raw_message["content"]

            if role == "user":
                content: list[dict[str, Any]] = []

                if not image_attached and images:
                    content.extend(
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in images
                    )

                    image_attached = True

                content.append(
                    {
                        "type": "text",
                        "text": text,
                    }
                )

            else:
                # System / assistant messages contain text only.
                content = [
                    {
                        "type": "text",
                        "text": text,
                    }
                ]

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return Conversation(messages=messages)

