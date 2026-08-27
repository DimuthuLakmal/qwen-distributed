"""
Transformers-backed Qwen-VL inference engine (Stage 1).

Loads the exported Hugging Face artifact once at construction time and
serves `generate()` calls against the already-loaded model. Deliberately
does not reuse `src.models.qwen_vl.build_qwen_vl`: that builder is
training-oriented (freezes modules, enables gradient checkpointing, forces
`use_cache=False`, and leaves the model off-device for FSDP to place). This
engine instead loads with inference-appropriate settings directly: `eval()`
mode, `use_cache=True`, and the model placed on its configured device.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from src.config.schema import InferenceModelConfig
from src.inference.engines.base import InferenceEngine
from src.inference.preprocessing import build_conversation_messages
from src.inference.schema import InferenceRequest, InferenceResponse
from src.utils.logger import logger
from src.utils.tensor_utils import move_to_device


class TransformersQwenVLEngine(InferenceEngine):
    def __init__(self, config: InferenceModelConfig, model_version: str) -> None:
        self.config = config
        self.model_version = model_version
        self.device = torch.device(config.device)
        self._ready = False

        self.processor, self.model = self._load(config)

        self._ready = True

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.perf_counter()

        messages = build_conversation_messages(request)

        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if "input_ids" not in encoded:
            raise RuntimeError("Processor output does not contain input_ids.")

        encoded = move_to_device(dict(encoded), self.device)
        prompt_length = encoded["input_ids"].shape[1]

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": request.generation.max_new_tokens,
            "num_beams": request.generation.num_beams,
            "do_sample": request.generation.do_sample,
            "pad_token_id": self.processor.pad_token_id,
        }
        if request.generation.do_sample:
            generation_kwargs["temperature"] = request.generation.temperature

        with torch.inference_mode():
            generated_ids = self.model.generate(**encoded, **generation_kwargs)

        new_tokens = generated_ids[:, prompt_length:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        generated_text = decoded[0].strip() if decoded else ""

        output_token_count = int(new_tokens.shape[1])
        latency_seconds = time.perf_counter() - start_time

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            model_version=self.model_version,
            input_token_count=int(prompt_length),
            output_token_count=output_token_count,
            latency_seconds=latency_seconds,
        )

    def health(self) -> bool:
        return self._ready

    def close(self) -> None:
        """Release GPU memory held by the model. Called on service shutdown."""

        self._ready = False
        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _load(config: InferenceModelConfig):
        processor_kwargs: dict[str, object] = {}
        if config.min_pixels:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels:
            processor_kwargs["max_pixels"] = config.max_pixels

        try:
            processor = AutoTokenizer.from_pretrained(
                config.model_path,
                **processor_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load inference processor from {config.model_path!r}: {exc}"
            ) from exc

        if processor.pad_token_id is None:
            processor.pad_token_id = processor.eos_token_id

        try:
            model = AutoModelForImageTextToText.from_pretrained(
                config.model_path,
                torch_dtype=config.dtype,
                attn_implementation=config.attn_implementation,
                low_cpu_mem_usage=True,
                device_map=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load inference model from {config.model_path!r}: {exc}"
            ) from exc

        model.config.use_cache = True

        device = torch.device(config.device)
        model = model.to(device)
        model.eval()

        logger.info(
            "Transformers Qwen-VL engine loaded | model_path=%s | device=%s",
            config.model_path,
            device,
        )

        return processor, model
