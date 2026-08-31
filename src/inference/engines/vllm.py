"""
vLLM-backed Qwen-VL inference engine (Stage 2).

Uses a single in-process `AsyncLLM` with vLLM's own internal data-parallel
replica management (`data_parallel_size` > 1 spawns vLLM-managed engine-core
subprocesses internally -- this module never creates per-GPU worker
processes or an application-level replica router itself). Resolves
`engine.vllm.model_id` itself (a Hugging Face hub repo id or a local model
directory, downloaded/cached under `engine.vllm.cache_dir`) -- independent of
the `transformers:` config section, which only the Transformers engine reads.
Never loads a second Transformers model or calls `model.to(...)`.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.config.profiler import ProfilerConfig as VLLMProfilerConfig
from vllm.inputs.llm import TextPrompt
from vllm.v1.engine.async_llm import AsyncLLM

from src.config.schema import ProfilerConfig, VLLMConfig
from src.inference.engines.base import InferenceEngine
from src.inference.exception import InvalidGenerationOptionsError
from src.inference.preprocessing import build_conversation_messages
from src.inference.schema import GenerationConfig, InferenceRequest, InferenceResponse
from src.utils.logger import logger


def _build_vllm_profiler_config(profiler_config: ProfilerConfig | None) -> VLLMProfilerConfig | None:
    """
    Translates our generic ProfilerConfig into vLLM's own engine-native
    profiler config. This is the only kind of torch.profiler that reaches
    into vLLM's data-parallel engine-core subprocesses -- wrapping our own
    process in torch.profiler.profile() (as the Transformers engine does)
    would only capture this process's async/HTTP glue, not the actual model
    forward passes.
    """
    if profiler_config is None or not profiler_config.enabled:
        return None

    return VLLMProfilerConfig(
        profiler="torch",
        torch_profiler_dir=os.path.abspath(profiler_config.output_dir),
        wait_iterations=profiler_config.wait_steps,
        warmup_iterations=profiler_config.warmup_steps,
        active_iterations=profiler_config.active_steps,
        torch_profiler_record_shapes=profiler_config.record_shapes,
        torch_profiler_with_memory=profiler_config.profile_memory,
        torch_profiler_with_stack=profiler_config.with_stack,
    )


class VLLMEngine(InferenceEngine):
    def __init__(
        self,
        vllm_config: VLLMConfig,
        model_version: str,
        data_parallel_size: int,
        profiler_config: ProfilerConfig | None = None,
    ) -> None:
        self.model_version = model_version
        self._ready = False
        self._profiler_enabled = profiler_config is not None and profiler_config.enabled
        self._profiler_started = False

        engine_args_kwargs: dict = dict(
            model=vllm_config.model_id,
            download_dir=str(vllm_config.cache_dir) if vllm_config.cache_dir else None,
            dtype=vllm_config.dtype,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            data_parallel_backend="mp",
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            max_model_len=vllm_config.max_model_len,
            max_num_seqs=vllm_config.max_num_seqs,
            limit_mm_per_prompt=dict(vllm_config.limit_mm_per_prompt),
            mm_processor_kwargs=dict(vllm_config.mm_processor_kwargs) or None,
        )
        # AsyncEngineArgs.profiler_config is a required-typed field (not
        # Optional) with its own disabled-by-default value -- only override
        # it when we actually want profiling, never pass None explicitly.
        vllm_profiler_config = _build_vllm_profiler_config(profiler_config)
        if vllm_profiler_config is not None:
            engine_args_kwargs["profiler_config"] = vllm_profiler_config

        engine_args = AsyncEngineArgs(**engine_args_kwargs)

        try:
            self._engine = AsyncLLM.from_engine_args(engine_args)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to start vLLM engine from {vllm_config.model_id!r}: {exc}"
            ) from exc

        # Text-template rendering only -- no weights, no GPU, not "a second
        # Transformers model". Image preprocessing/token-placeholder expansion
        # is vLLM's own job once `multi_modal_data` is attached.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                vllm_config.model_id,
                cache_dir=str(vllm_config.cache_dir) if vllm_config.cache_dir else None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer from {vllm_config.model_id!r}: {exc}"
            ) from exc

        self._ready = True
        logger.info(
            "vLLM engine loaded | model_id=%s | data_parallel_size=%d | tensor_parallel_size=%d",
            vllm_config.model_id,
            data_parallel_size,
            vllm_config.tensor_parallel_size,
        )

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        if request.generation.num_beams > 1:
            raise InvalidGenerationOptionsError(
                "vLLM backend does not support num_beams > 1 (no beam search in the "
                "installed sampler)."
            )

        if self._profiler_enabled and not self._profiler_started:
            # Deferred to first request rather than __init__ since
            # start_profile() is async and __init__ isn't. Runs vLLM's own
            # schedule (wait/warmup/active_iterations from profiler_config)
            # from here for the rest of the process's lifetime.
            self._profiler_started = True
            try:
                await self._engine.start_profile()
            except Exception:
                logger.warning("Failed to start vLLM torch profiler", exc_info=True)

        start_time = time.perf_counter()
        vllm_request_id = str(uuid.uuid4())
        prompt = self._build_prompt(request)
        sampling_params = self._build_sampling_params(request.generation)

        final_output = None
        try:
            async for output in self._engine.generate(prompt, sampling_params, vllm_request_id):
                final_output = output
        except asyncio.CancelledError:
            await self._engine.abort(vllm_request_id)
            raise

        if final_output is None or not final_output.outputs:
            raise RuntimeError("vLLM engine returned no output for this request.")

        completion = final_output.outputs[0]

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=completion.text,
            model_version=self.model_version,
            input_token_count=len(final_output.prompt_token_ids or []),
            output_token_count=len(completion.token_ids),
            latency_seconds=time.perf_counter() - start_time,
        )

    async def health(self) -> bool:
        if not self._ready:
            return False

        try:
            await self._engine.check_health()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """
        Shut down the vLLM engine (and any engine-core subprocesses it owns).

        Async (unlike TransformersQwenVLEngine.close) so a running torch
        profiler can be stopped -- and its trace flushed to disk -- before
        the engine-core subprocesses holding it are torn down. Callers must
        await this; see the awaitable-close handling in
        src/inference/vllm/app.py's lifespan.
        """

        self._ready = False

        if self._profiler_started:
            try:
                await self._engine.stop_profile()
            except Exception:
                logger.warning("Failed to stop vLLM torch profiler", exc_info=True)

        try:
            self._engine.shutdown()
        except Exception:
            logger.warning("vLLM engine shutdown raised an error", exc_info=True)

    def _build_prompt(self, request: InferenceRequest) -> TextPrompt:
        messages = build_conversation_messages(request)
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        images = [image_input.image for image_input in request.images]
        if images:
            return TextPrompt(prompt=rendered, multi_modal_data={"image": images})

        return TextPrompt(prompt=rendered)

    @staticmethod
    def _build_sampling_params(generation: GenerationConfig) -> SamplingParams:
        return SamplingParams(
            max_tokens=generation.max_new_tokens,
            temperature=generation.temperature if generation.do_sample else 0.0,
            n=1,
        )