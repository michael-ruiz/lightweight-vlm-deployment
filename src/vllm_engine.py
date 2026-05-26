"""vLLM inference engine for Jetson deployment."""

import logging
import os
import time
from typing import Any

# Must be set BEFORE importing vLLM or instantiating LLM() so that the
# spawned EngineCore subprocess inherits these values. Shell env vars are
# NOT inherited across Python multiprocessing 'spawn' boundaries.
# NOTE: Do NOT use expandable_segments=True on Jetson — it enables CUDA VMM
# (cuMemCreate/cuMemMap) which conflicts with Jetson NvMap and causes
# NVML_SUCCESS assertion failures on every allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
# Force vLLM V0 engine: V1's profiler runs a full 512-token LLM dummy forward
# pass to measure peak memory, which exhausts all remaining VRAM on Jetson,
# leaving no room for KV cache blocks and raising ValueError.
# V0 uses a simpler heuristic-based approach that works within Jetson's budget.
os.environ.setdefault("VLLM_USE_V1", "0")

import numpy as np
from PIL import Image

from src.hardware_monitor import GenerationTiming, HardwareMonitor
from src.model_engine import GenerationResult, VLMEngine

LOGGER = logging.getLogger(__name__)


class VLLMEngine:
    """vLLM wrapper matching the VLMEngine interface."""

    def __init__(
        self,
        model_id: str,
        monitor: HardwareMonitor | None = None,
        labels: tuple[str, ...] = ("Driving", "Texting", "Drinking", "Reaching", "Asleep"),
        confidence_threshold: float = 1.0,
        confidence_fallback: dict[str, str] | None = None,
        vllm_gpu_memory_utilization: float = 0.9,
        multi_frame_mode: bool = False,
    ) -> None:
        self.model_id = model_id
        self.monitor = monitor or HardwareMonitor()
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.confidence_fallback = confidence_fallback or {}
        self.gpu_memory_utilization = vllm_gpu_memory_utilization
        self.multi_frame_mode = multi_frame_mode

        LOGGER.info(
            "Loading vLLM engine for %s (gpu_utilization=%.2f)",
            model_id,
            self.gpu_memory_utilization,
        )
        try:
            from vllm import LLM
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. VLLMEngine requires the "
                "vllm package to be installed in the environment."
            ) from e

        # Token budget: single-frame uses 361 img + ~100 text = ~461 tokens (fits in 512).
        # Multi-frame single-call: 3 × 361 img + ~150 text = ~1233 tokens → use 1536 budget.
        token_budget = 1536 if multi_frame_mode else 512
        # KV cache blocks must satisfy: num_blocks × block_size(16) >= max_model_len
        # Single-frame: 64 × 16 = 1024 ≥ 512 ✓  (1 MB/layer per block)
        # Multi-frame:  128 × 16 = 2048 ≥ 1536 ✓ (2 MB/layer per block — still fine for NvMap)
        kv_blocks = 128 if multi_frame_mode else 64

        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=token_budget,
            limit_mm_per_prompt={"image": 3 if multi_frame_mode else 1, "video": 0},
            enforce_eager=True,
            dtype="half",
            max_num_batched_tokens=token_budget,
            max_num_seqs=1,
            num_gpu_blocks_override=kv_blocks,
            swap_space=0,
            mm_processor_kwargs={
                "max_pixels": 532 * 532,  # 38×38 patches / 4 = 361 tokens; +~100 text = 461 < 512 single-frame budget
                "min_pixels": 28 * 28,
            },
        )
        
        # Load the tokenizer from the LLM for token manipulation
        tokenizer = self.llm.get_tokenizer()
        self._label_token_ids = VLMEngine._build_label_token_ids(tokenizer, labels)
        LOGGER.info("vLLM engine loaded successfully.")

    def generate_action(self, image_array: np.ndarray | Image.Image, prompt: str) -> GenerationResult:
        """Run multimodal inference using vLLM."""
        import base64
        import io
        start_time = time.perf_counter()

        if isinstance(image_array, np.ndarray):
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = image_array

        # Encode image as base64 JPEG for vLLM chat API
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"

        from vllm import SamplingParams

        # Request logprobs for our labels to do confidence gating
        sampling_params = SamplingParams(
            max_tokens=5,
            temperature=0.0,
            logprobs=20,  # Grab top 20 logprobs to find our labels
        )

        # For vLLM multimodal chat: use image_url with base64-encoded JPEG
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        try:
            outputs = self.llm.chat(
                messages,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except Exception as e:
            return GenerationResult(
                text="",
                normalized_label="Unknown",
                timing=GenerationTiming(total_seconds=0.0, ttft_seconds=None, generated_tokens=0),
                generated_tokens=0,
                confidence=0.0,
                runner_up_label="Unknown",
                runner_up_confidence=0.0,
                inference_seconds=0.0,
                error=f"vLLM inference failed: {e}",
            )

        end_time = time.perf_counter()
        inference_seconds = end_time - start_time

        output = outputs[0]
        text = output.outputs[0].text.strip()
        generated_tokens = len(output.outputs[0].token_ids)
        
        # Parse metrics if available from vLLM's internal metrics
        first_token_time = 0.0
        ttft_seconds = 0.0
        if hasattr(output.metrics, "first_token_time") and output.metrics.first_token_time:
            first_token_time = output.metrics.first_token_time
            arrival_time = output.metrics.arrival_time if hasattr(output.metrics, "arrival_time") else start_time
            ttft_seconds = first_token_time - arrival_time
        
        # Logprob extraction for confidence
        # vLLM returns a list of dictionaries mapping token_id -> Logprob object
        best_label = VLMEngine.normalize_label(text, self.labels)
        best_conf = 0.0
        runner_up_label = "Unknown"
        runner_up_conf = 0.0
        
        logprobs_list = output.outputs[0].logprobs
        if logprobs_list and len(logprobs_list) > 0:
            first_token_logprobs = logprobs_list[0]
            # Convert dict of token_id -> Logprob to standard format
            # We want to match our labels to their tokens
            label_probs = {}
            for label, tokens in self._label_token_ids.items():
                prob_sum = 0.0
                for token in tokens:
                    if token in first_token_logprobs:
                        # Convert logprob to linear probability
                        prob = np.exp(first_token_logprobs[token].logprob)
                        prob_sum += prob
                label_probs[label] = prob_sum
                
            # Sort labels by probability
            sorted_labels = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
            if sorted_labels:
                best_conf = sorted_labels[0][1]
                if len(sorted_labels) > 1:
                    runner_up_label = sorted_labels[1][0]
                    runner_up_conf = sorted_labels[1][1]

        # Confidence fallback logic identical to VLMEngine
        if (
            best_label in self.confidence_fallback
            and best_conf < self.confidence_threshold
            and runner_up_label == self.confidence_fallback[best_label]
        ):
            LOGGER.debug(
                "Confidence fallback: %s (%.3f) -> %s (%.3f)",
                best_label, best_conf, runner_up_label, runner_up_conf,
            )
            best_label, best_conf, runner_up_label, runner_up_conf = (
                runner_up_label, runner_up_conf, best_label, best_conf
            )

        timing = GenerationTiming(
            total_seconds=inference_seconds,
            ttft_seconds=ttft_seconds if ttft_seconds else None,
            generated_tokens=generated_tokens,
        )

        return GenerationResult(
            text=text,
            normalized_label=best_label,
            timing=timing,
            generated_tokens=generated_tokens,
            confidence=best_conf,
            runner_up_label=runner_up_label,
            runner_up_confidence=runner_up_conf,
            inference_seconds=inference_seconds,
            error=None,
        )

    def generate_multi_frame_action(
        self,
        images: list,
        prompt: str,
    ) -> GenerationResult:
        """Run a single inference call with all frames in one message.

        Instead of majority-voting N separate single-frame predictions, the model
        sees all N frames simultaneously and reasons holistically across them.
        Requires the engine to have been initialised with multi_frame_mode=True
        (token budget of 1536 to fit 3×361 image tokens + text).
        """
        import base64
        import io

        start_time = time.perf_counter()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=5,
            temperature=0.0,
            logprobs=20,
        )

        # Build content list: one image_url per frame, then the text prompt
        content: list[dict] = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_image = Image.fromarray(img)
            else:
                pil_image = img
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        # Append the instruction — ask the model to reason across all frames together
        label_list = ", ".join(self.labels)
        content.append({
            "type": "text",
            "text": (
                f"{prompt}\n\n"
                f"The {len(images)} images above are consecutive frames from the same video segment. "
                f"Consider all frames together and reply with ONE word from [{label_list}]."
            ),
        })

        messages = [{"role": "user", "content": content}]

        try:
            outputs = self.llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        except Exception as e:
            return GenerationResult(
                text="",
                normalized_label="Unknown",
                timing=GenerationTiming(total_seconds=0.0, ttft_seconds=None, generated_tokens=0),
                generated_tokens=0,
                confidence=0.0,
                runner_up_label="Unknown",
                runner_up_confidence=0.0,
                inference_seconds=0.0,
                error=f"vLLM multi-frame inference failed: {e}",
            )

        end_time = time.perf_counter()
        inference_seconds = end_time - start_time

        output = outputs[0]
        text = output.outputs[0].text.strip()
        generated_tokens = len(output.outputs[0].token_ids)

        ttft_seconds = None
        if hasattr(output.metrics, "first_token_time") and output.metrics.first_token_time:
            arrival_time = getattr(output.metrics, "arrival_time", start_time)
            ttft_seconds = output.metrics.first_token_time - arrival_time

        best_label = VLMEngine.normalize_label(text, self.labels)
        best_conf = 0.0
        runner_up_label = "Unknown"
        runner_up_conf = 0.0

        logprobs_list = output.outputs[0].logprobs
        if logprobs_list:
            first_token_logprobs = logprobs_list[0]
            label_probs = {}
            for label, tokens in self._label_token_ids.items():
                prob_sum = sum(
                    np.exp(first_token_logprobs[t].logprob)
                    for t in tokens
                    if t in first_token_logprobs
                )
                label_probs[label] = prob_sum
            sorted_labels = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
            if sorted_labels:
                best_conf = sorted_labels[0][1]
                if len(sorted_labels) > 1:
                    runner_up_label = sorted_labels[1][0]
                    runner_up_conf = sorted_labels[1][1]

        timing = GenerationTiming(
            total_seconds=inference_seconds,
            ttft_seconds=ttft_seconds,
            generated_tokens=generated_tokens,
        )

        return GenerationResult(
            text=text,
            normalized_label=best_label,
            timing=timing,
            generated_tokens=generated_tokens,
            confidence=best_conf,
            runner_up_label=runner_up_label,
            runner_up_confidence=runner_up_conf,
            inference_seconds=inference_seconds,
            error=None,
        )
