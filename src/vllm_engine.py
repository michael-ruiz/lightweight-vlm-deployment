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
    ) -> None:
        self.model_id = model_id
        self.monitor = monitor or HardwareMonitor()
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.confidence_fallback = confidence_fallback or {}
        self.gpu_memory_utilization = vllm_gpu_memory_utilization

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

        # Initialize the vLLM engine with PagedAttention
        # enforce max_model_len to fit context in 8GB Jetson
        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=4096,  # Cap context to save KV cache VRAM
            limit_mm_per_prompt={"image": 1, "video": 0},  # We only pass single images; disable video profiling to avoid OOM on Jetson
            enforce_eager=True,  # Disables CUDA graphs (avoids contiguous VRAM alloc on Jetson)
            dtype="half",  # Force FP16; BF16 default requires more contiguous memory headroom
            max_num_batched_tokens=512,  # Limit batch size: 512/144tokens_per_image = ~3 images in profiling run (vs default 8192/144 = 57)
            max_num_seqs=1,  # Single sequence at a time for Jetson memory budget
            mm_processor_kwargs={
                # Limit image resolution for the vLLM profiling run.
                # Default is max_pixels=~16384 tokens (~4096x4096px) which exhausts
                # Jetson VRAM during the vision encoder dummy forward pass.
                # 336x336 gives ~196 tokens — sufficient for dashcam frames.
                "max_pixels": 336 * 336,
                "min_pixels": 28 * 28,
            },
        )
        
        # Load the tokenizer from the LLM for token manipulation
        tokenizer = self.llm.get_tokenizer()
        self._label_token_ids = VLMEngine._build_label_token_ids(tokenizer, labels)
        LOGGER.info("vLLM engine loaded successfully.")

    def generate_action(self, image_array: np.ndarray | Image.Image, prompt: str) -> GenerationResult:
        """Run multimodal inference using vLLM."""
        start_time = time.perf_counter()

        if isinstance(image_array, np.ndarray):
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = image_array

        from vllm import SamplingParams

        # Request logprobs for our labels to do confidence gating
        sampling_params = SamplingParams(
            max_tokens=5,
            temperature=0.0,
            logprobs=20,  # Grab top 20 logprobs to find our labels
        )

        # For vLLM 0.6.0+ multimodal dict structure
        # The prompt for Qwen2.5-VL natively via vLLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
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
                timing=GenerationTiming(total_seconds=0.0, first_token_seconds=0.0, output_tokens=0),
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
            first_token_seconds=ttft_seconds,
            output_tokens=generated_tokens,
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
