"""Hugging Face VLM loading and low-memory generation."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - removed in newer transformers versions
    AutoModelForVision2Seq = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM
except ImportError:  # pragma: no cover - defensive fallback
    AutoModelForCausalLM = None  # type: ignore[assignment]

from src.hardware_monitor import GenerationTiming, HardwareMonitor


LOGGER = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a Hugging Face VLM cannot be loaded."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Structured result for a single VLM generation."""

    text: str
    normalized_label: str
    timing: GenerationTiming
    generated_tokens: int
    confidence: float = 0.0
    runner_up_label: str = ""
    runner_up_confidence: float = 0.0
    inference_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tokens_per_second"] = self.timing.tokens_per_second
        return data


class VLMEngine:
    """Quantized VLM inference wrapper."""

    def __init__(
        self,
        model_id: str,
        monitor: HardwareMonitor | None = None,
        labels: tuple[str, ...] = ("Driving", "Texting", "Drinking", "Reaching", "Asleep"),
        confidence_threshold: float = 1.0,
        confidence_fallback: dict[str, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.monitor = monitor or HardwareMonitor()
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        # e.g. {"Driving": "Reaching"} means: if top=Driving, conf<threshold,
        # and runner-up=Reaching, use Reaching instead.
        self.confidence_fallback: dict[str, str] = confidence_fallback or {}
        self.processor = self._load_processor(model_id)
        self.model = self._load_model(model_id)
        self.model.eval()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self._label_token_ids: dict[str, list[int]] = self._build_label_token_ids(tokenizer, labels)

    def generate_action(self, image_array: np.ndarray | Image.Image, prompt: str) -> GenerationResult:
        """Generate one short classification answer for a frame."""
        timer = self.monitor.create_timer()
        try:
            image = self._ensure_pil_image(image_array)
            inputs = self._prepare_inputs(image=image, prompt=prompt)
            input_token_count = self._input_token_count(inputs)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            timer.mark_first_token()
            generated_ids = outputs.sequences
            scores = outputs.scores  # tuple of (batch, vocab) tensors
            generated_tokens = max(0, int(generated_ids.shape[-1]) - input_token_count)
            output_ids = (
                generated_ids[:, input_token_count:]
                if generated_ids.shape[-1] > input_token_count
                else generated_ids
            )
            decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            text = self._strip_prompt_echo(decoded, prompt)

            # Primary classification: logit scores over label first-tokens.
            logit_label, confidence, runner_up, runner_up_conf = self._logit_label(
                scores, fallback=self.normalize_label(text, self.labels)
            )

            # Confidence-gated fallback: if top label is a trigger and confidence is
            # below the threshold, check whether the runner-up matches the fallback target.
            if (
                logit_label in self.confidence_fallback
                and confidence < self.confidence_threshold
                and runner_up == self.confidence_fallback[logit_label]
            ):
                LOGGER.debug(
                    "Confidence fallback: %s (%.3f) -> %s (%.3f)",
                    logit_label, confidence, runner_up, runner_up_conf,
                )
                logit_label, confidence, runner_up, runner_up_conf = (
                    runner_up, runner_up_conf, logit_label, confidence
                )

            timing = timer.finish(generated_tokens)
            self.monitor.update_peak_vram()
            return GenerationResult(
                text=text,
                normalized_label=logit_label,
                timing=timing,
                generated_tokens=generated_tokens,
                confidence=confidence,
                runner_up_label=runner_up,
                runner_up_confidence=runner_up_conf,
                inference_seconds=timing.total_seconds,
            )
        except torch.cuda.OutOfMemoryError as exc:
            self._clear_cuda_cache()
            LOGGER.exception("CUDA out of memory while generating action")
            timing = timer.finish(0)
            return GenerationResult(
                text="",
                normalized_label="Unknown",
                timing=timing,
                generated_tokens=0,
                confidence=0.0,
                error=f"CUDA OOM: {exc}",
            )
        except RuntimeError as exc:
            if self._is_cuda_oom(exc):
                self._clear_cuda_cache()
                LOGGER.exception("CUDA out of memory while generating action")
                timing = timer.finish(0)
                return GenerationResult(
                    text="",
                    normalized_label="Unknown",
                    timing=timing,
                    generated_tokens=0,
                    confidence=0.0,
                    error=f"CUDA OOM: {exc}",
                )
            raise

    @staticmethod
    def normalize_label(text: str, labels: tuple[str, ...]) -> str:
        """Map generated text to the canonical label set (fallback for logit scoring)."""
        normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
        for label in labels:
            if label.lower() in normalized.split() or label.lower() == normalized:
                return label
        return "Unknown"

    def _logit_label(
        self, scores: tuple[torch.Tensor, ...], fallback: str
    ) -> tuple[str, float, str, float]:
        """Return (best_label, best_conf, runner_up_label, runner_up_conf) from first-token logits."""
        if not scores or not self._label_token_ids:
            return fallback, 0.0, "", 0.0
        first_logits = scores[0][0]  # shape: (vocab_size,)
        label_scores: dict[str, float] = {}
        for label, token_ids in self._label_token_ids.items():
            if token_ids:
                label_scores[label] = first_logits[token_ids].max().item()
            else:
                label_scores[label] = float("-inf")
        if not label_scores:
            return fallback, 0.0, "", 0.0
        score_tensor = torch.tensor(list(label_scores.values()), dtype=torch.float32)
        probs = torch.softmax(score_tensor, dim=0)
        # Rank labels by probability descending.
        label_list = list(label_scores.keys())
        sorted_indices = probs.argsort(descending=True).tolist()
        best_idx = sorted_indices[0]
        best_label = label_list[best_idx]
        best_conf = float(probs[best_idx].item())
        runner_up_label = ""
        runner_up_conf = 0.0
        if len(sorted_indices) > 1:
            ru_idx = sorted_indices[1]
            runner_up_label = label_list[ru_idx]
            runner_up_conf = float(probs[ru_idx].item())
        LOGGER.debug(
            "Logit top=%s(%.3f) runner-up=%s(%.3f)",
            best_label, best_conf, runner_up_label, runner_up_conf,
        )
        return best_label, best_conf, runner_up_label, runner_up_conf

    @staticmethod
    def _build_label_token_ids(tokenizer: Any, labels: tuple[str, ...]) -> dict[str, list[int]]:
        """Build a cached map of label -> candidate first-token IDs for logit scoring."""
        label_token_ids: dict[str, list[int]] = {}
        for label in labels:
            seen: set[int] = set()
            candidates: list[int] = []
            # Try the label with common tokenizer prefixes (space-prefixed for BPE/SentencePiece).
            for variant in (label, " " + label, label.lower(), " " + label.lower()):
                try:
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                    if ids and ids[0] not in seen:
                        seen.add(ids[0])
                        candidates.append(ids[0])
                except Exception:  # noqa: BLE001
                    pass
            label_token_ids[label] = candidates
            LOGGER.debug("Label %r -> token IDs %s", label, candidates)
        return label_token_ids

    def _load_model(self, model_id: str) -> torch.nn.Module:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        model_classes = (
            ("AutoModelForImageTextToText", AutoModelForImageTextToText),
            ("AutoModelForVision2Seq", AutoModelForVision2Seq),
            ("AutoModelForCausalLM", AutoModelForCausalLM),
        )
        last_error: Exception | None = None
        for class_name, model_class in model_classes:
            if model_class is None:
                continue
            try:
                return model_class.from_pretrained(model_id, **model_kwargs)
            except (ValueError, OSError) as exc:
                last_error = exc
                LOGGER.debug("%s failed for %s: %s", class_name, model_id, exc)

        raise ModelLoadError(f"No compatible transformers auto-model class could load {model_id}") from last_error

    @staticmethod
    def _load_processor(model_id: str) -> Any:
        try:
            return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except OSError as exc:
            raise ModelLoadError(
                "Could not load the Hugging Face processor for "
                f"'{model_id}'. Check that the model ID is valid and public, or run "
                "`hf auth login` if it is private/gated. For SmolVLM, use "
                "'HuggingFaceTB/SmolVLM-256M-Instruct'."
            ) from exc

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        return self._move_inputs_to_model_device(inputs)

    def _move_inputs_to_model_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        device = self._model_device()
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _input_token_count(inputs: dict[str, Any]) -> int:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return 0
        return int(input_ids.shape[-1])

    @staticmethod
    def _ensure_pil_image(image_array: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image_array, Image.Image):
            return image_array.convert("RGB")
        return Image.fromarray(image_array.astype("uint8"), mode="RGB")

    @staticmethod
    def _strip_prompt_echo(decoded: str, prompt: str) -> str:
        text = decoded.strip()
        if prompt in text:
            text = text.split(prompt, maxsplit=1)[-1]
        return text.strip()

    @staticmethod
    def _clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "cuda" in message and "out of memory" in message


__all__ = ["GenerationResult", "ModelLoadError", "VLMEngine"]

