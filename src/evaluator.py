"""Benchmark loop and metric aggregation."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm

from src.data_loader import CANONICAL_LABELS, DriveAndActLoader, SegmentSample, SegmentSpec, pil_to_numpy
from src.hardware_monitor import HardwareMonitor
from src.model_engine import VLMEngine


LOGGER = logging.getLogger(__name__)

DRIVEACT_LABELS: tuple[str, ...] = ("Driving", "Texting", "Drinking", "Reaching")
SYSTEM_PROMPT = (
    "You are an automotive safety system. Analyze the driver's action in this image. "
    "Reply with ONLY ONE of the following words: [Driving, Texting, Drinking, Reaching, Asleep]."
)
DRIVEACT_PROMPT = (
    "You are an automotive safety system evaluating Drive&Act driver-cabin images. "
    "Choose the driver's current action from these labels only: [Driving, Texting, Drinking, Reaching]. "
    "Driving means the driver is seated in a normal attentive driving posture, even if the hands are relaxed or the driver is waiting. "
    "Reply with ONLY ONE word from that list."
)
# Qwen-specific: adds Reaching visual cues (arm extended away from wheel/torso)
# and places Reaching last so the model doesn't settle on Driving as a fallback.
DRIVEACT_QWEN_LABELS: tuple[str, ...] = ("Texting", "Drinking", "Driving", "Reaching")
DRIVEACT_QWEN_PROMPT = (
    "You are an automotive safety system evaluating Drive&Act driver-cabin footage. "
    "Classify the driver's action using EXACTLY ONE word from this list: [Texting, Drinking, Driving, Reaching].\n"
    "Label definitions:\n"
    "  Driving  — driver is seated with hands near or on the wheel, or in a neutral resting posture.\n"
    "  Texting  — driver is looking at or holding a phone, typing, or swiping.\n"
    "  Drinking — driver is raising a cup, bottle, or food item toward the mouth.\n"
    "  Reaching — driver's arm is clearly extended AWAY from the wheel toward the dashboard, "
    "back seat, glove box, door pocket, or a bag. The torso may be twisted sideways.\n"
    "Reply with ONLY the single word label."
)
# SmolVLM-specific: Texting moved to LAST position to break collapse bias;
# shorter, simpler instruction text for the 256M model.
DRIVEACT_SMOLVLM_LABELS: tuple[str, ...] = ("Driving", "Drinking", "Reaching", "Texting")
DRIVEACT_SMOLVLM_PROMPT = (
    "Driver-cabin image. Pick ONE label that best describes the driver's action:\n"
    "  Driving  — seated, hands on or near wheel, normal posture.\n"
    "  Drinking — arm raised, cup or bottle near mouth.\n"
    "  Reaching — arm extended away from wheel toward dash, back seat, or bag.\n"
    "  Texting  — looking at or holding a phone.\n"
    "Reply with ONE word only: Driving, Drinking, Reaching, or Texting."
)

# Model-ID substrings that map to a specific DriveAct sub-profile.
_MODEL_PROFILE_HINTS: dict[str, str] = {
    "smolvlm": "driveact_smolvlm",
    "qwen2.5-vl": "driveact_qwen",
    "qwen2-vl": "driveact_qwen",
}

PROMPT_PROFILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "default": (SYSTEM_PROMPT, CANONICAL_LABELS),
    "driveact": (DRIVEACT_PROMPT, DRIVEACT_LABELS),
    "driveact_qwen": (DRIVEACT_QWEN_PROMPT, DRIVEACT_QWEN_LABELS),
    "driveact_smolvlm": (DRIVEACT_SMOLVLM_PROMPT, DRIVEACT_SMOLVLM_LABELS),
}


def select_prompt_profile(profile: str, model_id: str) -> str:
    """Resolve 'driveact' to a model-specific sub-profile when available."""
    if profile != "driveact":
        return profile
    model_id_lower = model_id.lower()
    for hint, sub_profile in _MODEL_PROFILE_HINTS.items():
        if hint in model_id_lower:
            LOGGER.info(
                "Auto-selected prompt profile %r for model %r (matched hint %r).",
                sub_profile,
                model_id,
                hint,
            )
            return sub_profile
    return profile


@dataclass(frozen=True, slots=True)
class FramePrediction:
    """Per-frame output inside one segment."""

    frame_index: int
    prediction: str
    raw_text: str
    ttft_seconds: float | None
    tokens_per_second: float
    inference_seconds: float = 0.0
    confidence: float = 0.0
    runner_up_label: str = ""
    runner_up_confidence: float = 0.0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Per-segment benchmark result."""

    path: str
    ground_truth: str
    prediction: str
    raw_text: str
    ttft_seconds: float | None
    tokens_per_second: float
    frame_indices: tuple[int, ...]
    frame_predictions: tuple[FramePrediction, ...]
    avg_confidence: float = 0.0
    error: str | None = None


class BenchmarkEvaluator:
    """Run a VLM over a Drive&Act frame stream and aggregate metrics."""

    def __init__(
        self,
        dataset_root: str | Path,
        model_id: str,
        output_path: str | Path = "benchmark_report.json",
        image_size: tuple[int, int] = (448, 448),
        limit: int | None = None,
        prompt: str = SYSTEM_PROMPT,
        labels: Sequence[str] = CANONICAL_LABELS,
        frames_per_segment: int = 1,
        subset_mode: str = "sequential",
        random_seed: int = 7,
        confidence_threshold: float = 1.0,
        confidence_fallback: dict[str, str] | None = None,
        load_bits: int = 4,
        max_gpu_memory: str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.model_id = model_id
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.limit = limit
        self.prompt = prompt
        self.labels = tuple(labels)
        self.frames_per_segment = max(1, frames_per_segment)
        self.subset_mode = subset_mode
        self.random_seed = random_seed
        self.confidence_threshold = confidence_threshold
        self.confidence_fallback: dict[str, str] = confidence_fallback or {}
        self.load_bits = load_bits
        self.max_gpu_memory = max_gpu_memory
        self.monitor = HardwareMonitor()

    def run(self) -> dict[str, Any]:
        """Execute the benchmark and write the JSON report."""
        loader = DriveAndActLoader(
            self.dataset_root,
            image_size=self.image_size,
            labels=self.labels,
            frames_per_segment=self.frames_per_segment,
        )
        segment_specs = loader.get_segment_specs()
        selected_specs = self._select_segment_specs(segment_specs)
        if not selected_specs:
            LOGGER.warning("No labeled samples found under %s; writing empty report.", self.dataset_root)
            report = self._build_report(records=[], errors=0)
            self._write_report(report)
            return report

        engine = VLMEngine(
            model_id=self.model_id,
            monitor=self.monitor,
            labels=self.labels,
            confidence_threshold=self.confidence_threshold,
            confidence_fallback=self.confidence_fallback,
            load_bits=self.load_bits,
            max_gpu_memory=self.max_gpu_memory,
        )
        records: list[PredictionRecord] = []
        errors = 0

        self.monitor.start()
        try:
            for spec in tqdm(selected_specs, desc="Benchmarking segments", unit="segment"):
                sample = loader.load_segment(spec)
                record = self._evaluate_segment(sample, engine)
                if record.error is not None:
                    errors += 1
                records.append(record)
        finally:
            self.monitor.stop()

        report = self._build_report(records=records, errors=errors)
        self._write_report(report)
        return report

    def _evaluate_segment(self, sample: SegmentSample, engine: VLMEngine) -> PredictionRecord:
        frame_predictions: list[FramePrediction] = []
        votes: list[str] = []
        raw_text_parts: list[str] = []
        ttfts: list[float] = []
        tps_values: list[float] = []
        inference_times: list[float] = []
        confidences: list[float] = []
        errors: list[str] = []

        for frame_index, image in zip(sample.frame_indices, sample.images):
            result = engine.generate_action(pil_to_numpy(image), self.prompt)
            frame_predictions.append(
                FramePrediction(
                    frame_index=frame_index,
                    prediction=result.normalized_label,
                    raw_text=result.text,
                    ttft_seconds=result.timing.ttft_seconds,
                    tokens_per_second=result.timing.tokens_per_second,
                    inference_seconds=result.inference_seconds,
                    confidence=result.confidence,
                    runner_up_label=result.runner_up_label,
                    runner_up_confidence=result.runner_up_confidence,
                    error=result.error,
                )
            )
            votes.append(result.normalized_label)
            raw_text_parts.append(f"{frame_index}:{result.text}")
            if result.timing.ttft_seconds is not None:
                ttfts.append(result.timing.ttft_seconds)
            if result.timing.tokens_per_second > 0.0:
                tps_values.append(result.timing.tokens_per_second)
            if result.inference_seconds > 0.0:
                inference_times.append(result.inference_seconds)
            if result.confidence > 0.0:
                confidences.append(result.confidence)
            if result.error is not None:
                errors.append(result.error)

        return PredictionRecord(
            path=str(sample.path),
            ground_truth=sample.label,
            prediction=self._majority_vote(votes),
            raw_text=" | ".join(raw_text_parts),
            ttft_seconds=float(mean(ttfts)) if ttfts else None,
            tokens_per_second=float(mean(tps_values)) if tps_values else 0.0,
            frame_indices=sample.frame_indices,
            frame_predictions=tuple(frame_predictions),
            avg_confidence=float(mean(confidences)) if confidences else 0.0,
            error="; ".join(errors) if errors else None,
        )

    def _select_segment_specs(self, specs: Sequence[SegmentSpec]) -> list[SegmentSpec]:
        if self.limit is None or self.limit >= len(specs):
            return list(specs)
        if self.subset_mode == "sequential":
            return list(specs[: self.limit])
        if self.subset_mode == "random":
            rng = random.Random(self.random_seed)
            return rng.sample(list(specs), self.limit)
        if self.subset_mode == "stratified":
            return self._stratified_sample(specs, self.limit, self.random_seed)
        raise ValueError(f"Unsupported subset mode: {self.subset_mode}")

    @staticmethod
    def _stratified_sample(specs: Sequence[SegmentSpec], limit: int, seed: int) -> list[SegmentSpec]:
        rng = random.Random(seed)
        groups: dict[str, list[SegmentSpec]] = defaultdict(list)
        for spec in specs:
            groups[spec.label].append(spec)
        for group in groups.values():
            rng.shuffle(group)

        labels = sorted(groups)
        selected: list[SegmentSpec] = []
        while len(selected) < limit:
            progressed = False
            for label in labels:
                group = groups[label]
                if group and len(selected) < limit:
                    selected.append(group.pop())
                    progressed = True
            if not progressed:
                break
        return selected

    @staticmethod
    def _majority_vote(votes: Sequence[str]) -> str:
        if not votes:
            return "Unknown"
        counts = Counter(votes)
        best_count = max(counts.values())
        tied = {vote for vote, count in counts.items() if count == best_count}
        for vote in votes:
            if vote in tied:
                return vote
        return votes[0]

    def _build_report(self, records: list[PredictionRecord], errors: int) -> dict[str, Any]:
        y_true = [record.ground_truth for record in records]
        y_pred = [record.prediction for record in records]
        ttfts = [record.ttft_seconds for record in records if record.ttft_seconds is not None]
        tps_values = [record.tokens_per_second for record in records if record.tokens_per_second > 0.0]

        overall_accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
        macro_precision = (
            precision_score(
                y_true,
                y_pred,
                labels=list(self.labels),
                average="macro",
                zero_division=0,
            )
            if y_true
            else 0.0
        )

        all_confidences = [r.avg_confidence for r in records if r.avg_confidence > 0.0]
        # Collect per-frame inference times from all records.
        all_inference_times = [
            fp.inference_seconds
            for r in records
            for fp in r.frame_predictions
            if fp.inference_seconds > 0.0
        ]
        avg_inference = float(mean(all_inference_times)) if all_inference_times else None
        avg_fps = float(1.0 / avg_inference) if avg_inference else None

        return {
            "overall_accuracy": float(overall_accuracy),
            "macro_precision": float(macro_precision),
            "average_ttft_seconds": float(mean(ttfts)) if ttfts else None,
            "average_inference_seconds": avg_inference,
            "average_fps": avg_fps,
            "average_tps": float(mean(tps_values)) if tps_values else 0.0,
            "average_confidence": float(mean(all_confidences)) if all_confidences else 0.0,
            "peak_vram_mb": float(self.monitor.peak_vram_mb),
            "device_info": self.monitor.get_device_info(),
            "num_samples": len(records),
            "num_errors": errors,
            "frames_per_segment": self.frames_per_segment,
            "subset_mode": self.subset_mode,
            "random_seed": self.random_seed,
            "load_bits": self.load_bits,
            "quantization_mode": (
                f"{self.load_bits}-bit" if self.load_bits in (4, 8) else "fp16"
            ),
            "labels": list(self.labels),
            "prompt": self.prompt,
            "model_id": self.model_id,
            "dataset_root": str(self.dataset_root),
            "predictions": [asdict(record) for record in records],
        }

    def _write_report(self, report: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Wrote benchmark report to %s", self.output_path)


__all__ = [
    "BenchmarkEvaluator",
    "CANONICAL_LABELS",
    "DRIVEACT_LABELS",
    "DRIVEACT_PROMPT",
    "DRIVEACT_QWEN_LABELS",
    "DRIVEACT_QWEN_PROMPT",
    "DRIVEACT_SMOLVLM_LABELS",
    "DRIVEACT_SMOLVLM_PROMPT",
    "PROMPT_PROFILES",
    "SYSTEM_PROMPT",
    "select_prompt_profile",
]
