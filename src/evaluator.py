"""Benchmark loop and metric aggregation."""

from __future__ import annotations

import json
import logging
from itertools import chain
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm

from src.data_loader import CANONICAL_LABELS, DriveAndActLoader, pil_to_numpy
from src.hardware_monitor import HardwareMonitor
from src.model_engine import VLMEngine


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an automotive safety system. Analyze the driver's action in this image. "
    "Reply with ONLY ONE of the following words: [Driving, Texting, Drinking, Reaching, Asleep]."
)


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Per-frame benchmark result."""

    path: str
    ground_truth: str
    prediction: str
    raw_text: str
    ttft_seconds: float | None
    tokens_per_second: float
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
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.model_id = model_id
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.limit = limit
        self.prompt = prompt
        self.monitor = HardwareMonitor()

    def run(self) -> dict[str, Any]:
        """Execute the benchmark and write the JSON report."""
        loader = DriveAndActLoader(self.dataset_root, image_size=self.image_size)
        sample_iterator = iter(loader)
        try:
            first_sample = next(sample_iterator)
        except StopIteration:
            LOGGER.warning("No labeled image frames found under %s; writing empty report.", self.dataset_root)
            report = self._build_report(records=[], errors=0)
            self._write_report(report)
            return report

        engine = VLMEngine(model_id=self.model_id, monitor=self.monitor, labels=CANONICAL_LABELS)
        records: list[PredictionRecord] = []
        errors = 0

        self.monitor.start()
        try:
            iterator = enumerate(chain((first_sample,), sample_iterator))
            for index, sample in tqdm(iterator, desc="Benchmarking frames", unit="frame"):
                if self.limit is not None and index >= self.limit:
                    break

                result = engine.generate_action(pil_to_numpy(sample.image), self.prompt)
                if result.error is not None:
                    errors += 1

                records.append(
                    PredictionRecord(
                        path=str(sample.path),
                        ground_truth=sample.label,
                        prediction=result.normalized_label,
                        raw_text=result.text,
                        ttft_seconds=result.timing.ttft_seconds,
                        tokens_per_second=result.timing.tokens_per_second,
                        error=result.error,
                    )
                )
        finally:
            self.monitor.stop()

        report = self._build_report(records=records, errors=errors)
        self._write_report(report)
        return report

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
                labels=list(CANONICAL_LABELS),
                average="macro",
                zero_division=0,
            )
            if y_true
            else 0.0
        )

        return {
            "overall_accuracy": float(overall_accuracy),
            "macro_precision": float(macro_precision),
            "average_ttft_seconds": float(mean(ttfts)) if ttfts else None,
            "average_tps": float(mean(tps_values)) if tps_values else 0.0,
            "peak_vram_mb": float(self.monitor.peak_vram_mb),
            "num_samples": len(records),
            "num_errors": errors,
            "labels": list(CANONICAL_LABELS),
            "model_id": self.model_id,
            "dataset_root": str(self.dataset_root),
            "predictions": [asdict(record) for record in records],
        }

    def _write_report(self, report: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Wrote benchmark report to %s", self.output_path)


__all__ = ["BenchmarkEvaluator", "SYSTEM_PROMPT"]

