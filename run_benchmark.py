"""CLI entry point for offline Drive&Act VLM benchmarking."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.evaluator import BenchmarkEvaluator, SYSTEM_PROMPT
from src.model_engine import ModelLoadError


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark lightweight VLMs on Drive&Act frames.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"), help="Directory of image frames.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_report.json"), help="JSON report path.")
    parser.add_argument("--image-size", type=int, default=448, help="Square image resize dimension.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of frames.")
    parser.add_argument("--prompt", default=SYSTEM_PROMPT, help="Classifier prompt sent to the VLM.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    evaluator = BenchmarkEvaluator(
        dataset_root=args.dataset_root,
        model_id=args.model_id,
        output_path=args.output,
        image_size=(args.image_size, args.image_size),
        limit=args.limit,
        prompt=args.prompt,
    )
    try:
        report = evaluator.run()
    except ModelLoadError as exc:
        raise SystemExit(f"Model load failed: {exc}") from exc

    print(f"Accuracy: {report['overall_accuracy']:.4f}")
    print(f"Macro precision: {report['macro_precision']:.4f}")
    print(f"Average TTFT: {report['average_ttft_seconds']}")
    print(f"Average TPS: {report['average_tps']:.4f}")
    print(f"Peak VRAM MB: {report['peak_vram_mb']:.2f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

