"""CLI entry point for offline Drive&Act VLM benchmarking."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.evaluator import BenchmarkEvaluator, CANONICAL_LABELS, PROMPT_PROFILES, SYSTEM_PROMPT, select_prompt_profile
from src.model_engine import ModelLoadError


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark lightweight VLMs on Drive&Act frames.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"), help="Directory of image frames.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_report.json"), help="JSON report path.")
    parser.add_argument("--image-size", type=int, default=448, help="Square image resize dimension.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of segments.")
    parser.add_argument("--frames-per-segment", type=int, default=3, choices=(1, 3, 5), help="Frames sampled per labeled segment.")
    parser.add_argument("--subset-mode", default="sequential", choices=("sequential", "random", "stratified"), help="Subset selection mode when --limit is set.")
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for random/stratified subset selection.")
    parser.add_argument(
        "--prompt-profile",
        default="driveact",
        choices=tuple(PROMPT_PROFILES) + ("auto",),
        help=(
            "Prompt and label-candidate preset. 'auto' (or 'driveact') automatically selects "
            "the best sub-profile for the given model ID (driveact_qwen / driveact_smolvlm). "
            "Explicit choices: " + ", ".join(PROMPT_PROFILES) + "."
        ),
    )
    parser.add_argument("--prompt", default=None, help="Override the classifier prompt.")
    parser.add_argument("--candidate-labels", nargs='+', default=None, help="Override candidate labels used for normalization and metrics.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    profile_name = args.prompt_profile
    # 'auto' is a user-friendly alias for 'driveact' with auto-selection enabled.
    if profile_name == "auto":
        profile_name = "driveact"
    # Auto-select model-specific sub-profile when profile is 'driveact'.
    profile_name = select_prompt_profile(profile_name, args.model_id)
    profile_prompt, profile_labels = PROMPT_PROFILES[profile_name]
    prompt = args.prompt if args.prompt is not None else profile_prompt
    labels = tuple(args.candidate_labels) if args.candidate_labels is not None else tuple(profile_labels)
    if not labels:
        labels = CANONICAL_LABELS
    evaluator = BenchmarkEvaluator(
        dataset_root=args.dataset_root,
        model_id=args.model_id,
        output_path=args.output,
        image_size=(args.image_size, args.image_size),
        limit=args.limit,
        prompt=prompt,
        labels=labels,
        frames_per_segment=args.frames_per_segment,
        subset_mode=args.subset_mode,
        random_seed=args.random_seed,
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
