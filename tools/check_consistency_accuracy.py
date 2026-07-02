#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Compare CoT consistency judgments against a benchmark file.

Examples:
    uv run python tools/check_consistency_accuracy.py \
        --consistency-file data/results/llm_matrix/gpt.f_llm_map_graph.run_003.json \
        --benchmark-file data/benchmark/benchmark.json \
        --reliable-only

    uv run python tools/check_consistency_accuracy.py \
        --consistency-file data/results/rule/benchmark.rule_consistency.json \
        --benchmark-file data/benchmark/benchmark.json \
        --consistency-type alpasim_cot_consistency_report \
        --reliable-only
"""

import argparse
import json
from pathlib import Path

from benchmark_analysis import (
    CONSISTENCY_TYPES,
    DEFAULT_SCORE_THRESHOLD,
    BinaryConfusion,
    classification_metrics,
    cot_is_reliable,
    extract_entries,
    ground_truth_is_consistent,
    index_by_clip_id,
    judgment_for,
    load_json,
)


def _label_from_bool(is_consistent: bool) -> str:
    return "consistent" if is_consistent else "inconsistent"


def _disagreement_type(
    *,
    benchmark_is_consistent: bool,
    predicted_is_consistent: bool,
) -> str:
    if benchmark_is_consistent and not predicted_is_consistent:
        return "consistent_predicted_inconsistent"
    if not benchmark_is_consistent and predicted_is_consistent:
        return "inconsistent_predicted_consistent"
    return "none"


def compare_judgments(
    consistency_entries: list[dict],
    benchmark_map: dict[str, dict],
    consistency_type: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_parse_only: bool = False,
) -> dict:
    """Compare predicted consistency judgments against benchmark labels.

    When ``valid_parse_only`` is set, rule-based predictions whose CoT did not
    parse (label ``invalid_parse``) are skipped rather than counted, so accuracy
    reflects only clips the deterministic parser could actually judge.
    """
    confusion = BinaryConfusion()
    matches = []
    mismatches = []
    skipped = []

    for entry in consistency_entries:
        clip_id = str(entry.get("clip_id", "")).strip()
        if not clip_id:
            skipped.append({"clip_id": "", "reason": "missing clip_id"})
            continue
        if clip_id not in benchmark_map:
            skipped.append({"clip_id": clip_id, "reason": "clip_id not in benchmark"})
            continue

        try:
            prediction = judgment_for(entry, consistency_type, score_threshold)
            benchmark_is_consistent = ground_truth_is_consistent(
                benchmark_map[clip_id]
            )
        except ValueError as exc:
            skipped.append({"clip_id": clip_id, "reason": str(exc)})
            continue

        if valid_parse_only and not prediction.valid_parse:
            skipped.append(
                {"clip_id": clip_id, "reason": "invalid_parse (valid-parse-only)"}
            )
            continue

        confusion.add(
            actual_is_consistent=benchmark_is_consistent,
            predicted_is_consistent=prediction.is_consistent,
        )

        comparison = {
            "clip_id": clip_id,
            "predicted_is_consistent": prediction.is_consistent,
            "predicted_label": prediction.label,
            "prediction_score": prediction.score,
            "prediction_category": prediction.category,
            "benchmark_is_consistent": benchmark_is_consistent,
            "benchmark_label": _label_from_bool(benchmark_is_consistent),
            "disagreement_type": _disagreement_type(
                benchmark_is_consistent=benchmark_is_consistent,
                predicted_is_consistent=prediction.is_consistent,
            ),
        }

        if prediction.is_consistent == benchmark_is_consistent:
            matches.append(comparison)
        else:
            mismatches.append(comparison)

    return {
        "summary": {
            "total_matched": confusion.total,
            "correct": confusion.correct,
            "correct_consistent": confusion.true_negative,
            "correct_inconsistent": confusion.true_positive,
            "mismatches": len(mismatches),
            "skipped": len(skipped),
            "accuracy": confusion.accuracy,
            "accuracy_percent": 100 * confusion.accuracy,
        },
        "metrics": classification_metrics(confusion),
        "confusion_matrix": confusion.to_nested_dict(),
        "matches": matches,
        "mismatches": mismatches,
        "skipped": skipped,
    }


def print_results(report: dict, max_display: int = 20) -> None:
    """Print a human-readable comparison report."""
    summary = report["summary"]
    matrix = report["confusion_matrix"]
    mismatches = report["mismatches"]
    skipped = report["skipped"]

    print()
    print("=" * 80)
    print("CONSISTENCY JUDGMENT ACCURACY REPORT")
    print("=" * 80)
    print(f"Total matched entries:  {summary['total_matched']}")
    print(f"Correct predictions:    {summary['correct']}")
    print(f"  Correct consistent:   {summary['correct_consistent']}")
    print(f"  Correct inconsistent: {summary['correct_inconsistent']}")
    print(f"Mismatches:             {summary['mismatches']}")
    print(f"Skipped entries:        {summary['skipped']}")
    print(f"Accuracy:               {summary['accuracy_percent']:.2f}%")

    metrics = report.get("metrics")
    if metrics:
        inc = metrics["inconsistent"]
        con = metrics["consistent"]
        print()
        print("Classification metrics (positive class = inconsistent):")
        print(
            f"  Inconsistent:  precision={inc['precision']:.2f}  "
            f"recall={inc['recall']:.2f}  f1={inc['f1']:.2f}"
        )
        print(
            f"  Consistent:    precision={con['precision']:.2f}  "
            f"recall={con['recall']:.2f}  f1={con['f1']:.2f}"
        )
        print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.2f}")
        print(f"  Cohen's kappa:     {metrics['cohens_kappa']:.2f}")

    print()
    print("Confusion matrix:")
    print("                    Pred consistent   Pred inconsistent")
    print(
        "Actual consistent   "
        f"{matrix['actual_consistent']['predicted_consistent']:>15}   "
        f"{matrix['actual_consistent']['predicted_inconsistent']:>17}"
    )
    print(
        "Actual inconsistent "
        f"{matrix['actual_inconsistent']['predicted_consistent']:>15}   "
        f"{matrix['actual_inconsistent']['predicted_inconsistent']:>17}"
    )

    if mismatches and max_display > 0:
        print()
        print("-" * 80)
        print(f"MISMATCHES (first {min(max_display, len(mismatches))}):")
        print("-" * 80)

        for index, mismatch in enumerate(mismatches[:max_display], 1):
            print()
            print(f"{index}. {mismatch['clip_id']}")
            print(
                "   Prediction: "
                f"{mismatch['predicted_label']} "
                f"(score={mismatch['prediction_score']}, "
                f"category={mismatch['prediction_category']})"
            )
            print(f"   Benchmark:  {mismatch['benchmark_label']}")
            print(f"   Type:       {mismatch['disagreement_type']}")

        if len(mismatches) > max_display:
            print(f"\n... and {len(mismatches) - max_display} more mismatches")
    elif mismatches:
        print()
        print(f"{len(mismatches)} mismatches; use --json-output for details.")

    if skipped:
        print()
        print(f"Skipped {len(skipped)} entries; use --json-output for details.")


def build_report(
    *,
    consistency_file: Path,
    benchmark_file: Path,
    consistency_type: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    reliable_only: bool = False,
    valid_parse_only: bool = False,
) -> dict:
    consistency_entries = extract_entries(load_json(consistency_file))
    benchmark_entries = extract_entries(load_json(benchmark_file))
    if reliable_only:
        benchmark_entries = [
            entry for entry in benchmark_entries if cot_is_reliable(entry)
        ]
    benchmark_map = index_by_clip_id(benchmark_entries)

    report = compare_judgments(
        consistency_entries,
        benchmark_map,
        consistency_type,
        score_threshold,
        valid_parse_only,
    )
    report["consistency_file"] = str(consistency_file)
    report["benchmark_file"] = str(benchmark_file)
    report["consistency_type"] = consistency_type
    report["score_threshold"] = score_threshold
    report["reliable_only"] = reliable_only
    report["valid_parse_only"] = valid_parse_only
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check consistency judgment accuracy against a benchmark."
    )
    parser.add_argument(
        "--consistency-file",
        "--llm-file",
        dest="consistency_file",
        required=True,
        help="Path to a cot_consistency JSON file or aggregate report.",
    )
    parser.add_argument(
        "--benchmark-file",
        required=True,
        help="Path to benchmark JSON containing cot_action_consistency labels.",
    )
    parser.add_argument(
        "--consistency-type",
        choices=CONSISTENCY_TYPES,
        default="cot_output_alignment",
        help=(
            "How to classify predictions. cot_output_alignment uses the "
            "verdict field if present, else score > 2 as consistent; "
            "alpasim_cot_consistency(_report) uses label/result or binary "
            "score."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=(
            "LLM score cutoff for cot_output_alignment: score <= threshold is "
            "classified inconsistent (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--reliable-only",
        action="store_true",
        help="Only evaluate pairs whose benchmark entry has cot_reliable=true.",
    )
    parser.add_argument(
        "--valid-parse-only",
        action="store_true",
        help=(
            "Rule-based only: skip predictions whose CoT did not parse "
            "(label invalid_parse) instead of counting them as inconsistent."
        ),
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Maximum mismatches to display (default: 20).",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path for writing the full comparison report as JSON.",
    )
    args = parser.parse_args()

    consistency_file = Path(args.consistency_file)
    benchmark_file = Path(args.benchmark_file)
    if not consistency_file.exists():
        raise FileNotFoundError(f"Consistency file not found: {consistency_file}")
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    report = build_report(
        consistency_file=consistency_file,
        benchmark_file=benchmark_file,
        consistency_type=args.consistency_type,
        score_threshold=args.score_threshold,
        reliable_only=args.reliable_only,
        valid_parse_only=args.valid_parse_only,
    )
    print_results(report, args.max_display)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nJSON results saved to: {output_path}")


if __name__ == "__main__":
    main()
