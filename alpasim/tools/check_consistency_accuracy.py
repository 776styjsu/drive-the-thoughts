#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Compare CoT consistency judgments against a benchmark file.

Examples:
    python tools/check_consistency_accuracy.py \
        --consistency-file benchmark_expanded_50.cot_consistency.json \
        --benchmark-file benchmark_expanded_50.json \
        --consistency-type cot_output_alignment

    python tools/check_consistency_accuracy.py \
        --consistency-file cot_consistency_report.json \
        --benchmark-file benchmark_expanded_50.json \
        --consistency-type alpasim_cot_consistency_report
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


COT_OUTPUT_ALIGNMENT = "cot_output_alignment"
ALPASIM_COT_CONSISTENCY = "alpasim_cot_consistency"
ALPASIM_COT_CONSISTENCY_REPORT = "alpasim_cot_consistency_report"
RULE_BASED_ALIASES = {
    ALPASIM_COT_CONSISTENCY,
    ALPASIM_COT_CONSISTENCY_REPORT,
    "rule_based",
}
SUPPORTED_CONSISTENCY_TYPES = (
    COT_OUTPUT_ALIGNMENT,
    ALPASIM_COT_CONSISTENCY,
    ALPASIM_COT_CONSISTENCY_REPORT,
    "rule_based",
)

# Rule-based label emitted when the CoT yields no parseable ego intent. These
# are parse failures rather than genuine consistency judgments, so they can be
# excluded from accuracy via --valid-parse-only.
INVALID_PARSE_LABEL = "invalid_parse"

CONSISTENT_STRINGS = {
    "consistent",
    "true",
    "pass",
    "passed",
    "match",
    "matched",
    "aligned",
    "yes",
}
INCONSISTENT_STRINGS = {
    "inconsistent",
    "contradictory",
    "invalid_parse",
    "false",
    "fail",
    "failed",
    "mismatch",
    "mismatched",
    "not_consistent",
    "not consistent",
    "partial",
    "partial_consistency",
    "partially_consistent",
    "partially consistent",
    "unaligned",
    "no",
}


def load_json_file(file_path: Union[str, Path]) -> Union[dict, list]:
    """Load a JSON file and return its contents."""
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_entries(data: Union[dict, list]) -> List[dict]:
    """Extract per-clip entries from common consistency or benchmark shapes."""
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object or list")

    if data.get("clip_id"):
        return [data]

    for key in ("results", "entries"):
        entries = data.get(key)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]

    raise ValueError("Unrecognized JSON format; expected clip_id, results, or entries")


def filter_empty_benchmark_entries(benchmark_entries: List[dict]) -> List[dict]:
    """Remove benchmark entries without a usable clip_id."""
    return [
        entry
        for entry in benchmark_entries
        if str(entry.get("clip_id", "")).strip()
    ]


def cot_reliability_flag(item: Optional[dict]) -> Optional[bool]:
    """CoT reliability from a benchmark entry, supporting both on-disk schemas.

    - flat ``cot_reliable`` (bool/str), used by benchmark_expanded_*.json
    - nested ``cot_reliability.reliable`` (bool), used by benchmark.json

    Returns True/False when a signal is present, else None (caller's default).
    """
    if not isinstance(item, dict):
        return None
    nested = item.get("cot_reliability")
    value = (
        nested.get("reliable")
        if isinstance(nested, dict) and "reliable" in nested
        else item.get("cot_reliable")
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "reliable", "yes", "1"}:
            return True
        if text in {"false", "unreliable", "no", "0"}:
            return False
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_from_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in CONSISTENT_STRINGS:
            return True
        if normalized in INCONSISTENT_STRINGS:
            return False
    return None


def _label_from_bool(is_consistent: bool) -> str:
    return "consistent" if is_consistent else "inconsistent"


def _benchmark_is_consistent(entry: dict) -> bool:
    """Read the benchmark ground-truth consistency label."""
    for key in ("cot_action_consistency", "is_consistent", "consistent"):
        if key in entry:
            parsed = _bool_from_value(entry[key])
            if parsed is not None:
                return parsed

    for key in ("label", "result"):
        if key in entry:
            parsed = _bool_from_value(entry[key])
            if parsed is not None:
                return parsed

    raise ValueError(f"Benchmark entry has no consistency label: {entry.get('clip_id')}")


def _extract_cot_output_alignment(entry: dict, score_threshold: float = 2) -> dict:
    alignment = entry.get(COT_OUTPUT_ALIGNMENT)
    if not isinstance(alignment, dict):
        alignment = entry.get("evaluation", {}).get(COT_OUTPUT_ALIGNMENT, {})
    if not isinstance(alignment, dict):
        alignment = {}

    justification = alignment.get("justification", entry.get("justification", ""))

    # Verdict schema (detector prompt): {"verdict": "consistent" | "inconsistent"}.
    is_consistent = _bool_from_value(alignment.get("verdict"))
    if is_consistent is not None:
        return {
            "is_consistent": is_consistent,
            "label": _label_from_bool(is_consistent),
            "score": _to_float(alignment.get("score")),
            "category": alignment.get("inconsistency_type")
            or alignment.get("category", entry.get("category", "")),
            "justification": justification,
            "valid_parse": True,
        }

    # Graded schema: {"score": 1-5}, consistent when score > threshold.
    score = _to_float(alignment.get("score"))
    if score is None:
        score = _to_float(entry.get("score"))
    if score is None:
        raise ValueError(
            f"Missing cot_output_alignment verdict/score for "
            f"clip_id={entry.get('clip_id')}"
        )

    is_consistent = score > score_threshold
    return {
        "is_consistent": is_consistent,
        "label": _label_from_bool(is_consistent),
        "score": score,
        "category": alignment.get("inconsistency_type")
        or alignment.get("category", entry.get("category", "")),
        "justification": justification,
        "valid_parse": True,
    }


def _extract_rule_based_consistency(entry: dict) -> dict:
    report = entry.get("report") if isinstance(entry.get("report"), dict) else entry

    label = report.get("label")
    result = report.get("result")
    score = _to_float(report.get("score"))

    is_consistent = _bool_from_value(label)
    if is_consistent is None:
        is_consistent = _bool_from_value(result)
    if is_consistent is None and score is not None:
        is_consistent = score > 0
    if is_consistent is None:
        raise ValueError(
            f"Missing rule-based result/label/score for clip_id={entry.get('clip_id')}"
        )

    raw_label = str(label or result or "").strip().lower()
    return {
        "is_consistent": is_consistent,
        "label": label or result or _label_from_bool(is_consistent),
        "score": score,
        "category": label or result or "",
        "justification": "",
        "valid_parse": raw_label != INVALID_PARSE_LABEL,
    }


def get_consistency_judgment(
    entry: dict, consistency_type: str, score_threshold: float = 2
) -> dict:
    """Extract a normalized predicted consistency judgment from one entry."""
    if consistency_type == COT_OUTPUT_ALIGNMENT:
        return _extract_cot_output_alignment(entry, score_threshold)
    if consistency_type in RULE_BASED_ALIASES:
        return _extract_rule_based_consistency(entry)
    raise ValueError(f"Unsupported consistency type: {consistency_type}")


def _empty_confusion_matrix() -> dict:
    return {
        "actual_consistent": {
            "predicted_consistent": 0,
            "predicted_inconsistent": 0,
        },
        "actual_inconsistent": {
            "predicted_consistent": 0,
            "predicted_inconsistent": 0,
        },
    }


def _add_to_confusion_matrix(
    confusion_matrix: dict,
    *,
    benchmark_is_consistent: bool,
    predicted_is_consistent: bool,
) -> None:
    actual_key = (
        "actual_consistent"
        if benchmark_is_consistent
        else "actual_inconsistent"
    )
    predicted_key = (
        "predicted_consistent"
        if predicted_is_consistent
        else "predicted_inconsistent"
    )
    confusion_matrix[actual_key][predicted_key] += 1


def _classification_metrics(confusion_matrix: dict) -> dict:
    """Compute precision/recall/F1, balanced accuracy, and Cohen's kappa.

    "inconsistent" is treated as the positive class (the detection target);
    consistent-class precision/recall/F1 are also reported.
    """
    tp = confusion_matrix["actual_inconsistent"]["predicted_inconsistent"]
    fp = confusion_matrix["actual_consistent"]["predicted_inconsistent"]
    tn = confusion_matrix["actual_consistent"]["predicted_consistent"]
    fn = confusion_matrix["actual_inconsistent"]["predicted_consistent"]
    total = tp + fp + tn + fn

    def _safe_div(num: float, den: float) -> float:
        return num / den if den else 0.0

    def _prf(tp_: int, fp_: int, fn_: int) -> dict:
        precision = _safe_div(tp_, tp_ + fp_)
        recall = _safe_div(tp_, tp_ + fn_)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    recall_inconsistent = _safe_div(tp, tp + fn)
    recall_consistent = _safe_div(tn, tn + fp)
    balanced_accuracy = (recall_inconsistent + recall_consistent) / 2

    observed_agreement = _safe_div(tp + tn, total)
    expected_agreement = _safe_div(
        (tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), total * total
    )
    kappa = _safe_div(observed_agreement - expected_agreement, 1 - expected_agreement)

    return {
        "positive_class": "inconsistent",
        "inconsistent": _prf(tp, fp, fn),
        "consistent": _prf(tn, fn, fp),
        "balanced_accuracy": balanced_accuracy,
        "cohens_kappa": kappa,
    }


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
    consistency_entries: List[dict],
    benchmark_map: Dict[str, dict],
    consistency_type: str,
    score_threshold: float = 2,
    valid_parse_only: bool = False,
) -> dict:
    """Compare predicted consistency judgments against benchmark labels.

    When ``valid_parse_only`` is set, rule-based predictions whose CoT did not
    parse (label ``invalid_parse``) are skipped rather than counted, so accuracy
    reflects only clips the deterministic parser could actually judge.
    """
    confusion_matrix = _empty_confusion_matrix()
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
            prediction = get_consistency_judgment(
                entry, consistency_type, score_threshold
            )
            benchmark_is_consistent = _benchmark_is_consistent(benchmark_map[clip_id])
        except ValueError as exc:
            skipped.append({"clip_id": clip_id, "reason": str(exc)})
            continue

        if valid_parse_only and not prediction.get("valid_parse", True):
            skipped.append(
                {"clip_id": clip_id, "reason": "invalid_parse (valid-parse-only)"}
            )
            continue

        predicted_is_consistent = prediction["is_consistent"]
        _add_to_confusion_matrix(
            confusion_matrix,
            benchmark_is_consistent=benchmark_is_consistent,
            predicted_is_consistent=predicted_is_consistent,
        )

        comparison = {
            "clip_id": clip_id,
            "predicted_is_consistent": predicted_is_consistent,
            "predicted_label": prediction["label"],
            "prediction_score": prediction["score"],
            "prediction_category": prediction["category"],
            "benchmark_is_consistent": benchmark_is_consistent,
            "benchmark_label": _label_from_bool(benchmark_is_consistent),
            "disagreement_type": _disagreement_type(
                benchmark_is_consistent=benchmark_is_consistent,
                predicted_is_consistent=predicted_is_consistent,
            ),
        }

        if predicted_is_consistent == benchmark_is_consistent:
            matches.append(comparison)
        else:
            mismatches.append(comparison)

    total = len(matches) + len(mismatches)
    correct_consistent = confusion_matrix["actual_consistent"]["predicted_consistent"]
    correct_inconsistent = confusion_matrix["actual_inconsistent"][
        "predicted_inconsistent"
    ]
    accuracy = (correct_consistent + correct_inconsistent) / total if total else 0.0

    return {
        "summary": {
            "total_matched": total,
            "correct": correct_consistent + correct_inconsistent,
            "correct_consistent": correct_consistent,
            "correct_inconsistent": correct_inconsistent,
            "mismatches": len(mismatches),
            "skipped": len(skipped),
            "accuracy": accuracy,
            "accuracy_percent": 100 * accuracy,
        },
        "metrics": _classification_metrics(confusion_matrix),
        "confusion_matrix": confusion_matrix,
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
    score_threshold: float = 2,
    reliable_only: bool = False,
    valid_parse_only: bool = False,
) -> dict:
    consistency_data = load_json_file(consistency_file)
    benchmark_data = load_json_file(benchmark_file)

    consistency_entries = extract_entries(consistency_data)
    benchmark_entries = filter_empty_benchmark_entries(extract_entries(benchmark_data))
    if reliable_only:
        benchmark_entries = [
            entry
            for entry in benchmark_entries
            if cot_reliability_flag(entry) is True
        ]
    benchmark_map = {entry["clip_id"]: entry for entry in benchmark_entries}

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
        choices=SUPPORTED_CONSISTENCY_TYPES,
        default=COT_OUTPUT_ALIGNMENT,
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
        default=2,
        help=(
            "LLM score cutoff for cot_output_alignment: score <= threshold is "
            "classified inconsistent (default: 2)."
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
