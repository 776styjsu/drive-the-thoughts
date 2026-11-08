#!/usr/bin/env python3
"""Compare causal voting and weighted-decay policies on labeled target frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark_analysis import (
    CONSISTENCY_TYPES,
    DEFAULT_SCORE_THRESHOLD,
    BinaryConfusion,
    classification_metrics,
    extract_entries,
    judgment_for,
    load_json,
)


POLICIES = {
    "single_frame": (1, 1),
    "two_consecutive": (2, 2),
    "three_consecutive": (3, 3),
    "two_of_three": (2, 3),
    "three_of_five": (3, 5),
    "weighted_decay": (None, 5),
}


def weighted_decay_score(flags: list[bool], half_life_frames: float) -> float:
    """Fraction of flagged evidence, with each older frame exponentially downweighted."""
    weights = [2 ** (-age / half_life_frames) for age in range(len(flags) - 1, -1, -1)]
    return sum(flag * weight for flag, weight in zip(flags, weights)) / sum(weights)


def paired_intervals(rows: list[dict], repeats: int, seed: int) -> dict:
    """Percentile CIs for metric differences; resample whole scenes jointly."""
    scenes = sorted({row["scene_id"] for row in rows})
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    counts = np.zeros((len(scenes), len(POLICIES), 4), dtype=int)
    for row in rows:
        for p, name in enumerate(POLICIES):
            actual = not row["is_consistent"]
            predicted = row["alarms"][name]
            # TP, FP, TN, FN
            cell = (0 if predicted else 3) if actual else (1 if predicted else 2)
            counts[scene_index[row["scene_id"]], p, cell] += 1
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        sample = counts[rng.integers(len(scenes), size=len(scenes))].sum(axis=0)
        tp, fp, tn, fn = sample.T
        accuracy = (tp + tn) / sample.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
        metrics = np.stack((accuracy, balanced), axis=1)
        deltas.append(metrics - metrics[0])
    samples = np.asarray(deltas)
    result = {}
    for p, name in enumerate(POLICIES):
        result[name] = {}
        for m, metric in enumerate(("accuracy", "balanced_accuracy")):
            values = samples[:, p, m]
            values = values[np.isfinite(values)]
            result[name][metric] = {
                "ci95": np.quantile(values, [0.025, 0.975]).tolist()
                if len(values)
                else None,
                "n_bootstrap_valid": len(values),
            }
    return result


def evaluate(
    manifest: dict,
    predictions: dict | list,
    consistency_type: str,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    decay_half_life_frames: float = 2.0,
    decay_threshold: float = 0.5,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    if (
        manifest["kind"] != "temporal_consistency_windows"
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported temporal window manifest")
    if bootstrap < 1 or not np.isfinite(score_threshold):
        raise ValueError("Bootstrap count must be positive and score threshold finite")
    if not np.isfinite(decay_half_life_frames) or decay_half_life_frames <= 0:
        raise ValueError("Decay half-life must be positive and finite")
    if not np.isfinite(decay_threshold) or not 0 < decay_threshold <= 1:
        raise ValueError("Decay threshold must be greater than 0 and at most 1")
    judgments, errors, timestamps, seen = {}, {}, {}, set()
    for entry in extract_entries(predictions):
        clip_id = entry["clip_id"]
        if clip_id in seen:
            raise ValueError(
                f"Duplicate prediction clip_id: {clip_id}; evaluate repeats separately"
            )
        seen.add(clip_id)
        try:
            evaluation = entry.get("evaluation") or {}
            error = entry.get("error") or next(
                (
                    key
                    for key in ("dry_run", "error", "parse_error")
                    if evaluation.get(key)
                ),
                None,
            )
            if error:
                raise ValueError(str(error))
            judgment = judgment_for(entry, consistency_type, score_threshold)
            if not judgment.valid_parse:
                raise ValueError("invalid_parse")
            if judgment.score is not None and not np.isfinite(judgment.score):
                raise ValueError("Nonfinite score")
            judgments[clip_id] = not judgment.is_consistent
            timestamps[clip_id] = entry.get("timestamp_us")
        except ValueError as exc:
            errors[clip_id] = str(exc)

    rows, excluded, target_ids = [], [], set()
    for window in manifest["windows"]:
        ids = window["frame_ids"]
        times = window["timestamps_us"]
        if window["clip_id"] in target_ids:
            raise ValueError(f"Duplicate target: {window['clip_id']}")
        target_ids.add(window["clip_id"])
        if (
            len(ids) != 5
            or len(set(ids)) != 5
            or len(times) != 5
            or ids[-1] != window["clip_id"]
            or any(
                b - a != manifest["config"]["step_us"] for a, b in zip(times, times[1:])
            )
            or any(frame.rsplit("_t", 1)[0] != window["scene_id"] for frame in ids)
        ):
            raise ValueError(f"Invalid causal window: {window['clip_id']}")
        missing = [
            {"clip_id": frame, "reason": errors.get(frame, "Missing prediction")}
            for frame in ids
            if frame not in judgments
        ]
        missing.extend(
            {"clip_id": frame, "reason": "Prediction timestamp differs from window"}
            for frame, timestamp in zip(ids, times)
            if timestamps.get(frame) is not None and timestamps[frame] != timestamp
        )
        if missing:
            excluded.append(
                {
                    "clip_id": window["clip_id"],
                    "is_consistent": window["is_consistent"],
                    "frames": missing,
                }
            )
            continue
        flags = [judgments[frame] for frame in ids]
        decay_score = weighted_decay_score(flags, decay_half_life_frames)
        rows.append(
            {
                **window,
                "flags": flags,
                "weighted_decay_score": decay_score,
                "alarms": {
                    name: (
                        decay_score >= decay_threshold
                        if name == "weighted_decay"
                        else sum(flags[-n:]) >= k
                    )
                    for name, (k, n) in POLICIES.items()
                },
            }
        )

    summaries = {}
    intervals = paired_intervals(rows, bootstrap, seed) if rows else {}
    for name, (k, n) in POLICIES.items():
        confusion = BinaryConfusion()
        for row in rows:
            confusion.add(
                actual_is_consistent=row["is_consistent"],
                predicted_is_consistent=not row["alarms"][name],
            )
        metrics = classification_metrics(confusion)
        both_classes = (
            confusion.true_positive + confusion.false_negative > 0
            and confusion.true_negative + confusion.false_positive > 0
        )
        summaries[name] = {
            "k": k,
            "n": n,
            **(
                {
                    "half_life_frames": decay_half_life_frames,
                    "threshold": decay_threshold,
                }
                if name == "weighted_decay"
                else {}
            ),
            "n_targets": len(rows),
            "accuracy": confusion.accuracy if rows else None,
            "balanced_accuracy": metrics["balanced_accuracy"] if both_classes else None,
            **metrics["inconsistent"],
            "confusion": asdict(confusion),
        }
        # Undefined precision/recall are not evidence of zero performance.
        if not confusion.true_positive + confusion.false_positive:
            summaries[name]["precision"] = None
        if not confusion.true_positive + confusion.false_negative:
            summaries[name]["recall"] = None
        if not rows:
            summaries[name]["f1"] = None
        summaries[name]["difference_vs_single_frame"] = {}
        for metric in ("accuracy", "balanced_accuracy"):
            value = summaries[name][metric]
            baseline = summaries["single_frame"][metric]
            summaries[name]["difference_vs_single_frame"][metric] = {
                "estimate": value - baseline if value is not None else None,
                **intervals.get(name, {}).get(
                    metric, {"ci95": None, "n_bootstrap_valid": 0}
                ),
            }
    return {
        "schema_version": 1,
        "kind": "temporal_consistency_evaluation",
        "config": {
            "consistency_type": consistency_type,
            "score_threshold": score_threshold,
            "bootstrap": bootstrap,
            "seed": seed,
            "bootstrap_unit": "scene",
            "positive_class": "inconsistent",
            "window_order": "oldest_to_current",
            "step_us": manifest["config"]["step_us"],
            "decay_half_life_frames": decay_half_life_frames,
            "decay_threshold": decay_threshold,
        },
        "n_targets": manifest["n_targets"],
        "n_prepared": len(manifest["windows"]),
        "n_evaluated": len(rows),
        "n_scenes": len({row["scene_id"] for row in rows}),
        "n_consistent": sum(row["is_consistent"] for row in rows),
        "n_inconsistent": sum(not row["is_consistent"] for row in rows),
        "preparation_exclusions": manifest["excluded"],
        "prediction_exclusions": excluded,
        "policies": summaries,
        "targets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--consistency-type", choices=CONSISTENCY_TYPES, default="cot_output_alignment"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--decay-half-life-frames",
        type=float,
        default=2.0,
        help="Half-life of older frame flags (default: 2 frames).",
    )
    parser.add_argument(
        "--decay-threshold",
        type=float,
        default=0.5,
        help="Alarm when the weighted fraction of flags reaches this value (default: 0.5).",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must end in .json (the CSV uses the same stem)")
    result = evaluate(
        load_json(args.windows),
        load_json(args.predictions),
        args.consistency_type,
        score_threshold=args.score_threshold,
        decay_half_life_frames=args.decay_half_life_frames,
        decay_threshold=args.decay_threshold,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    result["config"].update(
        windows=str(args.windows.resolve()), predictions=str(args.predictions.resolve())
    )
    result["config"]["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("windows", args.windows), ("predictions", args.predictions))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = [
        "policy",
        "n_targets",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, summary in result["policies"].items():
            row = {"policy": name, **summary, **summary["confusion"]}
            writer.writerow({key: row[key] for key in columns})
    print(
        f"Evaluated {result['n_evaluated']}/{result['n_targets']} targets on a shared cohort."
    )
    print(
        f"{'Policy':22} {'Accuracy':>10} {'Balanced':>10} {'Precision':>10} {'Recall':>10}"
    )
    for name, summary in result["policies"].items():
        values = [
            "n/a" if summary[key] is None else f"{summary[key]:.3f}"
            for key in ("accuracy", "balanced_accuracy", "precision", "recall")
        ]
        print(f"{name:22}" + "".join(f" {value:>10}" for value in values))
    print(
        f"Report: {args.output} (with paired scene-bootstrap CIs); table: {args.output.with_suffix('.csv')}"
    )
    return 0 if result["n_evaluated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
