# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Closed-loop online CoT monitoring (Experiment B), CLI.

Walks a directory of Alpasim closed-loop rollouts, runs the rule-based CoT
consistency monitor over every decision step of each ``rollout.asl``, and joins
the per-step inconsistency flags with the per-timestep simulated safety outcomes
the evaluator already wrote to ``metrics.parquet``. Reports:

* whether an inconsistency flag *leads* a simulated failure (event recall, both
  within a horizon ``H`` and over the whole pre-failure window),
* the lead-time distribution,
* per-flag precision (does a flag imply an imminent failure within ``H``),
* a rollout-level confusion table + AUROC of the inconsistency rate, and
* an offline upper bound on the failures an at-flag fallback could have averted.

Usage (from the ``alpasim/`` workspace root)::

    PYTHONPATH=src/tools uv run --project src/tools python -m rollout_monitoring \\
        --rollout_root <dir-with-scene_*/rollouts/**/rollout.asl> \\
        --outcome offroad_or_collision_at_fault --horizon_s 4 \\
        --output online_monitoring.json \\
        --csv online_monitoring.csv

This needs no GPU and no re-simulation: the rollouts and their per-timestep
metrics already exist; the tool replays the logged decisions through the monitor.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import polars as pl

from rollout_monitoring.analysis import (
    DEFAULT_MAX_DIST_TO_GT_M,
    DEFAULT_OUTCOME,
    OUTCOME_COMPONENTS,
    PREDICTORS,
    RolloutResult,
    analyze_rollout,
    load_outcome_series,
)
from rollout_monitoring.monitor import monitor_steps


def _discover_rollouts(root: Path) -> list[tuple[Path, Path]]:
    """Find ``(rollout.asl, metrics.parquet)`` pairs under ``root``."""
    pairs: list[tuple[Path, Path]] = []
    for asl in sorted(root.glob("**/rollouts/**/rollout.asl")):
        parquet = asl.with_name("metrics.parquet")
        if parquet.exists():
            pairs.append((asl, parquet))
    return pairs


def _read_ids(parquet: Path, fallback: Path) -> tuple[str, str]:
    """Pull clip / rollout ids from the parquet, else from the path."""
    try:
        df = pl.read_parquet(str(parquet), columns=["clipgt_id", "rollout_id"])
        if df.height:
            return str(df["clipgt_id"][0]), str(df["rollout_id"][0])
    except Exception:
        pass
    return fallback.parent.parent.name, fallback.parent.name


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U); higher score => more likely positive."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _summarize(results: list[RolloutResult]) -> dict:
    """Aggregate per-rollout results into the leading-indicator summary."""
    failed = [r for r in results if r.failed]
    safe = [r for r in results if not r.failed]
    n_fail, n_safe = len(failed), len(safe)

    tp = sum(1 for r in failed if r.flag_before_failure)
    tp_h = sum(1 for r in failed if r.flag_within_horizon)
    fp = sum(1 for r in safe if r.false_alarm)
    tn = n_safe - fp

    n_flags = sum(r.n_flag for r in results)
    n_flags_followed = sum(r.n_flags_followed_by_failure for r in results)
    leads = [r.lead_time_s for r in failed if r.lead_time_s is not None]

    def _frac(a: int, b: int) -> float | None:
        return a / b if b else None

    lead_stats = None
    if leads:
        lead_stats = {
            "n": len(leads),
            "median_s": round(statistics.median(leads), 3),
            "mean_s": round(statistics.mean(leads), 3),
            "min_s": round(min(leads), 3),
            "max_s": round(max(leads), 3),
        }

    return {
        "n_rollouts": len(results),
        "n_failed": n_fail,
        "n_safe": n_safe,
        "event_recall_any_pre_failure": _frac(tp, n_fail),
        "event_recall_within_horizon": _frac(tp_h, n_fail),
        "rollout_false_alarm_rate": _frac(fp, n_safe),
        "rollout_confusion": {"tp": tp, "fn": n_fail - tp, "fp": fp, "tn": tn},
        "rollout_precision": _frac(tp, tp + fp),
        "flag_precision_within_horizon": _frac(n_flags_followed, n_flags),
        "n_flags_total": n_flags,
        "n_flags_followed_by_failure": n_flags_followed,
        "auroc_inconsistency_rate_vs_failure": (
            round(v, 3)
            if (
                v := _auroc(
                    [r.inconsistency_rate for r in results], [r.failed for r in results]
                )
            )
            is not None
            else None
        ),
        "lead_time": lead_stats,
        "intervention_upper_bound_failures_averted": _frac(tp, n_fail),
    }


def _print_summary(predictor: str, summ: dict) -> None:
    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    conf = summ["rollout_confusion"]
    print(f"\n── predictor = {predictor} " + "─" * 40)
    print(
        f"  rollouts={summ['n_rollouts']}  failed={summ['n_failed']}  safe={summ['n_safe']}"
    )
    print(
        f"  leading-indicator recall: any-pre={f(summ['event_recall_any_pre_failure'])}  "
        f"within-H={f(summ['event_recall_within_horizon'])}"
    )
    print(
        f"  rollout confusion: tp={conf['tp']} fn={conf['fn']} fp={conf['fp']} tn={conf['tn']}  "
        f"precision={f(summ['rollout_precision'])}  "
        f"false-alarm-rate={f(summ['rollout_false_alarm_rate'])}"
    )
    print(
        f"  per-flag precision (within-H)={f(summ['flag_precision_within_horizon'])}  "
        f"({summ['n_flags_followed_by_failure']}/{summ['n_flags_total']} flags)"
    )
    print(
        f"  AUROC(inconsistency-rate -> failure)={f(summ['auroc_inconsistency_rate_vs_failure'])}"
    )
    if summ["lead_time"]:
        lt = summ["lead_time"]
        print(
            f"  lead time (n={lt['n']}): median={lt['median_s']}s mean={lt['mean_s']}s "
            f"range=[{lt['min_s']}, {lt['max_s']}]s"
        )
    print(
        "  offline intervention upper bound (failures preceded by a warning): "
        f"{f(summ['intervention_upper_bound_failures_averted'])}"
    )


def _write_csv(results: list[RolloutResult], path: Path) -> None:
    if not results:
        return
    fields = list(results[0].to_dict().keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r.to_dict() for r in results)


def run(args: argparse.Namespace) -> int:
    root = Path(args.rollout_root)
    pairs = _discover_rollouts(root)
    if not pairs:
        print(f"No rollout.asl/metrics.parquet pairs under {root}", file=sys.stderr)
        return 1

    predictors = list(PREDICTORS)
    results_by_pred: dict[str, list[RolloutResult]] = {p: [] for p in predictors}
    skipped: list[dict] = []

    print(f"Discovered {len(pairs)} rollouts under {root}")
    for idx, (asl, parquet) in enumerate(pairs, 1):
        clip_id, rollout_id = _read_ids(parquet, asl.parent)
        try:
            steps = monitor_steps(asl)
            series = load_outcome_series(parquet, max_dist_to_gt_m=args.max_dist_to_gt)
        except Exception as exc:  # noqa: BLE001 - report and continue
            skipped.append({"clip_id": clip_id, "asl": str(asl), "reason": str(exc)})
            continue
        if series.img_black:
            # Mirrors RemoveTrajectoryWithEvent(img_is_black>0): drop the rollout.
            skipped.append(
                {"clip_id": clip_id, "asl": str(asl), "reason": "img_is_black"}
            )
            continue
        for predictor in predictors:
            results_by_pred[predictor].append(
                analyze_rollout(
                    steps,
                    series,
                    clip_id=clip_id,
                    rollout_id=rollout_id,
                    outcome=args.outcome,
                    predictor=predictor,
                    horizon_s=args.horizon_s,
                )
            )
        if idx % 10 == 0 or idx == len(pairs):
            print(f"  processed {idx}/{len(pairs)} ...", file=sys.stderr)

    primary = predictors[0]
    if not results_by_pred[primary]:
        print("No rollouts produced results.", file=sys.stderr)
        return 1

    print("=" * 70)
    print(
        f"Online CoT monitoring vs simulated outcome '{args.outcome}' "
        f"(horizon {args.horizon_s}s, {len(results_by_pred[primary])} rollouts, "
        f"{len(skipped)} skipped)"
    )
    print("=" * 70)
    summaries = {p: _summarize(results_by_pred[p]) for p in predictors}
    for predictor in predictors:
        _print_summary(predictor, summaries[predictor])

    out = {
        "config": {
            "rollout_root": str(root),
            "outcome": args.outcome,
            "predictors": predictors,
            "horizon_s": args.horizon_s,
            "max_dist_to_gt_m": args.max_dist_to_gt,
        },
        "n_rollouts": len(results_by_pred[primary]),
        "n_skipped": len(skipped),
        "summary": summaries,
        "results": [r.to_dict() for r in results_by_pred[primary]],
        "skipped": skipped,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote per-rollout results ({primary}) -> {args.output}")
    if args.csv:
        _write_csv(results_by_pred[primary], Path(args.csv))
        print(f"Wrote CSV -> {args.csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rollout_monitoring",
        description="Closed-loop online CoT monitoring leading-indicator analysis (Experiment B).",
    )
    p.add_argument(
        "--rollout_root",
        required=True,
        help="Directory holding scene_*/rollouts/**/rollout.asl with sibling metrics.parquet.",
    )
    p.add_argument(
        "--outcome",
        default=DEFAULT_OUTCOME,
        choices=list(OUTCOME_COMPONENTS),
        help=f"Simulated failure to predict (default: {DEFAULT_OUTCOME}).",
    )
    p.add_argument(
        "--horizon_s",
        type=float,
        default=4.0,
        help="Lead-time horizon in seconds for within-H recall and per-flag precision.",
    )
    p.add_argument(
        "--max_dist_to_gt",
        type=float,
        default=DEFAULT_MAX_DIST_TO_GT_M,
        help="Truncate the valid window once dist_to_gt_trajectory reaches this (default: 4.0 m).",
    )
    p.add_argument(
        "--output",
        default="online_monitoring.json",
        help="Per-rollout results JSON.",
    )
    p.add_argument(
        "--csv", default=None, help="Optional flat CSV of per-rollout results."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
