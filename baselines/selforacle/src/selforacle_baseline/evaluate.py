"""Adapt camera alarms to the existing AlpaSim rollout safety evaluator.

Run this module in the simulator environment; importing it requires only stdlib.
"""

from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from .data import read_json, sha256, write_json


def align_to_decisions(rows: list[dict], decisions: list[int], max_age_us: int) -> list[dict]:
    """Select the latest camera verdict at or before each driver decision."""
    if max_age_us < 0:
        raise ValueError("max-age-us must be nonnegative")
    if len({r["sequence_id"] for r in rows}) != 1:
        raise ValueError("Align one nonempty recording at a time")
    rows = sorted(rows, key=lambda r: r["now_us"])
    times = [r["now_us"] for r in rows]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate camera timestamps")
    if decisions != sorted(set(decisions)):
        raise ValueError("Driver decision timestamps must be unique and increasing")
    aligned = []
    for now in decisions:
        index = bisect_right(times, now) - 1
        row = rows[index] if index >= 0 else None
        usable = row is not None and now - row["now_us"] <= max_age_us and row["scored"] is True
        if usable and type(row["flag"]) is not bool:
            raise ValueError("Scored prediction must have a boolean flag")
        aligned.append({"now_us": now, "camera_time_us": row["now_us"] if row else None,
                        "scored": bool(usable), "flag": row["flag"] if usable else False})
    return aligned


async def _decision_times(path: Path) -> list[int]:
    from alpasim_utils.logs import async_read_pb_log

    pending = None
    times = []
    async for message in async_read_pb_log(str(path)):
        kind = message.WhichOneof("log_entry")
        if kind == "driver_request":
            if pending is not None:
                raise ValueError("Overlapping driver requests; cannot pair decisions safely")
            pending = message.driver_request.time_now_us
        elif kind == "driver_return" and pending is not None:
            times.append(pending)
            pending = None
    return times


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--rollout-map", type=Path, required=True,
                        help="JSON object mapping sequence IDs to rollout.asl paths; metrics must be adjacent")
    parser.add_argument("--outcome", default="offroad_or_collision_at_fault")
    parser.add_argument("--horizon-s", type=float, default=4.0)
    parser.add_argument("--max-age-us", type=int, default=150_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from rollout_monitoring.analysis import analyze_rollout, load_outcome_series
    from rollout_monitoring.monitor import MonitorStep
    from rollout_monitoring.__main__ import _summarize

    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    if args.horizon_s <= 0 or args.max_age_us < 0:
        raise ValueError("Require a positive horizon and nonnegative maximum image age")
    predictions = read_json(args.predictions)
    if predictions.get("monitor") != "selforacle_alpasim_v1" or predictions.get("schema_version") != 1:
        raise ValueError("Expected SelfOracle prediction schema version 1")
    groups = defaultdict(list)
    for row in predictions["results"]:
        groups[row["sequence_id"]].append(row)
    mapping = read_json(args.rollout_map)
    if not groups or set(mapping) != set(groups):
        raise ValueError("Rollout map must contain exactly the prediction sequence IDs")
    results, excluded, coverage, seen_paths = [], [], {}, set()
    for sequence, rows in sorted(groups.items()):
        asl = Path(mapping[sequence])
        if not asl.is_absolute():
            asl = args.rollout_map.resolve().parent / asl
        asl = asl.resolve()
        if asl in seen_paths:
            raise ValueError("Multiple sequences mapped to the same rollout; this would double count it")
        seen_paths.add(asl)
        scene_ids = {r["scene_id"] for r in rows}
        if scene_ids != {asl.parent.parent.name}:
            raise ValueError(f"Prediction scene does not match rollout directory: {asl}")
        metrics = asl.with_name("metrics.parquet")
        decisions = asyncio.run(_decision_times(asl))
        aligned = align_to_decisions(rows, decisions, args.max_age_us)
        steps = [MonitorStep(
            now_us=r["now_us"], n_poses=0, has_cot=False, scored=r["scored"],
            score=None, label=None, longitudinal_verdict=None, lateral_verdict=None,
            flag=r["flag"],
        ) for r in aligned]
        series = load_outcome_series(metrics)
        usable = [r for r in aligned if r["scored"] and
                  series.eval_start_us <= r["now_us"] <= series.window_end_us]
        coverage[sequence] = {"rollout_asl": str(asl), "metrics_sha256": sha256(metrics),
                              "eligible_decision_timestamps_us": [r["now_us"] for r in usable],
                              "total_decisions": len(decisions), "scored_decisions": len(usable)}
        if series.img_black or not usable:
            excluded.append({"sequence_id": sequence,
                             "reason": "black_render" if series.img_black else "no_scored_decisions"})
            continue
        result = analyze_rollout(steps, series, clip_id=next(iter(scene_ids)),
                                 rollout_id=asl.parent.name, outcome=args.outcome,
                                 horizon_s=args.horizon_s)
        results.append(result)
    summary = _summarize(results)
    summary["auroc_anomaly_flag_rate_vs_failure"] = summary.pop("auroc_inconsistency_rate_vs_failure")
    # This adapter measures alarms; it does not establish intervention effectiveness.
    summary.pop("intervention_upper_bound_failures_averted", None)
    records = []
    for result in results:
        row = result.to_dict()
        row["anomaly_flag_rate"] = row.pop("inconsistency_rate")
        records.append(row)
    write_json(args.output, {
        "schema_version": 1, "monitor": predictions["monitor"],
        "predictions_sha256": sha256(args.predictions), "rollout_map_sha256": sha256(args.rollout_map),
        "max_age_us": args.max_age_us, "horizon_s": args.horizon_s, "outcome": args.outcome,
        "summary": summary, "rollouts": records, "coverage": coverage, "excluded": excluded,
    })


if __name__ == "__main__":
    main()

