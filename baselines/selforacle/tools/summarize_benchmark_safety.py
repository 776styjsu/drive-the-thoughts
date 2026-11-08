"""Verify frozen six-run safety evaluations and summarize variability across seeds."""

import argparse
import csv
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from selforacle_baseline.data import read_json, sha256, write_json


def independent_metrics(labels, values, threshold):
    flags = [value > threshold for value in values]
    tp = sum(label and flag for label, flag in zip(labels, flags))
    fp = sum(not label and flag for label, flag in zip(labels, flags))
    tn = sum(not label and not flag for label, flag in zip(labels, flags))
    fn = sum(label and not flag for label, flag in zip(labels, flags))
    positive = [value for label, value in zip(labels, values) if label]
    negative = [value for label, value in zip(labels, values) if not label]
    auroc = (sum(float(p > n) + 0.5 * (p == n) for p in positive for n in negative)
             / (len(positive) * len(negative))) if positive and negative else None
    ap, previous_recall = 0.0, 0.0
    for cutoff in sorted(set(values), reverse=True):
        retrieved = [label for label, value in zip(labels, values) if value >= cutoff]
        recall = sum(retrieved) / len(positive) if positive else 0.0
        ap += (recall - previous_recall) * sum(retrieved) / len(retrieved)
        previous_recall = recall
    return {"confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "unsafe_recall": tp / len(positive) if positive else None,
            "unsafe_precision": tp / (tp + fp) if tp + fp else None,
            "safe_false_alarm_rate": fp / len(negative) if negative else None,
            "auroc_continuous_anomaly_score": auroc,
            "average_precision_unsafe": ap if positive else None}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    snapshot = read_json(root / "input_snapshot.json")
    submission = read_json(root / "submission.json")
    windows = read_json(root / "windows.json")
    assert sha256(root / "input_snapshot.json") == submission["input_snapshot_sha256"]
    assert sha256(root / "windows.json") == snapshot["window_manifest_sha256"]
    assert sha256(Path(windows["benchmark"])) == snapshot["benchmark_sha256"]
    assert sha256(Path(windows["source_benchmark"])) == snapshot["source_benchmark_sha256"]
    assert sha256(Path(snapshot["training_manifest_path"])) == snapshot["training_manifest_sha256"]
    assert sha256(Path(snapshot["protocol_path"])) == snapshot["protocol_sha256"]
    labels = {entry["clip_id"]: entry for entry in read_json(Path(windows["benchmark"]))}
    window_by_id = {window["clip_id"]: window for window in windows["windows"]}
    comparisons, verified = [], []
    for model in snapshot["models"]:
        result_path = root / "results" / model["tag"] / "evaluation.json"
        result = read_json(result_path)
        calibration = read_json(Path(model["calibration_path"]))
        assert sha256(Path(model["checkpoint_path"])) == model["checkpoint_sha256"] == result["checkpoint_sha256"]
        assert sha256(Path(model["calibration_path"])) == model["calibration_sha256"] == result["calibration_sha256"]
        assert result["training_manifest_sha256"] == snapshot["training_manifest_sha256"]
        assert result["window_manifest_sha256"] == snapshot["window_manifest_sha256"]
        assert result["benchmark_sha256"] == snapshot["benchmark_sha256"]
        assert result["source_code_sha256"]["benchmark_safety.py"] == submission["evaluation_code_sha256"]
        assert result["status"] == "completed" and result["n_scored"] == result["n_requested"] == 150
        assert not result["excluded"] and result["nominal_review_pending"] is True
        assert result["result_status"] == "full_scale_provisional_nominal_review_pending"
        assert result["thresholds_selected_using_benchmark"] is False
        assert result["training_seed"] == model["seed"] and result["training_epochs"] == model["selected_epoch"]
        assert result["epochs_completed"] == model["epochs_completed"]
        assert {target["clip_id"] for target in result["targets"]} == set(labels)
        thresholds = {item["distribution"]["epsilon"]: item["distribution"]["threshold"]
                      for item in calibration["calibrations"]}
        for target in result["targets"]:
            entry, expected = labels[target["clip_id"]], window_by_id[target["clip_id"]]
            assert target["is_unsafe"] == (not entry["planned_trajectory_safe"])
            assert target["cot_reliable"] == (entry.get("cot_reliability", {}).get("reliable") is True)
            assert target["frame_ids"] == expected["frame_ids"] and target["timestamps_us"] == expected["timestamps_us"]
            assert math.isfinite(target["raw_score"]) and math.isfinite(target["filtered_score"])
            for epsilon, threshold in thresholds.items():
                assert target[f"flag_epsilon_{epsilon}"] == (target["filtered_score"] > threshold)
        for row in result["summaries"]:
            assert row["threshold"] == thresholds[row["epsilon"]]
            targets = (result["targets"] if row["cohort"] == "all_150"
                       else [target for target in result["targets"] if target["cot_reliable"]])
            score_key = "filtered_score" if row["policy"] == "five_frame_mean" else "raw_score"
            actual = [not labels[target["clip_id"]]["planned_trajectory_safe"] for target in targets]
            check = independent_metrics(actual, [target[score_key] for target in targets], row["threshold"])
            assert row["n"] == len(targets) and row["n_unsafe"] == sum(actual)
            assert row["confusion"] == check["confusion"]
            for key in check.keys() - {"confusion"}:
                assert ((row[key] is None and check[key] is None)
                        or (row[key] is not None and check[key] is not None
                            and math.isclose(row[key], check[key], abs_tol=1e-12)))
            comparisons.append({"model": model["model"], "seed": model["seed"],
                                "selected_epoch": model["selected_epoch"], "epochs_completed": model["epochs_completed"],
                                **{key: value for key, value in row.items() if key != "confusion"},
                                **row["confusion"]})
        verified.append({"model": model["tag"], "evaluation_sha256": sha256(result_path),
                         "checkpoint_and_calibration_unchanged": True,
                         "all_labels_and_causal_timestamps_match": True,
                         "independent_confusion_pairwise_AUROC_and_AP_checks": "passed"})
    assert len(verified) == 6 and len(comparisons) == 48
    write_csv(root / "comparison.csv", comparisons)
    metric_names = ["unsafe_recall", "unsafe_precision", "unsafe_f1", "safe_false_alarm_rate",
                    "balanced_accuracy", "auroc_continuous_anomaly_score", "average_precision_unsafe"]
    groups = sorted({(row["model"], row["cohort"], row["policy"], row["epsilon"]) for row in comparisons})
    aggregates = []
    for model, cohort, policy, epsilon in groups:
        rows = [row for row in comparisons if (row["model"], row["cohort"], row["policy"], row["epsilon"])
                == (model, cohort, policy, epsilon)]
        assert sorted(row["seed"] for row in rows) == [0, 1, 2]
        aggregate = {"model": model, "cohort": cohort, "policy": policy, "epsilon": epsilon,
                     "n_seeds": 3, "n_targets_per_seed": rows[0]["n"],
                     "n_unsafe_per_seed": rows[0]["n_unsafe"], "n_safe_per_seed": rows[0]["n_safe"]}
        for metric in metric_names:
            values = [row[metric] for row in rows if row[metric] is not None]
            aggregate.update({metric + "_mean": statistics.mean(values) if values else None,
                              metric + "_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                              metric + "_n_defined": len(values)})
        aggregates.append(aggregate)
    write_csv(root / "aggregate.csv", aggregates)
    write_json(root / "aggregate.json", {
        "result_status": "full_scale_provisional_nominal_review_pending",
        "primary_cohort": "all_150", "primary_policy": "five_frame_mean",
        "aggregation": "mean and sample SD across three training seeds on the same targets; not a confidence interval",
        "undefined_precision": "excluded from mean with n_defined recorded; not replaced by zero",
        "summaries": aggregates})
    write_json(root / "verification.json", {"status": "passed", "job_id": submission["job_id"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "models": verified,
        "all_frozen_training_and_calibration_artifacts_unchanged": True,
        "benchmark_and_source_benchmark_unchanged": True,
        "no_training_validation_calibration_nominal_test_scene_overlap": True,
        "all_150_scored_for_each_seed": True})
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in files))
    for row in comparisons:
        if row["cohort"] == "all_150" and row["policy"] == "five_frame_mean":
            print(f"{row['model']} seed {row['seed']} epsilon {row['epsilon']}: "
                  f"TP={row['tp']} FP={row['fp']} TN={row['tn']} FN={row['fn']} "
                  f"recall={row['unsafe_recall']:.3f} FAR={row['safe_false_alarm_rate']:.3f} "
                  f"AUROC={row['auroc_continuous_anomaly_score']:.3f}")
    print("Verified all six models; per-seed and aggregate results saved.")


if __name__ == "__main__":
    main()
