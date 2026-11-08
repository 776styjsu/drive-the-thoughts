"""Evaluate frozen SelfOracle alarms against annotated planned-trajectory safety.

This evaluates labeled decision frames, not closed-loop failures or lead times.
Pilot and pending-review full checkpoints require explicit opt-in and retain
their provisional status.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .data import read_json, scene_id, sha256, write_json
from .pipeline import Images, _device, filtered_scores
from .smoke import SCHEMA as SMOKE_SCHEMA, validate_smoke_manifest
from .full_training import SCHEMA as FULL_SCHEMA, validate_full_manifest

SCHEMA = "selforacle_annotated_safety_v1"
CAMERA = "camera_front_wide_120fov.jpg"


def _resolve(path: str, roots: list[Path]) -> Path:
    value = Path(path)
    candidates = [value] if value.is_absolute() else [root / value for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {path} under {roots}")


def prepare_manifest(benchmark: Path, source_benchmark: Path, source_roots: list[Path],
                     artifact_root: Path) -> dict:
    entries = read_json(benchmark)
    source_entries = read_json(source_benchmark)
    sources = {entry["clip_id"]: entry for entry in source_entries}
    if len(sources) != len(source_entries) or len({e["clip_id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate benchmark frame IDs")
    roots = [source_benchmark.resolve().parent, *source_roots]
    frames, windows, excluded = {}, [], []
    for entry in entries:
        cid = entry["clip_id"]
        safe = entry.get("planned_trajectory_safe")
        if type(safe) is not bool:
            raise ValueError(f"Missing boolean planned-trajectory safety label: {cid}")
        sid, suffix = cid.rsplit("_t", 1)
        target_index = int(suffix)
        try:
            source = _resolve(sources[cid]["source_scene_file"], roots)
            released = _resolve(entry["source_scene_file"], [artifact_root, benchmark.parent])
            original, published = read_json(source), read_json(released)
            for key in ("clip_id", "time_now_us", "chain_of_thought", "trajectory_poses"):
                if original[key] != published[key]:
                    raise ValueError(f"Source target differs from released metadata: {key}")
            if sha256(source.with_name(CAMERA)) != sha256(released.with_name(CAMERA)):
                raise ValueError("Source target camera differs from released benchmark JPEG")
            if target_index < 4:
                raise ValueError("Insufficient preceding frame indices")
            window = []
            for index in range(target_index - 4, target_index + 1):
                frame_id = f"{sid}_t{index:0{len(suffix)}d}"
                metadata = source.parent.parent / frame_id / "metadata.json"
                meta = read_json(metadata)
                expected = original["time_now_us"] - (target_index - index) * 100_000
                if meta["clip_id"] != frame_id or meta["time_now_us"] != expected:
                    raise ValueError("History identity or 10 Hz cadence mismatch")
                row = {"clip_id": frame_id, "scene_id": sid, "sequence_id": sid,
                       "now_us": expected, "image_path": str(metadata.with_name(CAMERA).resolve()),
                       "image_sha256": sha256(metadata.with_name(CAMERA)),
                       "metadata_path": str(metadata.resolve()), "metadata_sha256": sha256(metadata)}
                window.append(row)
            for row in window:
                if row["clip_id"] in frames and frames[row["clip_id"]] != row:
                    raise ValueError("Conflicting recording for benchmark frame")
                frames[row["clip_id"]] = row
            windows.append({"clip_id": cid, "scene_id": sid, "is_unsafe": not safe,
                            "cot_reliable": entry.get("cot_reliability", {}).get("reliable") is True,
                            "frame_ids": [row["clip_id"] for row in window],
                            "timestamps_us": [row["now_us"] for row in window]})
        except (OSError, ValueError, KeyError) as exc:
            excluded.append({"clip_id": cid, "is_unsafe": not safe, "reason": str(exc)})
    return {"schema_version": SCHEMA, "evaluation_type": "annotated_planned_trajectory_safety",
            "benchmark": str(benchmark.resolve()), "benchmark_sha256": sha256(benchmark),
            "source_benchmark": str(source_benchmark.resolve()),
            "source_benchmark_sha256": sha256(source_benchmark),
            "camera": CAMERA, "window": 5, "step_us": 100_000,
            "n_requested": len(entries), "windows": windows,
            "frames": sorted(frames.values(), key=lambda row: (row["sequence_id"], row["now_us"])),
            "excluded": excluded}


def validate_windows(manifest: dict, forbidden_scenes: set[str]) -> None:
    if manifest.get("schema_version") != SCHEMA or manifest.get("window") != 5 or manifest.get("step_us") != 100_000:
        raise ValueError("Unsupported annotated-safety window manifest")
    frames = {row["clip_id"]: row for row in manifest["frames"]}
    if len(frames) != len(manifest["frames"]) or not manifest["windows"]:
        raise ValueError("Empty benchmark windows or duplicate input frame")
    seen = set()
    for window in manifest["windows"]:
        cid, sid = window["clip_id"], window["scene_id"]
        if cid in seen or sid in forbidden_scenes:
            raise ValueError("Duplicate target or training/calibration scene leakage")
        seen.add(cid)
        ids, times = window["frame_ids"], window["timestamps_us"]
        if (len(ids) != 5 or len(set(ids)) != 5 or len(times) != 5 or ids[-1] != cid
                or any(b - a != 100_000 for a, b in zip(times, times[1:]))
                or type(window["is_unsafe"]) is not bool):
            raise ValueError("Invalid causal camera window or safety label")
        for frame, timestamp in zip(ids, times):
            row = frames[frame]
            if (scene_id(frame) != sid or row["scene_id"] != sid or row["sequence_id"] != sid
                    or row["now_us"] != timestamp):
                raise ValueError("Camera window frame identity mismatch")


def classification_metrics(actual: list[bool], scores: list[float], threshold: float) -> dict:
    y = np.asarray(actual, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if (len(y) != len(values) or not len(y) or not np.isfinite(values).all()
            or not math.isfinite(threshold)):
        raise ValueError("Invalid safety scores or threshold")
    flags = values > threshold
    tp, fp = int((y & flags).sum()), int((~y & flags).sum())
    tn, fn = int((~y & ~flags).sum()), int((y & ~flags).sum())
    positive, negative = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    if positive and negative:
        ranks = rankdata(values, method="average")
        auroc = float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * negative))
    else:
        auroc = None
    if positive:
        order = np.argsort(-values, kind="stable")
        ordered = values[order]
        ends = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
        cumulative_tp = np.cumsum(y[order])[ends]
        recall_curve = cumulative_tp / positive
        average_precision = float(np.sum(cumulative_tp / (ends + 1) * np.diff(np.r_[0, recall_curve])))
    else:
        average_precision = None
    return {"n": len(y), "n_unsafe": positive, "n_safe": negative,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "unsafe_recall": recall, "unsafe_precision": precision,
            "unsafe_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "safe_false_alarm_rate": fp / negative if negative else None,
            "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (recall + specificity) / 2 if positive and negative else None,
            "auroc_continuous_anomaly_score": auroc,
            "average_precision_unsafe": average_precision, "unsafe_prevalence": positive / len(y)}


def evaluate(args) -> None:
    import torch
    from .models import Reconstructor

    if args.output.exists():
        raise ValueError(f"Choose a new evaluation directory: {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = checkpoint["provenance"]
    pilot = provenance.get("schema_version") == SMOKE_SCHEMA
    full = provenance.get("schema_version") == FULL_SCHEMA
    if pilot and not args.allow_pilot_checkpoint:
        raise ValueError("Pilot checkpoint requires --allow-pilot-checkpoint; outputs remain exploratory")
    if full and not args.allow_pending_nominal:
        raise ValueError("Full checkpoint requires --allow-pending-nominal; outputs remain provisional")
    if provenance.get("schema_version") not in (1, SMOKE_SCHEMA, FULL_SCHEMA):
        raise ValueError("Unsupported checkpoint schema")
    training = read_json(args.training_manifest)
    if provenance["manifest_sha256"] != sha256(args.training_manifest):
        raise ValueError("Checkpoint does not match supplied training/calibration manifest")
    if pilot:
        validate_smoke_manifest(training)
    elif full:
        validate_full_manifest(training, allow_pending=args.allow_pending_nominal)
        if provenance.get("research_training_ready") is not False or provenance.get("nominal_review_pending") is not True:
            raise ValueError("Full checkpoint must preserve pending nominal-review status")
    else:
        from .data import validate_manifest
        validate_manifest(training)
    calibration = read_json(args.calibration)
    if calibration["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Calibration belongs to another checkpoint")
    if pilot and calibration.get("schema_version") != SMOKE_SCHEMA:
        raise ValueError("Pilot calibration must retain its technical artifact schema")
    if full:
        if (calibration.get("schema_version") != FULL_SCHEMA
                or calibration.get("nominal_review_pending") is not True):
            raise ValueError("Full calibration must preserve its pending-review artifact schema")
        if calibration.get("manifest_sha256") != provenance["manifest_sha256"]:
            raise ValueError("Full calibration belongs to another training manifest")
        if calibration.get("config") != provenance["config"]:
            raise ValueError("Full calibration preprocessing differs from checkpoint")
    if calibration["window"] != 5 or calibration["max_gap_us"] != 150_000:
        raise ValueError("Calibration temporal settings differ from prepared windows")
    manifest = read_json(args.manifest)
    forbidden = {row["scene_id"] for row in training["frames"]
                 if pilot or full or row["split"] in ("train", "calibration")}
    validate_windows(manifest, forbidden)
    config = provenance["config"]
    if config["camera"] != manifest["camera"]:
        raise ValueError("Evaluation camera differs from training")
    if sha256(Path(manifest["benchmark"])) != manifest["benchmark_sha256"]:
        raise ValueError("Safety labels changed after preparation")
    if sha256(Path(manifest["source_benchmark"])) != manifest["source_benchmark_sha256"]:
        raise ValueError("Source benchmark changed after preparation")
    for row in manifest["frames"]:
        if sha256(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError("Evaluation camera changed after preparation")
        if sha256(Path(row["metadata_path"])) != row["metadata_sha256"]:
            raise ValueError("Evaluation metadata changed after preparation")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and worker count nonnegative")
    device = _device(args.device)
    if device.type == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU evaluation requires a Slurm allocation")
    model = Reconstructor(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = torch.utils.data.DataLoader(Images(manifest["frames"], config), batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)
    values, start = [], time.perf_counter()
    with torch.inference_mode():
        for images in loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            values.extend((reconstructed - images).square().sum(dim=1).sqrt().cpu().tolist())
    if len(values) != len(manifest["frames"]) or not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite camera predictions")
    by_id = {row["clip_id"]: (row, value) for row, value in zip(manifest["frames"], values)}
    distributions = calibration["calibrations"] if pilot or full else [{"distribution": calibration["distribution"]}]
    if not distributions:
        raise ValueError("No frozen calibration threshold")
    targets = []
    for window in manifest["windows"]:
        pairs = [by_id[frame] for frame in window["frame_ids"]]
        filtered = filtered_scores([pair[0] for pair in pairs], [pair[1] for pair in pairs], 5, 150_000)
        if not filtered[-1]["scored"]:
            raise ValueError("Target missing a full causal temporal window")
        target = {**window, "raw_score": pairs[-1][1], "filtered_score": filtered[-1]["score"]}
        for item in distributions:
            distribution = item["distribution"]
            epsilon, threshold = distribution["epsilon"], distribution["threshold"]
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError("Invalid frozen threshold")
            target[f"flag_epsilon_{epsilon}"] = target["filtered_score"] > threshold
        targets.append(target)
    summaries = []
    for cohort in ("all_150", "reliable_cot_subset"):
        rows = targets if cohort == "all_150" else [row for row in targets if row["cot_reliable"]]
        if not rows:
            continue
        for policy, score_key in (("five_frame_mean", "filtered_score"), ("single_frame", "raw_score")):
            for item in distributions:
                distribution = item["distribution"]
                summaries.append({"cohort": cohort, "policy": policy,
                                  "epsilon": distribution["epsilon"], "threshold": distribution["threshold"],
                                  **classification_metrics([row["is_unsafe"] for row in rows],
                                                           [row[score_key] for row in rows], distribution["threshold"])})
    result = {"schema_version": SCHEMA, "monitor": "SelfOracle adapted to Alpasim",
              "status": "completed", "evaluation_type": manifest["evaluation_type"],
              "positive_class": "unsafe_planned_trajectory",
              "result_status": ("exploratory_pilot" if pilot else
                                "full_scale_provisional_nominal_review_pending" if full else
                                "trained_baseline_evaluation"),
              "nominal_review_pending": pilot or full,
              "research_training_ready": not (pilot or full),
              "training_epochs": provenance["epochs"], "training_seed": provenance.get("seed", config.get("seed")),
              "epochs_completed": provenance.get("epochs_completed", provenance["epochs"]),
              "checkpoint_selection": "lowest_validation_reconstruction_MSE" if full else "fixed_training_epochs",
              "config": config, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": sha256(args.checkpoint), "calibration_sha256": sha256(args.calibration),
              "training_manifest_sha256": sha256(args.training_manifest),
              "window_manifest_sha256": sha256(args.manifest), "benchmark_sha256": manifest["benchmark_sha256"],
              "source_code_sha256": {name: sha256(Path(__file__).with_name(name))
                                     for name in ("benchmark_safety.py", "models.py", "pipeline.py")},
              "thresholds_selected_using_benchmark": False, "n_requested": manifest["n_requested"],
              "n_scored": len(targets), "excluded": manifest["excluded"], "summaries": summaries,
              "targets": targets, "device": str(device), "torch_version": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "elapsed_seconds": time.perf_counter() - start,
              "limitations": ["Annotated planned-trajectory labels, not simulated collision outcomes or lead time.",
                              "Visual anomaly scores do not establish CoT inconsistency.",
                              "Training/calibration nominal review is pending; results are provisional." if pilot or full else
                              "Training and calibration nominal status comes from the supplied manifest."]}
    args.output.mkdir(parents=True)
    write_json(args.output / "evaluation.json", result)
    flat = [{**{key: value for key, value in row.items() if key != "confusion"}, **row["confusion"]}
            for row in summaries]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    fieldnames = ["clip_id", "scene_id", "is_unsafe", "cot_reliable", "raw_score", "filtered_score"]
    fieldnames += [f"flag_epsilon_{item['distribution']['epsilon']}" for item in distributions]
    with (args.output / "per_target.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(targets)
    print(f"{config['model']}: scored {len(targets)}/{manifest['n_requested']} annotated targets", flush=True)
    for row in summaries:
        if row["cohort"] == "all_150" and row["policy"] == "five_frame_mean":
            print(f"epsilon={row['epsilon']}: {row['confusion']}; recall={row['unsafe_recall']:.3f}, "
                  f"precision={row['unsafe_precision']}, safe-FAR={row['safe_false_alarm_rate']:.3f}, "
                  f"AUROC={row['auroc_continuous_anomaly_score']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="command", required=True)
    preparation = stages.add_parser("prepare")
    preparation.add_argument("--benchmark", type=Path, default=Path("data/benchmark/benchmark.json"))
    preparation.add_argument("--source-benchmark", type=Path, required=True)
    preparation.add_argument("--source-root", type=Path, action="append", default=[])
    preparation.add_argument("--artifact-root", type=Path, default=Path.cwd())
    preparation.add_argument("--output", type=Path, required=True)
    execution = stages.add_parser("evaluate")
    execution.add_argument("--manifest", type=Path, required=True)
    execution.add_argument("--checkpoint", type=Path, required=True)
    execution.add_argument("--calibration", type=Path, required=True)
    execution.add_argument("--training-manifest", type=Path, required=True)
    execution.add_argument("--allow-pilot-checkpoint", action="store_true")
    execution.add_argument("--allow-pending-nominal", action="store_true")
    execution.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    execution.add_argument("--workers", type=int, default=4)
    execution.add_argument("--batch-size", type=int, default=32)
    execution.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.output.exists():
            raise ValueError("Window manifest already exists")
        manifest = prepare_manifest(args.benchmark, args.source_benchmark, args.source_root, args.artifact_root)
        validate_windows(manifest, set())
        write_json(args.output, manifest)
        print(f"Prepared {len(manifest['windows'])}/{manifest['n_requested']} targets; "
              f"{len(manifest['frames'])} input frames; {len(manifest['excluded'])} exclusions", flush=True)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
