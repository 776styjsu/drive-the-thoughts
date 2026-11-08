"""Technical training pilot on audited, possibly unreviewed training-pool scenes.

This deliberately uses a different artifact schema from the research pipeline.
It never declares pending scenes nominal or relaxes the production data gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .data import read_json, scene_id, sha256, write_json
from .pipeline import filtered_scores, fit_threshold

SCHEMA = "selforacle_technical_smoke_v1"
PURPOSE = "technical_smoke_test_only_not_for_paper_evaluation"


def validate_smoke_manifest(manifest: dict) -> dict:
    if manifest.get("schema_version") != SCHEMA or manifest.get("intended_use") != PURPOSE:
        raise ValueError("Requires a dedicated technical-smoke manifest")
    if manifest.get("training_ready") is not False:
        raise ValueError("A smoke manifest must not claim research training readiness")
    scenes, sequences, timestamps, paths = {}, {}, set(), set()
    counts = Counter()
    if not manifest.get("frames"):
        raise ValueError("Empty smoke manifest")
    for row in manifest["frames"]:
        sid, split, sequence = row["scene_id"], row["split"], row["sequence_id"]
        if row.get("source_split") != "train":
            raise ValueError("Smoke scenes must come exclusively from the original training pool")
        if sid in manifest.get("reserved_test_scenes", []):
            raise ValueError("Reserved evaluation scene used in smoke pilot")
        if split not in ("train", "calibration", "test"):
            raise ValueError("Invalid smoke split")
        if scenes.setdefault(sid, split) != split:
            raise ValueError("Scene leakage across smoke splits")
        if sequences.setdefault(sequence, sid) != sid:
            raise ValueError("Smoke sequence reused across scenes")
        if scene_id(row["clip_id"]) != sid:
            raise ValueError("Smoke frame and scene IDs disagree")
        if row.get("nominal") not in (None, True):
            raise ValueError("An explicitly rejected nominal scene cannot enter the pilot")
        if type(row["now_us"]) is not int or row["now_us"] < 0:
            raise ValueError("Invalid smoke timestamp")
        key = (sequence, row["now_us"])
        image = Path(row["image_path"])
        if key in timestamps or image in paths:
            raise ValueError("Duplicate smoke frame")
        if not image.is_absolute() or image.name != manifest["camera"]:
            raise ValueError("Invalid smoke camera path")
        timestamps.add(key)
        paths.add(image)
        counts[split] += 1
    if set(counts) != {"train", "calibration", "test"}:
        raise ValueError("All three smoke splits are required")
    for sequence in sequences:
        times = sorted(r["now_us"] for r in manifest["frames"] if r["sequence_id"] == sequence)
        if any(b - a != 100_000 for a, b in zip(times, times[1:])):
            raise ValueError("Smoke pilot requires complete consecutive 10 Hz streams")
    return {"frames": dict(counts), "scenes": dict(Counter(scenes.values()))}


def _select(pool: list[dict], count: int, seed: int, split: str) -> list[dict]:
    if count > len(pool) or count < 1:
        raise ValueError("Insufficient training-pool scenes for pilot")
    groups = defaultdict(list)
    for row in pool:
        groups[(row["time_of_day"], row["ego_speed"])].append(row)
    quotas = {key: count * len(rows) / len(pool) for key, rows in groups.items()}
    sizes = {key: int(value) for key, value in quotas.items()}
    order = sorted(groups, key=lambda key: (-(quotas[key] - sizes[key]), key))
    for key in order[:count - sum(sizes.values())]:
        sizes[key] += 1
    selected = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda row: hashlib.sha256(
            f"selforacle-smoke:{seed}:{split}:{row['scene_id']}".encode()).hexdigest())
        selected.extend(ranked[:sizes[key]])
    return selected


def prepare(args) -> None:
    if args.output.exists():
        raise ValueError(f"Smoke manifest already exists: {args.output}")
    audit = args.campaign / "manifest_audit"
    checksums = {name: digest for digest, name in
                 (line.split(maxsplit=1) for line in (audit / "SHA256SUMS").read_text().splitlines())}
    for name in ("manifest.draft.json", "inventory.draft.json", "validation.json", "audit_inputs.json"):
        if sha256(audit / name) != checksums[name]:
            raise ValueError(f"Audited input changed: {name}")
    inputs = read_json(audit / "audit_inputs.json")
    if sha256(args.campaign / "cohort.csv") != inputs["cohort_sha256"]:
        raise ValueError("Original source cohort changed after audit")
    if sha256(Path(inputs["reserved_scenes_path"])) != inputs["reserved_scenes_sha256"]:
        raise ValueError("Reserved evaluation cohort changed after audit")
    validation = read_json(audit / "validation.json")
    if validation["integrity_status"] != "passed" or validation["leakage_status"] != "passed":
        raise ValueError("Source data audit has not passed")
    source = read_json(audit / "manifest.draft.json")
    inventory = read_json(audit / "inventory.draft.json")
    if source["inventory_sha256"] != sha256(audit / "inventory.draft.json"):
        raise ValueError("Source manifest and inventory disagree")
    eligible = {s["scene_id"] for s in inventory["sequences"]
                if s["split"] == "train" and s["automatic_status"] == "passed"
                and not s["review_flags"] and s["nominal"] is not False}
    with (args.campaign / "cohort.csv").open(newline="") as stream:
        pool = [r for r in csv.DictReader(stream) if r["scene_id"] in eligible
                and r["proposed_split"] == "train"]
    assignments, selected = {}, []
    for split, count in (("train", 30), ("calibration", 5), ("test", 5)):
        chosen = _select(pool, count, args.seed, split)
        assignments.update({r["scene_id"]: split for r in chosen})
        selected.extend({**r, "smoke_split": split} for r in chosen)
        pool = [r for r in pool if r["scene_id"] not in assignments]
    frames = [{**r, "source_split": r["split"], "split": assignments[r["scene_id"]]}
              for r in source["frames"] if r["scene_id"] in assignments]
    manifest = {"schema_version": SCHEMA, "intended_use": PURPOSE, "training_ready": False,
                "camera": source["camera"], "selection_seed": args.seed,
                "selection": "lighting_x_speed_proportional_largest_remainder_SHA256",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_manifest_sha256": sha256(audit / "manifest.draft.json"),
                "cohort_sha256": inputs["cohort_sha256"],
                "reserved_test_scenes": source["reserved_test_scenes"],
                "selected_scenes": selected, "frames": frames,
                "note": "Nominal labels are preserved. Calibration/prediction are engineering checks only."}
    manifest["summary"] = validate_smoke_manifest(manifest)
    write_json(args.output, manifest)
    print(manifest["summary"], flush=True)


def _decode(row: dict) -> np.ndarray:
    payload = Path(row["image_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
        raise ValueError(f"Image changed after audit: {row['image_path']}")
    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(image.convert("RGB").resize((160, 80), Image.Resampling.BOX))
    if hashlib.sha256(pixels.tobytes()).hexdigest() != row["preprocessed_rgb_sha256"]:
        raise ValueError("Preprocessed pixels differ from audited preprocessing")
    return pixels.reshape(-1).copy()


def run(args) -> None:
    import torch
    from .models import Reconstructor

    if args.output.exists():
        raise ValueError(f"Choose a new smoke run directory: {args.output}")
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("This pilot requires a Slurm allocation with working CUDA")
    if args.batch_size < 1 or args.workers < 1:
        raise ValueError("Positive batch size and worker count required")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    manifest = read_json(args.manifest)
    summary = validate_smoke_manifest(manifest)
    properties = torch.cuda.get_device_properties(0)
    environment = {"schema_version": SCHEMA, "intended_use": PURPOSE,
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename,
                   "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                   "gpu": properties.name, "vram_bytes": properties.total_memory,
                   "compute_capability": list(torch.cuda.get_device_capability(0)),
                   "visible_cuda_devices": torch.cuda.device_count(),
                   "manifest_sha256": sha256(args.manifest), "summary": summary,
                   "source_code_sha256": {name: sha256(Path(__file__).with_name(name))
                                          for name in ("smoke.py", "models.py", "pipeline.py")},
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "environment.json", environment)
    print(f"GPU: {properties.name}; visible VRAM {properties.total_memory / 2**30:.1f} GiB", flush=True)
    progress_path = args.output / "progress.json"

    def progress(stage, **values):
        write_json(progress_path, {"schema_version": SCHEMA, "intended_use": PURPOSE,
                                   "stage": stage, "elapsed_seconds": time.perf_counter() - start,
                                   "updated_at_utc": datetime.now(timezone.utc).isoformat(), **values})

    # Cache only within this process. Verify raw JPEG and resized pixel hashes once;
    # normalize identical uint8 pixels on the GPU to avoid repeated full-HD decoding.
    caches, rows_by_split = {}, {}
    decode_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for split in ("train", "calibration", "test"):
            rows = sorted((r for r in manifest["frames"] if r["split"] == split),
                          key=lambda r: (r["sequence_id"], r["now_us"]))
            cache = np.empty((len(rows), 80 * 160 * 3), dtype=np.uint8)
            for index, pixels in enumerate(executor.map(_decode, rows), 1):
                cache[index - 1] = pixels
                if index % 250 == 0 or index == len(rows):
                    progress("decode", split=split, frames_done=index, frames_total=len(rows))
                    print(f"Decoded {split}: {index}/{len(rows)}", flush=True)
            caches[split] = torch.from_numpy(cache)
            rows_by_split[split] = rows
    decode_seconds = time.perf_counter() - decode_start
    print(f"Verified and cached all frames in {decode_seconds:.1f}s", flush=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(args.workers)

    def batch(split, indices):
        return caches[split][indices].to(device).float().div_(255.0)

    def score(model, split):
        model.eval()
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(caches[split]), args.batch_size):
                images = batch(split, slice(offset, offset + args.batch_size))
                reconstruction, _ = model(images)
                errors = (reconstruction - images).square().sum(dim=1).sqrt()
                scores.extend(errors.cpu().tolist())
        if not all(math.isfinite(value) and value > 0 for value in scores):
            raise ValueError("Nonfinite or nonpositive inference score")
        return scores

    reports = []
    for model_name, epochs in (("sae", 1), ("sae", 5), ("vae", 1)):
        tag = f"{model_name}_seed{args.seed}_{epochs}epoch"
        directory = args.output / tag
        directory.mkdir()
        progress("training", model=tag, epoch=0, epochs=epochs)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        config = {"model": model_name, "height": 80, "width": 160, "camera": manifest["camera"],
                  "preprocessing": "rgb_box_resize_0_1_hwc", "vae_inference": "posterior_mean"}
        model = Reconstructor(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        training_start = time.perf_counter()
        history = []
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(caches["train"]), generator=generator)
            total, epoch_start = 0.0, time.perf_counter()
            for offset in range(0, len(order), args.batch_size):
                images = batch("train", order[offset:offset + args.batch_size])
                optimizer.zero_grad(set_to_none=True)
                reconstruction, penalty = model(images)
                loss = (reconstruction - images).square().mean() + penalty
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became nonfinite")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise ValueError("Gradient became nonfinite")
                optimizer.step()
                total += float(loss.detach()) * len(images)
                if offset % (args.batch_size * 50) == 0:
                    progress("training", model=tag, epoch=epoch + 1, epochs=epochs,
                             frames_done=min(offset + args.batch_size, len(order)), frames_total=len(order))
            record = {"epoch": epoch + 1, "training_loss": total / len(order),
                      "seconds": time.perf_counter() - epoch_start}
            history.append(record)
            print(f"{tag} epoch {epoch + 1}/{epochs}: loss={record['training_loss']:.7f}, "
                  f"time={record['seconds']:.1f}s", flush=True)
        training_seconds = time.perf_counter() - training_start
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("Model parameters became nonfinite")
        progress("checkpoint_and_calibration", model=tag)
        # A string schema ensures the production pipeline rejects these checkpoints.
        provenance = {"schema_version": SCHEMA, "intended_use": PURPOSE, "config": config,
                      "manifest_sha256": environment["manifest_sha256"], "seed": args.seed,
                      "epochs": epochs, "batch_size": args.batch_size, "learning_rate": 1e-3,
                      "augmentation": "none", "history": history,
                      "nominal_review_pending": any(r.get("nominal") is None for r in manifest["frames"])}
        checkpoint_path = directory / "smoke_model.pt"
        torch.save({"provenance": provenance,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
                   checkpoint_path)
        test_scores = score(model, "test")
        reloaded = Reconstructor(config).to(device)
        reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"])
        repeated_scores = score(reloaded, "test")
        if test_scores != repeated_scores:
            raise ValueError("Checkpoint reload changed deterministic inference scores")
        calibration_scores = score(reloaded, "calibration")
        calibration_filtered = filtered_scores(rows_by_split["calibration"], calibration_scores, 5, 150_000)
        calibration_usable = [row["score"] for row in calibration_filtered if row["scored"]]
        predictions = filtered_scores(rows_by_split["test"], test_scores, 5, 150_000)
        expected_scored = len(predictions) - 4 * len({row["sequence_id"] for row in predictions})
        if sum(row["scored"] for row in predictions) != expected_scored:
            raise ValueError("Unexpected temporal warmup or missing inference frames")
        calibrations = []
        for epsilon in (0.05, 0.01):
            distribution = fit_threshold(calibration_scores, epsilon)
            threshold = distribution["threshold"]
            calibrations.append({"distribution": distribution,
                                 "raw_exceedance": float(np.mean(np.asarray(calibration_scores) > threshold)),
                                 "filtered_exceedance": float(np.mean(np.asarray(calibration_usable) > threshold))})
        write_json(directory / "smoke_calibration.json", {
            "schema_version": SCHEMA, "intended_use": PURPOSE,
            "checkpoint_sha256": sha256(checkpoint_path), "window": 5, "max_gap_us": 150_000,
            "calibration_frames": len(calibration_scores), "calibrations": calibrations})
        prediction_rows = [{key: row[key] for key in (
            "scene_id", "sequence_id", "clip_id", "now_us", "raw_score", "scored", "score")}
            for row in predictions]
        for row in prediction_rows:
            row["flag_epsilon_005"] = (row["score"] > calibrations[0]["distribution"]["threshold"]
                                       if row["scored"] else None)
        write_json(directory / "smoke_predictions.json", {
            "schema_version": SCHEMA, "intended_use": PURPOSE, "results": prediction_rows})
        report = {**provenance, "checkpoint_reload_identical": True,
                  "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "training_seconds": training_seconds,
                  "training_frames_per_second": len(caches["train"]) * epochs / training_seconds,
                  "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                  "prediction_frames": len(predictions), "scored_prediction_frames": expected_scored,
                  "calibrations": calibrations}
        write_json(directory / "training.json", report)
        reports.append(report)
        print(f"{tag}: checkpoint reload, inference, Gamma fit and temporal warmup passed; "
              f"peak allocated {report['peak_allocated_vram_bytes'] / 2**30:.2f} GiB", flush=True)
        del model, reloaded, optimizer, images, reconstruction, penalty, loss
        torch.cuda.empty_cache()
    result = {**environment, "status": "passed", "decode_seconds": decode_seconds,
              "elapsed_seconds": time.perf_counter() - start, "runs": reports,
              "limitations": ["Nominal review pending; not a paper baseline result.",
                              "No augmentation, validation checkpoint selection or early stopping.",
                              "Training throughput uses an in-process cache, not repeated full-HD JPEG loading.",
                              "Development GT replays do not measure unsafe-event detection."]}
    write_json(args.output / "result.json", result)
    progress("completed", status="passed")
    print(f"All technical training checks passed in {result['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="command", required=True)
    preparation = stages.add_parser("prepare")
    preparation.add_argument("--campaign", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--seed", type=int, default=0)
    execution = stages.add_parser("run")
    execution.add_argument("--manifest", type=Path, required=True)
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--seed", type=int, default=0)
    execution.add_argument("--batch-size", type=int, default=32)
    execution.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        globals()[args.command](args)
    except Exception as exc:
        if args.command == "run" and args.output.is_dir():
            write_json(args.output / "failure.json", {"status": "failed", "intended_use": PURPOSE,
                       "exception": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
