"""Full SelfOracle training with scene validation and audited RGB caches.

Pending nominal labels remain pending. An explicit opt-in allows provisional
full-scale training without changing the production pipeline's curation gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import UPSTREAM_COMMIT
from .data import read_json, scene_id, sha256, write_json
from .pipeline import filtered_scores, fit_threshold
from .smoke import _decode, _select

SCHEMA = "selforacle_full_training_v1"
SPLITS = ("train", "validation", "calibration", "test")
AUGMENTATION = {
    "probability": 0.6, "space": "BOX_resized_RGB_160x80",
    "horizontal_flip_probability_when_augmented": 0.5,
    "translation_raw_pixel_ranges_xy": [100.0, 10.0],
    "translation_scale": "resized_resolution/source_render_resolution",
    "translation_interpolation": "bilinear_zero_padding",
    "shadow": "random_line_half_plane_HLS_lightness",
    "shadow_lightness_ratio_range": [0.2, 0.5],
    "brightness": "HSV_value", "brightness_ratio_range": [0.8, 1.2],
    "note": "Same transform families/ranges as release; applied after RGB resize, with source-scaled translations.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['clip_id']}\t{row['now_us']}\t{row['image_sha256']}\n".encode())
    return digest.hexdigest()


def validate_full_manifest(manifest: dict, *, allow_pending: bool) -> dict:
    if manifest.get("schema_version") != SCHEMA or manifest.get("nominal_policy") != "audited_GT_driving_pending_nominal_review":
        raise ValueError("Unsupported full-training manifest or nominal policy")
    if manifest.get("research_training_ready") is not False:
        raise ValueError("Provisional manifest must not claim research readiness")
    scenes, sequences, times, paths, counts = {}, {}, set(), set(), Counter()
    reserved = set(manifest["reserved_test_scenes"])
    for row in manifest["frames"]:
        sid, split, sequence = row["scene_id"], row["split"], row["sequence_id"]
        if split not in SPLITS or sid in reserved:
            raise ValueError("Invalid split or reserved benchmark scene in full-training cohort")
        expected_source = "train" if split in ("train", "validation") else split
        if row.get("source_split") != expected_source:
            raise ValueError("Original calibration/test holdout reassigned")
        if scenes.setdefault(sid, split) != split or sequences.setdefault(sequence, sid) != sid:
            raise ValueError("Scene or sequence leakage across full-training splits")
        nominal = row.get("nominal")
        if nominal is False or (nominal is not True and nominal is not None):
            raise ValueError("Rejected or invalid nominal status")
        if nominal is None and not allow_pending:
            raise ValueError("Nominal review is pending; provisional training requires explicit opt-in")
        if scene_id(row["clip_id"]) != sid or type(row["now_us"]) is not int or row["now_us"] < 0:
            raise ValueError("Invalid frame identity or timestamp")
        key, image = (sequence, row["now_us"]), Path(row["image_path"])
        if key in times or image in paths or not image.is_absolute():
            raise ValueError("Duplicate frame or invalid absolute image path")
        if image.name != manifest["camera"] or row["render_width"] <= 0 or row["render_height"] <= 0:
            raise ValueError("Invalid camera or render dimensions")
        times.add(key)
        paths.add(image)
        counts[split] += 1
    if set(counts) != set(SPLITS):
        raise ValueError("All four full-training splits are required")
    by_sequence = {}
    for row in manifest["frames"]:
        by_sequence.setdefault(row["sequence_id"], []).append(row["now_us"])
    for sequence_times in by_sequence.values():
        ordered = sorted(sequence_times)
        if any(b - a != 100_000 for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Full training requires consecutive 10 Hz source streams")
    return {"frames": dict(counts), "scenes": dict(Counter(scenes.values())),
            "pending_nominal_frames": sum(row.get("nominal") is None for row in manifest["frames"])}


def prepare(args) -> None:
    if args.output.exists():
        raise ValueError(f"Choose a new full-training campaign: {args.output}")
    audit = args.campaign / "manifest_audit"
    checksums = {name: digest for digest, name in
                 (line.split(maxsplit=1) for line in (audit / "SHA256SUMS").read_text().splitlines())}
    for name in ("manifest.draft.json", "inventory.draft.json", "validation.json", "audit_inputs.json", "scene_audit.json"):
        if sha256(audit / name) != checksums[name]:
            raise ValueError(f"Source audit changed: {name}")
    validation = read_json(audit / "validation.json")
    inputs = read_json(audit / "audit_inputs.json")
    if validation["integrity_status"] != "passed" or validation["leakage_status"] != "passed":
        raise ValueError("Source audit has not passed")
    if sha256(args.campaign / "cohort.csv") != inputs["cohort_sha256"]:
        raise ValueError("Source cohort changed after audit")
    if sha256(Path(inputs["reserved_scenes_path"])) != inputs["reserved_scenes_sha256"]:
        raise ValueError("Reserved evaluation scenes changed after audit")
    source = read_json(audit / "manifest.draft.json")
    inventory = read_json(audit / "inventory.draft.json")
    if source["inventory_sha256"] != sha256(audit / "inventory.draft.json"):
        raise ValueError("Source manifest/inventory binding changed")
    if len(inventory["sequences"]) != 900 or any(
        s["automatic_status"] != "passed" or s["review_flags"] or s["nominal"] is False
        for s in inventory["sequences"]
    ):
        raise ValueError("Full cohort must contain 900 integrity-passed, unflagged source scenes")
    with (args.campaign / "cohort.csv").open(newline="") as stream:
        cohort = list(csv.DictReader(stream))
    train_pool = [row for row in cohort if row["proposed_split"] == "train"]
    validation_scenes = {row["scene_id"] for row in _select(train_pool, 70, 0, "full_validation")}
    dimensions = {}
    audited_scenes = {s["scene_id"]: s for s in read_json(audit / "scene_audit.json")["scenes"]}
    for sequence in inventory["sequences"]:
        report_path = Path(sequence["frames_dir"]).parent / "render_complete.json"
        if sha256(report_path) != audited_scenes[sequence["scene_id"]]["provenance_sha256"]["render_complete.json"]:
            raise ValueError("Render completion metadata changed after audit")
        report = read_json(report_path)
        dimensions[sequence["scene_id"]] = report["dimensions_wh"]
    rows = []
    for row in source["frames"]:
        split = "validation" if row["scene_id"] in validation_scenes else row["split"]
        width, height = dimensions[row["scene_id"]]
        rows.append({**row, "source_split": row["split"], "split": split,
                     "render_width": width, "render_height": height})
    manifest = {"schema_version": SCHEMA, "nominal_policy": "audited_GT_driving_pending_nominal_review",
                "research_training_ready": False, "camera": source["camera"], "frames": rows,
                "reserved_test_scenes": source["reserved_test_scenes"],
                "source_manifest_sha256": sha256(audit / "manifest.draft.json"),
                "source_campaign": str(args.campaign.resolve()), "created_at_utc": utc_now(),
                "split_method": "original700_train_split_630_train70_validation_lighting_x_speed_SHA256_seed0",
                "source_audit_inputs": inputs}
    manifest["summary"] = validate_full_manifest(manifest, allow_pending=True)
    if manifest["summary"]["scenes"] != {"train": 630, "validation": 70, "calibration": 100, "test": 100}:
        raise ValueError("Unexpected full-training scene counts")
    protocol = {"schema_version": SCHEMA, "created_at_utc": utc_now(),
                "models": ["sae", "vae"], "seeds": [0, 1, 2], "input_wh": [160, 80],
                "batch_size": 32, "optimizer": "Adam", "learning_rate": 0.001,
                "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-8, "weight_decay": 0,
                "max_epochs": 50, "patience": 5, "min_relative_improvement": 0.001,
                "selection_metric": "unaugmented_validation_reconstruction_MSE",
                "augmentation": AUGMENTATION, "vae_KL_weight": 0.0,
                "vae_inference": "posterior_mean", "calibration_epsilon": [0.05, 0.01],
                "temporal_window": 5, "max_gap_us": 150_000,
                "benchmark_outcomes_used_for_selection": False,
                "nominal_review_pending": True, "research_training_ready": False,
                "source_code_sha256": {name: sha256(Path(__file__).with_name(name))
                                       for name in ("full_training.py", "models.py", "pipeline.py", "smoke.py")}}
    write_json(args.output / "manifest.json", manifest)
    protocol["manifest_sha256"] = sha256(args.output / "manifest.json")
    write_json(args.output / "protocol.json", protocol)
    selected = [{**row, "full_training_split": "validation" if row["scene_id"] in validation_scenes
                 else ("test" if row["proposed_split"] == "nominal_test" else row["proposed_split"])} for row in cohort]
    with (args.output / "cohort.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(manifest["summary"], flush=True)


def build_cache(args) -> None:
    if args.output.exists():
        raise ValueError("Choose a new RGB cache directory")
    manifest = read_json(args.manifest)
    validate_full_manifest(manifest, allow_pending=True)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    index = {"schema_version": SCHEMA, "manifest_sha256": sha256(args.manifest),
             "preprocessing": "rgb_box_resize_uint8_hwc_160x80", "splits": {},
             "raw_JPEG_hashes_verified": True, "audited_resized_pixel_hashes_verified": True,
             "created_at_utc": utc_now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for split in SPLITS:
            rows = sorted((row for row in manifest["frames"] if row["split"] == split),
                          key=lambda row: (row["sequence_id"], row["now_us"]))
            path = args.output / f"{split}.npy"
            temporary = path.with_suffix(".npy.tmp")
            cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                             shape=(len(rows), 80 * 160 * 3))
            for number, pixels in enumerate(executor.map(_decode, rows), 1):
                cache[number - 1] = pixels
                if number % 1000 == 0 or number == len(rows):
                    write_json(args.output / "progress.json", {"stage": "cache_build", "split": split,
                               "frames_done": number, "frames_total": len(rows),
                               "elapsed_seconds": time.perf_counter() - start, "updated_at_utc": utc_now()})
                    print(f"Cached {split}: {number}/{len(rows)}", flush=True)
            cache.flush()
            del cache
            temporary.replace(path)
            index["splits"][split] = {"path": str(path.resolve()), "sha256": sha256(path),
                                      "rows_sha256": rows_digest(rows), "frames": len(rows)}
    index["elapsed_seconds"] = time.perf_counter() - start
    write_json(args.output / "index.json", index)
    write_json(args.output / "progress.json", {"stage": "completed", "status": "passed",
               "elapsed_seconds": index["elapsed_seconds"], "updated_at_utc": utc_now()})
    print(f"All RGB cache hashes verified in {index['elapsed_seconds']:.1f}s", flush=True)


class EarlyStopping:
    def __init__(self, patience: int, min_relative: float):
        if patience < 1 or not 0 <= min_relative < 1:
            raise ValueError("Invalid early-stopping settings")
        self.patience, self.min_relative = patience, min_relative
        self.best, self.meaningful_best = math.inf, math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Nonfinite validation reconstruction loss")
        best = value < self.best
        self.best = min(self.best, value)
        if value < self.meaningful_best * (1 - self.min_relative):
            self.meaningful_best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return best, self.bad_epochs >= self.patience


def augment(images, render_sizes, generator, probability=0.6):
    """Batched RGB augmentation; the reconstruction target is the same image."""
    import torch
    import torch.nn.functional as functional

    if not 0 <= probability <= 1:
        raise ValueError("Augmentation probability must lie in [0,1]")
    if probability == 0:
        return images
    n = len(images)
    selected = torch.rand(n, device=images.device, generator=generator) < probability
    original = images.reshape(n, 80, 160, 3).permute(0, 3, 1, 2)
    flip = torch.rand(n, device=images.device, generator=generator) < 0.5
    transformed = torch.where(flip[:, None, None, None], original.flip(-1), original)
    transform = torch.zeros((n, 2, 3), device=images.device)
    transform[:, 0, 0] = transform[:, 1, 1] = 1
    offsets = torch.rand((n, 2), device=images.device, generator=generator) - 0.5
    transform[:, 0, 2] = -2 * 100 * offsets[:, 0] / render_sizes[:, 0]
    transform[:, 1, 2] = -2 * 10 * offsets[:, 1] / render_sizes[:, 1]
    grid = functional.affine_grid(transform, transformed.shape, align_corners=False)
    transformed = functional.grid_sample(transformed, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = torch.linspace(0, 1, 160, device=images.device)[None, None, None, :]
    y = torch.linspace(0, 1, 80, device=images.device)[None, None, :, None]
    endpoints = torch.rand((n, 2), device=images.device, generator=generator)
    x1, x2 = endpoints[:, 0, None, None, None], endpoints[:, 1, None, None, None]
    mask = y * (x2 - x1) - (x - x1) > 0
    side = torch.rand(n, device=images.device, generator=generator) < 0.5
    shadow = mask == side[:, None, None, None]
    ratio = 0.2 + 0.3 * torch.rand(n, device=images.device, generator=generator)
    factor = torch.where(shadow, ratio[:, None, None, None], 1.0)
    # Preserve HLS hue/saturation while changing lightness, without a color library.
    high, low = transformed.amax(1, keepdim=True), transformed.amin(1, keepdim=True)
    chroma = high - low
    lightness = (high + low) / 2
    saturation = chroma / (1 - (2 * lightness - 1).abs()).clamp_min(1e-7)
    new_lightness = lightness * factor
    new_chroma = (1 - (2 * new_lightness - 1).abs()) * saturation
    phase = (transformed - low) / chroma.clamp_min(1e-7)
    transformed = phase * new_chroma + new_lightness - new_chroma / 2
    brightness = 0.8 + 0.4 * torch.rand(n, device=images.device, generator=generator)
    value = transformed.amax(1, keepdim=True)
    new_value = (value * brightness[:, None, None, None]).clamp(0, 1)
    transformed = (transformed * new_value / value.clamp_min(1e-7)).clamp(0, 1)
    output = torch.where(selected[:, None, None, None], transformed, original)
    return output.permute(0, 2, 3, 1).reshape(n, -1)


def train(args) -> None:
    import torch
    from .models import Reconstructor

    if args.output.exists():
        raise ValueError(f"Choose a new model directory: {args.output}")
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Full training requires a Slurm allocation with CUDA")
    manifest, protocol = read_json(args.manifest), read_json(args.protocol)
    summary = validate_full_manifest(manifest, allow_pending=args.allow_pending_nominal)
    manifest_hash = sha256(args.manifest)
    if protocol["manifest_sha256"] != manifest_hash:
        raise ValueError("Protocol belongs to another manifest")
    for name, expected in protocol["source_code_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Training source changed after protocol freeze: {name}")
    index = read_json(args.cache_index)
    if index["schema_version"] != SCHEMA or index["manifest_sha256"] != manifest_hash:
        raise ValueError("RGB cache belongs to another manifest")
    if args.seed not in protocol["seeds"] or args.model not in protocol["models"]:
        raise ValueError("Model/seed not part of the frozen protocol")
    rows_by_split = {split: sorted((row for row in manifest["frames"] if row["split"] == split),
                                 key=lambda row: (row["sequence_id"], row["now_us"])) for split in SPLITS}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    progress_path = args.output / "progress.json"

    def progress(stage, **fields):
        write_json(progress_path, {"stage": stage, "model": args.model, "seed": args.seed,
                   "elapsed_seconds": time.perf_counter() - start, "updated_at_utc": utc_now(), **fields})

    config = {"model": args.model, "height": 80, "width": 160, "camera": manifest["camera"],
              "preprocessing": "rgb_box_resize_0_1_hwc", "score": "l2",
              "vae_inference": "posterior_mean", "seed": args.seed}
    provenance = {"schema_version": SCHEMA, "adaptation": "selforacle_alpasim_full_v1",
                  "upstream_commit": UPSTREAM_COMMIT, "config": config,
                  "result_status": "full_scale_provisional_nominal_review_pending",
                  "research_training_ready": False, "nominal_review_pending": True,
                  "manifest_sha256": manifest_hash, "protocol_sha256": sha256(args.protocol),
                  "cache_index_sha256": sha256(args.cache_index), "summary": summary,
                  "train_scenes": sorted({row["scene_id"] for row in rows_by_split["train"]}),
                  "validation_scenes": sorted({row["scene_id"] for row in rows_by_split["validation"]}),
                  "calibration_scenes": sorted({row["scene_id"] for row in rows_by_split["calibration"]}),
                  "reserved_test_scenes": sorted(set(manifest["reserved_test_scenes"]) |
                                                 {row["scene_id"] for row in rows_by_split["test"]}),
                  "seed": args.seed, "augmentation_seed": 100_000 + args.seed,
                  "batch_size": protocol["batch_size"], "learning_rate": protocol["learning_rate"],
                  "augmentation": protocol["augmentation"], "torch_version": str(torch.__version__),
                  "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename,
                  "max_epochs": protocol["max_epochs"], "created_at_utc": utc_now(),
                  "source_code_sha256": protocol["source_code_sha256"], "history": []}
    write_json(args.output / "training.json", provenance)
    caches = {}

    def load_cache(split):
        progress("load_cache", split=split)
        item, rows = index["splits"][split], rows_by_split[split]
        path = Path(item["path"])
        if rows_digest(rows) != item["rows_sha256"] or sha256(path) != item["sha256"]:
            raise ValueError(f"RGB cache row order/content changed: {split}")
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        if array.dtype != np.uint8 or array.shape != (len(rows), 80 * 160 * 3):
            raise ValueError("Invalid cached RGB dimensions/dtype")
        tensor = torch.from_numpy(array).to(device)
        print(f"Loaded {split} GPU cache: {len(rows)} frames", flush=True)
        return tensor

    for split in ("train", "validation"):
        caches[split] = load_cache(split)
    render_sizes = torch.tensor([[row["render_width"], row["render_height"]]
                                 for row in rows_by_split["train"]], dtype=torch.float32, device=device)
    model = Reconstructor(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"],
                                 betas=tuple(protocol["adam_betas"]), eps=protocol["adam_epsilon"], weight_decay=0)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    augmentation_generator = torch.Generator(device=device).manual_seed(100_000 + args.seed)
    stopping = EarlyStopping(protocol["patience"], protocol["min_relative_improvement"])
    batch_size = protocol["batch_size"]
    checkpoint_path = args.output / "model.pt"
    best_epoch = 0
    torch.cuda.reset_peak_memory_stats()

    def raw_batch(split, indices):
        return caches[split][indices].float().div_(255.0)

    def validation_loss():
        model.eval()
        total = torch.zeros((), device=device)
        with torch.inference_mode():
            for offset in range(0, len(caches["validation"]), batch_size):
                images = raw_batch("validation", slice(offset, offset + batch_size))
                reconstructed, _ = model(images)
                total += (reconstructed - images).square().mean() * len(images)
        return float(total / len(caches["validation"]))

    for epoch in range(1, protocol["max_epochs"] + 1):
        model.train()
        epoch_start = time.perf_counter()
        order = torch.randperm(len(caches["train"]), generator=shuffle_generator).to(device)
        loss_total, mse_total = torch.zeros((), device=device), torch.zeros((), device=device)
        for offset in range(0, len(order), batch_size):
            chosen = order[offset:offset + batch_size]
            images = augment(raw_batch("train", chosen), render_sizes[chosen], augmentation_generator,
                             protocol["augmentation"]["probability"])
            optimizer.zero_grad(set_to_none=True)
            reconstructed, penalty = model(images)
            mse = (reconstructed - images).square().mean()
            loss = mse + penalty
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite")
            loss.backward()
            # Infinite max_norm checks finite gradients without changing their scale.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
            optimizer.step()
            loss_total += loss.detach() * len(images)
            mse_total += mse.detach() * len(images)
            if offset % (batch_size * 500) == 0:
                progress("training", epoch=epoch, frames_done=min(offset + batch_size, len(order)),
                         frames_total=len(order), best_epoch=best_epoch)
        val_start = time.perf_counter()
        validation_mse = validation_loss()
        record = {"epoch": epoch, "training_loss": float(loss_total / len(order)),
                  "training_reconstruction_mse": float(mse_total / len(order)),
                  "validation_reconstruction_mse": validation_mse,
                  "training_seconds": val_start - epoch_start,
                  "validation_seconds": time.perf_counter() - val_start}
        provenance["history"].append(record)
        improved, stop = stopping.update(validation_mse)
        if improved:
            best_epoch = epoch
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"provenance": {**provenance, "epochs": epoch, "selected_epoch": epoch},
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, temporary)
            temporary.replace(checkpoint_path)
        provenance.update(epochs_completed=epoch, selected_epoch=best_epoch,
                          best_validation_reconstruction_mse=stopping.best,
                          early_stopping_bad_epochs=stopping.bad_epochs,
                          peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())
        write_json(args.output / "training.json", provenance)
        progress("epoch_completed", epoch=epoch, best_epoch=best_epoch,
                 training_loss=record["training_loss"], validation_mse=validation_mse,
                 bad_epochs=stopping.bad_epochs)
        print(f"{args.model} seed {args.seed} epoch {epoch}: train={record['training_loss']:.7f}, "
              f"validation={validation_mse:.7f}, best_epoch={best_epoch}, "
              f"bad_epochs={stopping.bad_epochs}/{stopping.patience}", flush=True)
        if stop:
            break
    provenance.update(epochs=best_epoch, early_stopping_triggered=stop,
                      max_epoch_cap_reached=epoch == protocol["max_epochs"] and not stop)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    saved["provenance"] = provenance
    final_temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(saved, final_temporary)
    final_temporary.replace(checkpoint_path)
    del saved, optimizer, caches["train"], caches["validation"], render_sizes, order, images, reconstructed, penalty, loss, mse
    torch.cuda.empty_cache()
    model.eval()

    def scores(split):
        caches[split] = load_cache(split)
        values = []
        with torch.inference_mode():
            for offset in range(0, len(caches[split]), batch_size):
                images = raw_batch(split, slice(offset, offset + batch_size))
                reconstruction, _ = model(images)
                values.extend((reconstruction - images).square().sum(dim=1).sqrt().cpu().tolist())
        del caches[split]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Nonfinite or nonpositive reconstruction score")
        return values

    calibration_scores = scores("calibration")
    calibration_filtered = filtered_scores(rows_by_split["calibration"], calibration_scores, 5, 150_000)
    usable = [row["score"] for row in calibration_filtered if row["scored"]]
    distributions = []
    for epsilon in protocol["calibration_epsilon"]:
        distribution = fit_threshold(calibration_scores, epsilon)
        threshold = distribution["threshold"]
        distributions.append({"distribution": distribution,
                              "empirical_raw_exceedance": float(np.mean(np.asarray(calibration_scores) > threshold)),
                              "empirical_filtered_exceedance": float(np.mean(np.asarray(usable) > threshold))})
    calibration = {"schema_version": SCHEMA, "result_status": provenance["result_status"],
                   "checkpoint_sha256": sha256(checkpoint_path), "manifest_sha256": manifest_hash,
                   "config": config, "window": 5, "max_gap_us": 150_000,
                   "calibration_scenes": provenance["calibration_scenes"],
                   "raw_frames": len(calibration_scores), "scored_frames": len(usable),
                   "calibrations": distributions, "nominal_review_pending": True}
    write_json(args.output / "calibration.json", calibration)
    test_scores = scores("test")
    predictions = filtered_scores(rows_by_split["test"], test_scores, 5, 150_000)
    test_rows = [{key: row[key] for key in ("scene_id", "sequence_id", "clip_id", "now_us", "raw_score", "score", "scored")}
                 for row in predictions]
    nominal_test_exceedance = {}
    for item in distributions:
        epsilon, threshold = item["distribution"]["epsilon"], item["distribution"]["threshold"]
        for row in test_rows:
            row[f"flag_epsilon_{epsilon}"] = row["score"] > threshold if row["scored"] else None
        valid = [row[f"flag_epsilon_{epsilon}"] for row in test_rows if row["scored"]]
        nominal_test_exceedance[str(epsilon)] = float(np.mean(valid))
    write_json(args.output / "nominal_test_predictions.json", {
        "schema_version": SCHEMA, "result_status": provenance["result_status"],
        "checkpoint_sha256": sha256(checkpoint_path), "results": test_rows})
    provenance.update(status="completed", elapsed_seconds=time.perf_counter() - start,
                      nominal_test_filtered_exceedance=nominal_test_exceedance,
                      selected_checkpoint_sha256=sha256(checkpoint_path))
    write_json(args.output / "training.json", provenance)
    progress("completed", status="passed", epochs_completed=epoch, selected_epoch=best_epoch)
    print(f"Completed {args.model} seed {args.seed}; best epoch {best_epoch}/{epoch}; "
          f"calibration and nominal-test scoring saved", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--campaign", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--workers", type=int, default=16)
    training = sub.add_parser("train")
    for name in ("manifest", "protocol", "cache-index", "output"):
        training.add_argument(f"--{name}", type=Path, required=True)
    training.add_argument("--model", choices=("sae", "vae"), required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--allow-pending-nominal", action="store_true")
    args = parser.parse_args()
    try:
        {"prepare": prepare, "cache": build_cache, "train": train}[args.command](args)
    except Exception as exc:
        if args.command in ("train", "cache") and args.output.is_dir():
            write_json(args.output / "failure.json", {"status": "failed", "exception": type(exc).__name__,
                       "message": str(exc), "created_at_utc": utc_now()})
        raise


if __name__ == "__main__":
    main()
