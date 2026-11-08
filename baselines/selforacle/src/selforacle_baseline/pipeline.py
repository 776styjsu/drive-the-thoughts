"""Train, calibrate on separate nominal scenes, and score causal camera streams."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import gamma

from . import UPSTREAM_COMMIT
from .data import SCHEMA_VERSION, load_split, read_json, sha256, write_json


def filtered_scores(rows: list[dict], scores: list[float], window: int, max_gap_us: int) -> list[dict]:
    """Trailing mean; full-window warmup resets at sequence boundaries and gaps."""
    if window < 1 or max_gap_us < 1 or len(rows) != len(scores):
        raise ValueError("Invalid temporal settings or score count")
    history = deque(maxlen=window)
    previous = None
    result = []
    for row, raw in zip(rows, scores):
        if not math.isfinite(raw) or raw < 0:
            raise ValueError("Reconstruction scores must be finite and nonnegative")
        if previous is None or row["sequence_id"] != previous["sequence_id"]:
            history.clear()
        else:
            gap = row["now_us"] - previous["now_us"]
            if gap <= 0:
                raise ValueError("Sequence timestamps must increase strictly")
            if gap > max_gap_us:
                history.clear()
        history.append(raw)
        result.append({**row, "raw_score": float(raw), "scored": len(history) == window,
                       "score": float(np.mean(history)) if len(history) == window else None})
        previous = row
    return result


def fit_threshold(scores: list[float], epsilon: float) -> dict:
    values = np.asarray(scores, dtype=np.float64)
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between 0 and 1")
    if len(values) < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Gamma fitting requires at least two finite, strictly positive scores")
    if float(np.std(values)) <= np.finfo(float).eps * float(np.mean(values)):
        raise ValueError("Cannot fit a Gamma distribution to constant scores")
    shape, loc, scale = gamma.fit(values, floc=0)
    threshold = float(gamma.ppf(1 - epsilon, shape, loc=loc, scale=scale))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Gamma fit produced an invalid threshold")
    return {"shape": float(shape), "loc": float(loc), "scale": float(scale),
            "epsilon": epsilon, "threshold": threshold, "fitted_on": "raw_l2_errors"}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Install a backend: uv sync --extra cpu (or --extra cuda)") from exc
    return torch


def _device(name: str):
    torch = _torch()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select --device cpu or install the CUDA extra")
    return torch.device(name)


class Images:
    def __init__(self, rows, config):
        self.rows, self.config = rows, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["image_path"]) as source:
            image = source.convert("RGB").resize(
                (self.config["width"], self.config["height"]), Image.Resampling.BOX)
            # HWC flattening matches the upstream dense architecture's input ordering.
            return (np.asarray(image, dtype=np.float32) / 255.0).reshape(-1).copy()


def train(args) -> None:
    torch = _torch()
    from .models import Reconstructor

    manifest, rows = load_split(args.manifest, "train")
    if min(args.epochs, args.batch_size, args.height, args.width) < 1 or args.learning_rate <= 0:
        raise ValueError("Training dimensions, epochs, batch size, and learning rate must be positive")
    if args.output.exists():
        raise ValueError(f"Choose a new training directory: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    config = {"model": args.model, "height": args.height, "width": args.width,
              "camera": manifest["camera"], "preprocessing": "rgb_box_resize_0_1_hwc",
              "score": "l2", "vae_inference": "posterior_mean", "seed": args.seed}
    model = Reconstructor(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(Images(rows, config), batch_size=args.batch_size,
                                         shuffle=True, generator=generator, num_workers=0)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, penalty = model(images)
            loss = (reconstructed - images).square().mean() + penalty
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
        record = {"epoch": epoch + 1, "training_loss": total / len(rows)}
        history.append(record)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={record['training_loss']:.7f}", flush=True)
    args.output.mkdir(parents=True)
    provenance = {
        "schema_version": SCHEMA_VERSION, "adaptation": "selforacle_alpasim_v1",
        "upstream_commit": UPSTREAM_COMMIT, "config": config,
        "manifest_sha256": sha256(args.manifest),
        "train_scenes": sorted({r["scene_id"] for r in rows}),
        "reserved_test_scenes": sorted(set(manifest.get("reserved_test_scenes", [])) |
                                       {r["scene_id"] for r in manifest["frames"] if r["split"] == "test"}),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "device": str(device), "torch_version": str(torch.__version__), "history": history,
    }
    torch.save({"provenance": provenance,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
               args.output / "model.pt")
    write_json(args.output / "training.json", provenance)


def _score(args, split: str):
    torch = _torch()
    from .models import Reconstructor

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    manifest, rows = load_split(args.manifest, split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = checkpoint["provenance"]
    if provenance["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    config = provenance["config"]
    if manifest["camera"] != config["camera"]:
        raise ValueError("Camera differs from training")
    scenes = {r["scene_id"] for r in rows}
    if scenes & set(provenance["train_scenes"]):
        raise ValueError("Training scene reused for calibration/evaluation")
    if split == "calibration" and scenes & set(provenance["reserved_test_scenes"]):
        raise ValueError("Held-out test scene reused for calibration")
    device = _device(args.device)
    model = Reconstructor(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = torch.utils.data.DataLoader(Images(rows, config), batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
    scores = []
    start = time.perf_counter()
    with torch.inference_mode():
        for images in loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            errors = (reconstructed - images).square().sum(dim=1).sqrt()
            scores.extend(errors.cpu().tolist())
    return rows, scores, provenance, time.perf_counter() - start


def calibrate(args) -> None:
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    rows, raw_scores, provenance, duration = _score(args, "calibration")
    distribution = fit_threshold(raw_scores, args.epsilon)
    filtered = filtered_scores(rows, raw_scores, args.window, args.max_gap_us)
    usable = [r["score"] for r in filtered if r["scored"]]
    if not usable:
        raise ValueError("No complete temporal windows; supply consecutive frames or explicitly use --window 1")
    threshold = distribution["threshold"]
    write_json(args.output, {
        "schema_version": SCHEMA_VERSION, "monitor": "selforacle_alpasim_v1",
        "checkpoint_sha256": sha256(args.checkpoint), "manifest_sha256": sha256(args.manifest),
        "config": provenance["config"], "distribution": distribution,
        "window": args.window, "max_gap_us": args.max_gap_us,
        "calibration_scenes": sorted({r["scene_id"] for r in rows}),
        "raw_frames": len(rows), "scored_frames": len(usable),
        "empirical_raw_exceedance": float(np.mean(np.asarray(raw_scores) > threshold)),
        "empirical_filtered_exceedance": float(np.mean(np.asarray(usable) > threshold)),
        "scoring_seconds": duration,
    })


def predict(args) -> None:
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    calibration = read_json(args.calibration)
    if calibration.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported calibration schema")
    if calibration["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Calibration belongs to a different checkpoint")
    # Check calibration overlap before loading or scoring the model.
    _, test_rows = load_split(args.manifest, "test")
    if {r["scene_id"] for r in test_rows} & set(calibration["calibration_scenes"]):
        raise ValueError("Calibration scene reused for evaluation")
    rows, scores, provenance, duration = _score(args, "test")
    if provenance["config"] != calibration["config"]:
        raise ValueError("Calibration preprocessing/model configuration mismatch")
    threshold = calibration["distribution"]["threshold"]
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid calibration threshold")
    filtered = filtered_scores(rows, scores, calibration["window"], calibration["max_gap_us"])
    output_rows = []
    for row in filtered:
        output_rows.append({key: row[key] for key in (
            "scene_id", "sequence_id", "clip_id", "now_us", "raw_score", "score", "scored")})
        output_rows[-1]["flag"] = row["score"] > threshold if row["scored"] else None
    write_json(args.output, {
        "schema_version": SCHEMA_VERSION, "monitor": "selforacle_alpasim_v1",
        "score_semantics": "higher_is_more_anomalous; not_a_probability_or_consistency_verdict",
        "checkpoint_sha256": sha256(args.checkpoint), "calibration_sha256": sha256(args.calibration),
        "manifest_sha256": sha256(args.manifest), "config": provenance["config"],
        "threshold": threshold, "window": calibration["window"],
        "max_gap_us": calibration["max_gap_us"], "scoring_seconds": duration,
        "batch_size": args.batch_size,
        "results": output_rows,
    })

