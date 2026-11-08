"""Explicit scene-disjoint datasets from AlpaSim's extracted camera frames."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1
SPLITS = ("train", "calibration", "test")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_id(clip_id: str) -> str:
    match = re.fullmatch(r"(.+)_t\d+", clip_id)
    if not match:
        raise ValueError(f"Expected a frame ID ending in _tNNNN: {clip_id}")
    return match[1]


def discover(roots: list[Path], camera: str, holdout: Path | None = None) -> dict:
    """Inventory only: never infer nominal status or a training assignment."""
    if Path(camera).name != camera:
        raise ValueError("camera must be a filename, not a path")
    reserved = set()
    if holdout:
        entries = read_json(holdout)
        if not isinstance(entries, list):
            raise ValueError("Holdout benchmark must be a JSON list of entries")
        reserved = {scene_id(e["clip_id"]) for e in entries}
    groups = {}
    seen = set()
    resolved_directories = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Missing frames root: {root}")
        for metadata in sorted(root.rglob("metadata.json")):
            directory = metadata.parent.parent
            if directory not in resolved_directories:
                resolved_directories[directory] = directory.resolve()
            frames_dir = resolved_directories[directory]
            canonical = frames_dir / metadata.parent.name / metadata.name
            if canonical in seen:
                continue
            seen.add(canonical)
            # Standard extraction names already identify the scene. Defer reading
            # full trajectory metadata until preparation (important on shared FS).
            sid = scene_id(metadata.parent.name)
            key = (str(frames_dir), sid)
            group = groups.setdefault(key, {
                "scene_id": sid,
                "sequence_id": hashlib.sha256(f"{frames_dir}/{sid}".encode()).hexdigest()[:20],
                "frames_dir": str(frames_dir),
                "split": "test" if sid in reserved else "unassigned",
                "nominal": None,
                "frame_count": 0,
                "missing_camera_count": 0,
            })
            group["frame_count"] += 1
            group["missing_camera_count"] += not metadata.with_name(camera).is_file()
    if not groups:
        raise ValueError("No extracted metadata.json files found")
    return {
        "schema_version": SCHEMA_VERSION,
        "camera": camera,
        "reserved_test_scenes": sorted(reserved),
        "holdout_benchmark_sha256": sha256(holdout) if holdout else None,
        "sequences": sorted(groups.values(), key=lambda x: (x["scene_id"], x["sequence_id"])),
    }


def validate_manifest(manifest: dict, *, check_files: bool = True) -> dict:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    if not manifest.get("frames"):
        raise ValueError("Empty frame manifest")
    scene_splits, sequence_scenes, seen, images = {}, {}, set(), set()
    counts = Counter()
    for row in manifest["frames"]:
        split, sid, sequence = row["split"], row["scene_id"], row["sequence_id"]
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")
        if scene_splits.setdefault(sid, split) != split:
            raise ValueError(f"Scene leakage across splits: {sid}")
        if sequence_scenes.setdefault(sequence, sid) != sid:
            raise ValueError(f"Sequence ID reused across scenes: {sequence}")
        if scene_id(row["clip_id"]) != sid:
            raise ValueError("Frame ID and scene ID disagree")
        if sid in manifest.get("reserved_test_scenes", []) and split != "test":
            raise ValueError(f"Reserved evaluation scene used for {split}: {sid}")
        if split in ("train", "calibration") and row.get("nominal") is not True:
            raise ValueError(f"{split} requires explicit nominal=true: {sid}")
        timestamp = row["now_us"]
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("now_us must be a nonnegative integer")
        key = (sequence, timestamp)
        if key in seen:
            raise ValueError(f"Duplicate sequence/timestamp: {key}")
        seen.add(key)
        image = Path(row["image_path"])
        if not image.is_absolute():
            raise ValueError("Manifest image paths must be absolute")
        resolved = image.resolve()
        if resolved in images:
            raise ValueError(f"Duplicate image: {resolved}")
        images.add(resolved)
        if check_files and not image.is_file():
            raise ValueError(f"Missing image: {image}")
        counts[split] += 1
    return {"frames": dict(counts), "scenes": dict(Counter(scene_splits.values())),
            "sequences": len(sequence_scenes)}


def prepare(inventory_path: Path) -> dict:
    inventory = read_json(inventory_path)
    if inventory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported inventory schema")
    camera = inventory["camera"]
    if Path(camera).name != camera:
        raise ValueError("camera must be a filename")
    rows = []
    seen_sequences = set()
    metadata_cache = {}
    for sequence in inventory["sequences"]:
        if sequence["split"] == "exclude":
            continue
        if sequence["split"] not in SPLITS:
            raise ValueError(f"Assign train/calibration/test/exclude to {sequence['scene_id']}")
        if sequence["sequence_id"] in seen_sequences:
            raise ValueError(f"Duplicate inventory sequence: {sequence['sequence_id']}")
        seen_sequences.add(sequence["sequence_id"])
        directory = Path(sequence["frames_dir"])
        if not directory.is_absolute():
            directory = inventory_path.resolve().parent / directory
        # Only inspect this sequence's frame directories, including for flattened
        # artifact layouts where many scenes share a single frames_dir.
        metadata_paths = metadata_cache.setdefault(directory, None)
        if metadata_paths is None:
            metadata_paths = {}
            for path in sorted(directory.glob("*/metadata.json")):
                metadata_paths.setdefault(scene_id(path.parent.name), []).append(path)
            metadata_cache[directory] = metadata_paths
        matched = 0
        for metadata in metadata_paths.get(sequence["scene_id"], []):
            meta = read_json(metadata)
            if meta["clip_id"] != metadata.parent.name:
                raise ValueError(f"Frame directory and metadata clip_id disagree: {metadata}")
            matched += 1
            # time_now_us is the timestamp used to associate the extracted camera image.
            timestamp = meta["time_now_us"]
            rows.append({
                "scene_id": sequence["scene_id"], "sequence_id": sequence["sequence_id"],
                "clip_id": meta["clip_id"], "now_us": timestamp,
                "image_path": str(metadata.with_name(camera).resolve()),
                "image_sha256": sha256(metadata.with_name(camera)),
                "split": sequence["split"], "nominal": sequence.get("nominal"),
            })
        if not matched:
            raise ValueError(f"No frames for {sequence['scene_id']} in {directory}")
    manifest = {
        "schema_version": SCHEMA_VERSION, "camera": camera,
        "inventory_sha256": sha256(inventory_path),
        "reserved_test_scenes": inventory.get("reserved_test_scenes", []),
        "frames": sorted(rows, key=lambda r: (r["sequence_id"], r["now_us"])),
    }
    manifest["summary"] = validate_manifest(manifest)
    return manifest


def load_split(path: Path, split: str) -> tuple[dict, list[dict]]:
    manifest = read_json(path)
    validate_manifest(manifest)
    rows = sorted((r for r in manifest["frames"] if r["split"] == split),
                  key=lambda r: (r["sequence_id"], r["now_us"]))
    if not rows:
        raise ValueError(f"Manifest contains no {split} frames")
    for row in rows:
        if sha256(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError(f"Image changed since preparation: {row['image_path']}")
    return manifest, rows

