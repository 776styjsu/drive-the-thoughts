"""Audit the pinned GT cohort without inferring nominal labels or starting training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CAMERA = "camera_front_wide_120fov.jpg"
SPLITS = {"train": "train", "calibration": "calibration", "nominal_test": "test"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, compact=False):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=None if compact else 2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def load_inputs(campaign, reserved_path):
    with (campaign / "cohort.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with reserved_path.open(newline="") as stream:
        reserved = sorted({r["scene_id"] for r in csv.DictReader(stream)})
    if not rows or not reserved:
        raise ValueError("Cohort and reserved scene list must both be nonempty")
    for key in ("scene_id", "uuid", "path"):
        values = [r[key] for r in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate cohort {key}")
    overlap = sorted({r["scene_id"] for r in rows} & set(reserved))
    if overlap:
        raise ValueError(f"Reserved evaluation scenes in nominal cohort: {overlap}")
    for row in rows:
        if row["proposed_split"] not in SPLITS:
            raise ValueError(f"Invalid cohort split: {row['proposed_split']}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["hf_revision"]):
            raise ValueError("Source revision must be a pinned commit")
        if row["scene_id"] != "clipgt-" + Path(row["path"]).stem:
            raise ValueError("Source asset path and scene ID disagree")
    return rows, reserved


def audit_scene(row, campaign, output, minimum_frames=140,
                width_bounds=(1880, 1930), expected_height=1080):
    sid = row["scene_id"]
    directory = campaign / "scenes" / sid
    frames_dir = directory / "extracted_frames"
    sequence = digest(f"{frames_dir.resolve()}/{sid}".encode())[:20]
    errors, flags, frames, measurements, thumbs = [], [], [], [], {}
    provenance = {}
    report = {}
    fingerprint = None
    try:
        report = read_json(directory / "render_complete.json")
        request = read_json(directory / "render_request.json")
        for field, expected in {"scene_id": sid, "artifact_uuid": row["uuid"],
                                "hf_revision": row["hf_revision"],
                                "proposed_split": row["proposed_split"],
                                "cadence_us": 100_000, "driver_calls": 0}.items():
            if report.get(field) != expected:
                errors.append(f"Completion report mismatch: {field}")
        if request.get("source") != row:
            errors.append("Render request differs from pinned cohort row")
        for key in ("source_usdz_sha256", "source_asl_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", report.get(key, "")):
                errors.append(f"Missing/invalid {key}")
        rig = read_json(directory / "rig_trajectories.json")
        trajectories = rig["rig_trajectories"]
        if len(trajectories) != 1 or trajectories[0]["sequence_id"] != sid:
            errors.append("GT trajectory source identity mismatch")
        trajectory = trajectories[0]
        # Exclude reconstruction/scene names: identical retained GT recordings
        # under different identifiers should still have the same fingerprint.
        signature = {"T_world_base": rig["T_world_base"],
                     "timestamps": trajectory["T_rig_world_timestamps_us"],
                     "poses": trajectory["T_rig_worlds"]}
        fingerprint = digest(json.dumps(signature, sort_keys=True,
                                        separators=(",", ":")).encode())
        gt_times = signature["timestamps"]
        if not gt_times or len(gt_times) != len(signature["poses"]):
            errors.append("Invalid retained GT trajectory")
        for name in ("render_complete.json", "render_request.json", "metadata.yaml",
                     "rig_trajectories.json", "datasource_summary.json"):
            provenance[name] = file_hash(directory / name)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        errors.append(f"Source provenance: {type(exc).__name__}: {exc}")

    paths = sorted(frames_dir.glob("*/metadata.json"))
    indices = []
    if len(paths) < minimum_frames:
        errors.append(f"Too few frames: {len(paths)} < {minimum_frames}")
    if report.get("frames") != len(paths):
        errors.append("Frame count differs from completion report")
    selected = {round(i * (len(paths) - 1) / 4) for i in range(5)} if paths else set()
    for index, metadata in enumerate(paths):
        clip = metadata.parent.name
        try:
            match = re.fullmatch(re.escape(sid) + r"_t(\d+)", clip)
            if not match:
                raise ValueError("Frame directory has wrong scene ID")
            indices.append(int(match[1]))
            meta = read_json(metadata)
            if meta.get("clip_id") != clip or meta.get("source") != "alpasim_full_gt_replay":
                raise ValueError("Frame metadata identity/source mismatch")
            timestamp = meta["time_now_us"]
            if type(timestamp) is not int or timestamp < 0:
                raise ValueError("Timestamp must be a nonnegative integer")
            image_path = metadata.with_name(CAMERA)
            payload = image_path.read_bytes()
            image_hash = digest(payload)
            if image_hash != meta.get("image_sha256"):
                raise ValueError("JPEG hash differs from extraction metadata")
            with Image.open(io.BytesIO(payload)) as image:
                image.load()  # Decode the full JPEG, including its final scanlines.
                if image.format != "JPEG" or image.mode != "RGB":
                    raise ValueError(f"Expected RGB JPEG, got {image.format}/{image.mode}")
                width, height = image.size
                if height != expected_height or not width_bounds[0] <= width <= width_bounds[1]:
                    raise ValueError(f"Unexpected camera dimensions: {image.size}")
                if report.get("dimensions_wh") != [width, height]:
                    raise ValueError("Dimensions differ from completion report")
                reduced = image.resize((160, 80), Image.Resampling.BOX)
                pixels = np.asarray(reduced)
                pixels_hash = digest(pixels.tobytes())
                gray = np.asarray(reduced.convert("L"))
                mean, std = float(gray.mean()), float(gray.std())
                black, white = float((gray <= 5).mean()), float((gray >= 250).mean())
                if index in selected:
                    thumb = image.copy()
                    thumb.thumbnail((320, 180), Image.Resampling.LANCZOS)
                    thumbs[index] = (thumb, clip, timestamp)
            frames.append({"scene_id": sid, "sequence_id": sequence, "clip_id": clip,
                           "now_us": timestamp, "image_path": str(image_path.resolve()),
                           "image_sha256": image_hash, "preprocessed_rgb_sha256": pixels_hash,
                           "split": SPLITS[row["proposed_split"]], "nominal": None})
            measurements.append({"clip_id": clip, "now_us": timestamp, "width": width,
                                 "height": height, "mean_luma": round(mean, 4),
                                 "std_luma": round(std, 4), "near_black_fraction": round(black, 6),
                                 "near_white_fraction": round(white, 6),
                                 "image_sha256": image_hash, "preprocessed_rgb_sha256": pixels_hash})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{clip}: {type(exc).__name__}: {exc}")
    if indices != list(range(len(paths))):
        errors.append("Frame indices are not contiguous from zero")
    times = [r["now_us"] for r in frames]
    if times:
        if any(b - a != 100_000 for a, b in zip(times, times[1:])):
            errors.append("Timestamps are not consecutive at 10 Hz")
        if (times[0] != report.get("first_timestamp_us") or
                times[-1] != report.get("last_timestamp_us")):
            errors.append("Timestamp range differs from completion report")
        if fingerprint and gt_times and (times[0] < min(gt_times) or times[-1] > max(gt_times)):
            errors.append("Camera timestamps fall outside retained GT trajectory")
    if measurements:
        for name, condition in (
            ("very_dark_frames", lambda r: r["mean_luma"] < 12),
            ("nearly_black_frames", lambda r: r["near_black_fraction"] > .98),
            ("nearly_white_frames", lambda r: r["near_white_fraction"] > .98),
            ("nearly_uniform_frames", lambda r: r["std_luma"] < 2),
        ):
            if any(condition(r) for r in measurements):
                flags.append(name)
    if errors:
        flags.insert(0, "integrity_failure")
    sheet = Image.new("RGB", (1600, 250), "#151a22")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{sid} | {row['proposed_split']} | {row.get('time_of_day', '')} | "
              f"review: {', '.join(flags) or 'no automatic flags'}", fill="white")
    for column, index in enumerate(sorted(selected)):
        x = column * 320
        if index in thumbs:
            thumbnail, clip, timestamp = thumbs[index]
            sheet.paste(thumbnail, (x, 40))
            draw.text((x + 5, 225), f"frame {index} | {timestamp} us", fill="white")
        else:
            draw.text((x + 10, 100), "FRAME UNAVAILABLE", fill="#ff7777")
    sheet_path = output / "contact_sheets" / f"{sid}.jpg"
    sheet.save(sheet_path, quality=85)
    scene = {"scene_id": sid, "sequence_id": sequence, "frames_dir": str(frames_dir.resolve()),
             "split": SPLITS[row["proposed_split"]], "proposed_split": row["proposed_split"],
             "nominal": None, "nominal_status": "needs_visual_and_render_review",
             "frame_count": len(paths), "validated_frame_count": len(frames),
             "automatic_status": "passed" if not errors else "failed",
             "errors": errors, "review_flags": flags,
             "contact_sheet": str(sheet_path.relative_to(output)),
             "source": row, "source_usdz_sha256": report.get("source_usdz_sha256"),
             "source_asl_sha256": report.get("source_asl_sha256"),
             "gt_recording_fingerprint": fingerprint, "provenance_sha256": provenance,
             "luma_range": [min(r["mean_luma"] for r in measurements),
                            max(r["mean_luma"] for r in measurements)] if measurements else None}
    return scene, frames, measurements


def duplicate_groups(records, field, *, frame_records):
    groups = defaultdict(list)
    for record in records:
        if record.get(field):
            groups[record[field]].append(record)
    result = []
    for signature, members in groups.items():
        if len(members) > 1:
            splits = sorted({m["split"] for m in members})
            result.append({"sha256": signature, "cross_split": len(splits) > 1,
                           "splits": splits, "members": [
                               {k: m[k] for k in (("scene_id", "clip_id", "split") if frame_records
                                                  else ("scene_id", "split"))} for m in members]})
    return sorted(result, key=lambda r: (not r["cross_split"], r["sha256"]))


def write_gallery(output, scenes):
    data = [{k: s[k] for k in ("scene_id", "split", "review_flags", "contact_sheet",
                              "automatic_status", "frame_count")} for s in scenes]
    template = '''<!doctype html><html><head><meta charset="utf-8"><title>SelfOracle clip review</title>
<style>body{font:16px system-ui;background:#111827;color:#eee;margin:24px}input,select,button{padding:8px;margin:4px}img{width:100%;max-width:1600px}article{margin:20px 0;border-top:1px solid #445;padding-top:12px}small{color:#cbd5e1}</style></head><body>
<h1>SelfOracle nominal-data review</h1><p>All nominal labels are pending. Check rendering quality and the intended nominal visual domain. Darkness and uniform images are review flags; they do not automatically exclude a clip. Record decisions in review_decisions.csv.</p>
<input id="search" placeholder="Search scene ID"><select id="split"><option value="">All splits</option><option>train</option><option>calibration</option><option>test</option></select><label><input id="flagged" type="checkbox">Only flagged clips</label>
<button id="prev">Previous</button><button id="next">Next</button><span id="count"></span><main id="clips"></main>
<script>const rows=DATA;let page=0;const size=30;const get=id=>document.getElementById(id);
function render(){let filtered=rows.filter(r=>r.scene_id.includes(get('search').value)&&(!get('split').value||r.split===get('split').value)&&(!get('flagged').checked||r.review_flags.length));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/size)-1));get('count').textContent=`${filtered.length} clips; page ${page+1} of ${Math.max(1,Math.ceil(filtered.length/size))}`;get('clips').replaceChildren();for(const r of filtered.slice(page*size,(page+1)*size)){const a=document.createElement('article'),p=document.createElement('p'),im=document.createElement('img');p.textContent=`${r.scene_id} | ${r.split} | ${r.frame_count} frames | ${r.automatic_status} | ${r.review_flags.join(', ')||'no automatic flags'}`;im.src=r.contact_sheet;im.loading='lazy';im.alt=r.scene_id;a.append(p,im);get('clips').append(a)}}
for(const id of ['search','split','flagged'])get(id).addEventListener('input',()=>{page=0;render()});get('prev').onclick=()=>{page--;render()};get('next').onclick=()=>{page++;render()};render();</script></body></html>'''
    (output / "review.html").write_text(template.replace("DATA", json.dumps(data).replace("<", "\\u003c")))


def run(campaign, reserved_path, output, workers):
    rows, reserved = load_inputs(campaign, reserved_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "contact_sheets").mkdir()
    started = time.monotonic()
    inputs = {"cohort_sha256": file_hash(campaign / "cohort.csv"),
              "reserved_scenes_sha256": file_hash(reserved_path),
              "audit_tool_sha256": file_hash(Path(__file__)),
              "campaign": str(campaign), "reserved_scenes_path": str(reserved_path),
              "camera": CAMERA, "workers": workers,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(output / "audit_inputs.json", inputs)
    scenes, frames = [], []
    with (output / "frame_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "now_us", "width", "height",
            "mean_luma", "std_luma", "near_black_fraction", "near_white_fraction",
            "image_sha256", "preprocessed_rgb_sha256"])
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(audit_scene, row, campaign, output): row for row in rows}
            for future in as_completed(futures):
                scene, sequence_frames, measures = future.result()
                scenes.append(scene)
                frames.extend(sequence_frames)
                writer.writerows(measures)
                if len(scenes) % 10 == 0 or len(scenes) == len(rows):
                    progress = {"status": "running", "completed_scenes": len(scenes),
                        "total_scenes": len(rows), "validated_frames": len(frames),
                        "failed_scenes": sum(s["automatic_status"] != "passed" for s in scenes),
                        "elapsed_seconds": round(time.monotonic() - started, 1)}
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    scenes.sort(key=lambda s: s["scene_id"])
    frames.sort(key=lambda f: (f["sequence_id"], f["now_us"]))
    duplicates = {
        "raw_jpeg": duplicate_groups(frames, "image_sha256", frame_records=True),
        "preprocessed_rgb_160x80": duplicate_groups(frames, "preprocessed_rgb_sha256", frame_records=True),
        "source_usdz": duplicate_groups(scenes, "source_usdz_sha256", frame_records=False),
        "gt_recording": duplicate_groups(scenes, "gt_recording_fingerprint", frame_records=False)}
    cross_duplicates = {k: sum(g["cross_split"] for g in v) for k, v in duplicates.items()}
    global_errors = []
    if len({f["image_path"] for f in frames}) != len(frames):
        global_errors.append("Duplicate resolved image paths")
    if len({f["clip_id"] for f in frames}) != len(frames):
        global_errors.append("Duplicate frame IDs")
    if any(cross_duplicates[k] for k in ("source_usdz", "gt_recording")):
        global_errors.append("Matching source assets/GT recordings across splits")
    if file_hash(campaign / "cohort.csv") != inputs["cohort_sha256"]:
        global_errors.append("Cohort changed during audit")
    if file_hash(reserved_path) != inputs["reserved_scenes_sha256"]:
        global_errors.append("Reserved scene list changed during audit")
    # Detect concurrent changes to source reports/configs after workers read them.
    for scene in scenes:
        for name, expected in scene["provenance_sha256"].items():
            if file_hash(campaign / "scenes" / scene["scene_id"] / name) != expected:
                global_errors.append(f"Provenance changed during audit: {scene['scene_id']}/{name}")
    failed = [s for s in scenes if s["errors"]]
    summary = {"frames": dict(Counter(f["split"] for f in frames)),
               "scenes": dict(Counter(s["split"] for s in scenes)), "sequences": len(scenes)}
    inventory = {"schema_version": 1, "status": "draft_pending_visual_review", "camera": CAMERA,
                 "reserved_test_scenes": reserved, "cohort_sha256": inputs["cohort_sha256"],
                 "reserved_scenes_sha256": inputs["reserved_scenes_sha256"],
                 "sequences": [{k: s[k] for k in ("scene_id", "sequence_id", "frames_dir", "split",
                     "nominal", "nominal_status", "frame_count", "automatic_status", "review_flags")}
                     for s in scenes]}
    write_json(output / "inventory.draft.json", inventory)
    manifest = {"schema_version": 1, "status": "draft_pending_visual_review", "training_ready": False,
                "camera": CAMERA, "inventory_sha256": file_hash(output / "inventory.draft.json"),
                "reserved_test_scenes": reserved, "frames": frames, "summary": summary}
    write_json(output / "manifest.draft.json", manifest, compact=True)
    write_json(output / "scene_audit.json", {"inputs": inputs, "scenes": scenes})
    write_json(output / "duplicate_groups.json", duplicates)
    with (output / "review_decisions.csv").open("w", newline="") as stream:
        columns = ["scene_id", "proposed_split", "nominal_decision", "reason", "reviewer",
                   "reviewed_at", "automatic_status", "review_flags", "contact_sheet"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for scene in scenes:
            writer.writerow({**{k: scene[k] for k in ("scene_id", "proposed_split", "automatic_status",
                                                      "contact_sheet")},
                             "nominal_decision": "pending", "reason": "", "reviewer": "",
                             "reviewed_at": "", "review_flags": ";".join(scene["review_flags"])})
    write_gallery(output, scenes)
    integrity_passed = not failed and not global_errors
    validation = {"integrity_status": "passed" if integrity_passed else "failed",
        "leakage_status": "failed" if global_errors else
                          "needs_duplicate_review" if any(cross_duplicates.values()) else "passed",
        "training_ready": False, "nominal_labels_pending": len(scenes),
        "summary": summary, "reserved_scene_count": len(reserved), "reserved_scene_overlap": 0,
        "failed_scene_count": len(failed), "failed_scenes": [{"scene_id": s["scene_id"],
            "errors": s["errors"]} for s in failed], "global_errors": global_errors,
        "review_flag_counts": dict(Counter(f for s in scenes for f in s["review_flags"])),
        "duplicate_group_counts": {k: len(v) for k, v in duplicates.items()},
        "cross_split_duplicate_group_counts": cross_duplicates,
        "preprocessing_for_duplicate_check": "RGB BOX resize to 160x80, uint8 HWC before /255",
        "quality_flag_policy": "Inspection aids only; no automated nominal labels or exclusions",
        "limitations": ["No visual/domain suitability decision is inferred by this audit.",
            "Different scene IDs with partially overlapping original drive logs cannot be ruled out without original log IDs/time ranges; exact retained GT fingerprints are checked.",
            "Reserved benchmark exclusions are verified by source scene ID; their source images/GT fingerprints are not available in this campaign."],
        "elapsed_seconds": round(time.monotonic() - started, 1), "inputs": inputs}
    write_json(output / "validation.json", validation)
    (output / "dataset_card.md").write_text(
        "# SelfOracle draft GT corpus\n\n"
        f"{len(scenes)} source scenes, {len(frames):,} validated front-wide RGB JPEG frames.\n\n"
        f"Frames by split: {summary['frames']}. Scenes by split: {summary['scenes']}.\n\n"
        "Camera streams follow the recorded GT trajectory at 10 Hz; driving-policy calls must be zero. "
        "Source scene, reconstruction UUID, pinned Hugging Face revision, source USDZ/ASL hashes, "
        "and retained configuration hashes are recorded in scene_audit.json.\n\n"
        f"All {len(reserved)} reserved evaluation source scenes are excluded. Split assignments "
        "come from the preselected cohort; nominal_test is represented as test in the manifest.\n\n"
        "Images are fully decoded, hashed against extraction metadata, and checked against "
        "completion counts, dimensions, timestamp bounds and consecutive 100 ms cadence. "
        "Raw JPEG hashes, RGB BOX-resized 160x80 pixel hashes, source asset hashes and exact "
        "retained GT recording fingerprints are checked for duplicates.\n\n"
        "The resized images are used only for auditing; raw JPEGs remain the training inputs. "
        "The current model loader converts RGB, BOX-resizes to 160x80, scales to [0,1], "
        "and flattens HWC. No augmentation or frame subsampling is performed by this audit.\n\n"
        "All nominal labels remain pending. The draft manifest is deliberately rejected by "
        "the training loader. Open review.html, inspect the contact sheets and any flagged "
        "clips, and record decisions in review_decisions.csv before preparing a frozen manifest. "
        "Ordinary unsafe driving is not automatically visual OOD. Darkness/uniformity flags "
        "are inspection aids and never automatic exclusions.\n\n"
        "Distinct scene IDs may share partially overlapping original drive logs; this cannot "
        "be fully checked without original log identifiers/time ranges. Exact retained GT "
        "duplicates and source scene overlap are checked.\n")
    write_json(output / "progress.json", {"status": "complete" if integrity_passed else "failed",
        "completed_scenes": len(scenes), "total_scenes": len(rows), "validated_frames": len(frames),
        "failed_scenes": len(failed), "elapsed_seconds": validation["elapsed_seconds"]})
    paths = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("w") as stream:
        for path in paths:
            stream.write(f"{file_hash(path)}  {path.relative_to(output)}\n")
    print(json.dumps(validation, indent=2), flush=True)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--reserved-scenes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    result = run(args.campaign.resolve(), args.reserved_scenes.resolve(), args.output.resolve(), args.workers)
    raise SystemExit(0 if result["integrity_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
