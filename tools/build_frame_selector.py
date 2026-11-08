#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a manual, one-frame-per-scene selection workspace for new benchmark cases.

Run with --help. The exported examples feed build_benchmark_viewer.py --annotator.
No reliability, consistency, or safety verdict is used to prioritize candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from build_benchmark_viewer import FRAME_NAME, REPO_ROOT, clip_index, video_duration_s

from benchmark_analysis import clean_cot, extract_entries, load_json
from benchmark_analysis.schema import cot_decision_label

TAXONOMY = {
    "longitudinal": {
        "none": "No longitudinal decision",
        "set_speed_tracking": "Set speed",
        "lead_obstacle_following": "Lead follow",
        "speed_adaptation_road_events": "Speed adaptation",
        "gap_searching_for_lane_change_merge_zipper": "Gap search",
        "acceleration_for_passing_overtaking": "Overtake acceleration",
        "yield_agent_right_of_way": "Yield",
        "stop_for_static_constraints": "Static stop",
    },
    "lateral": {
        "none": "No lateral decision",
        "lane_keeping_centering": "Lane keep",
        "merge_split_facility_change": "Merge or split",
        "out_of_lane_nudge": "Out-of-lane nudge",
        "in_lane_nudge": "In-lane nudge",
        "lane_change": "Lane change",
        "pull_over_curb_approach": "Pull over or curb approach",
        "turn": "Turn",
        "lateral_maneuver_abort": "Lateral maneuver abort",
    },
    "causal_factors": {
        "critical_objects": "Critical objects",
        "traffic_lights": "Traffic lights",
        "yield_stop_control": "Yield or stop control",
        "traffic_signs": "Traffic signs",
        "road_events": "Road events",
        "lane_lanelines": "Lanes and lane lines",
        "routing_intent": "Routing intent",
        "odd_constraints": "ODD constraints",
    },
}
SCENE_RE = re.compile(r"clipgt-[0-9a-f-]{36}")


def scene_id(value: str) -> str:
    match = SCENE_RE.search(value)
    if match is None:
        raise ValueError(f"No source-scene UUID in {value!r}")
    return match.group()


def cot_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def validate_categories(raw: dict) -> dict:
    if not isinstance(raw, dict) or set(raw) != set(TAXONOMY):
        raise ValueError(
            "Categories must contain longitudinal, lateral, causal_factors"
        )
    result = {}
    for axis in ("longitudinal", "lateral"):
        if raw[axis] not in TAXONOMY[axis]:
            raise ValueError(f"Unknown {axis} category: {raw[axis]!r}")
        result[axis] = raw[axis]
    values = raw["causal_factors"]
    if not isinstance(values, list) or any(
        v not in TAXONOMY["causal_factors"] for v in values
    ):
        raise ValueError(f"Unknown causal categories: {values!r}")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate causal categories")
    result["causal_factors"] = sorted(values)
    return result


def entry_categories(entry: dict) -> dict | None:
    label = cot_decision_label(entry)
    if not label:
        return None
    decision = label["high_level_decision"]
    return validate_categories(
        {
            "longitudinal": decision["longitudinal"],
            "lateral": decision["lateral"],
            "causal_factors": label["critical_component_categories"],
        }
    )


def category_keys(categories: dict) -> list[str]:
    return [f"{axis}:{categories[axis]}" for axis in ("longitudinal", "lateral")] + [
        f"causal_factors:{value}" for value in categories["causal_factors"]
    ]


def load_proposals(path: Path | None) -> dict:
    if path is None:
        return {}
    data = load_json(path)
    if data["schema_version"] != 1:
        raise ValueError("Unsupported CoT label file version")
    result = {}
    for row in data["labels"]:
        key = row["cot_id"]
        if key in result:
            raise ValueError(f"Duplicate CoT label: {key}")
        if not isinstance(row["source"], str) or not row["source"].strip():
            raise ValueError("Each label needs a nonempty source")
        result[key] = {
            "categories": validate_categories(row["categories"]),
            "source": row["source"],
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts",
        type=Path,
        required=True,
        help="Parent of scene_* folders or symlinks, e.g. rollouts_to_annotate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Workspace with index.html, manifests and media symlinks",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=REPO_ROOT / "data/benchmark/benchmark.json"
    )
    parser.add_argument(
        "--exclude-scenes",
        type=Path,
        action="append",
        default=[],
        help="Additional exclusion JSON entries or text paths (e.g. unsafe_traces.txt)",
    )
    parser.add_argument(
        "--cot-labels", type=Path, help="Provisional text-category labels; see tutorial"
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=20,
        help="Editable starting target per substantive category; not a power calculation",
    )
    parser.add_argument("--clip-step-s", type=float, default=0.1)
    parser.add_argument(
        "--video-offset-s",
        type=float,
        default=0.0,
        help="Video seek time = frame index * clip step + this offset",
    )
    parser.add_argument("--title", default="Select new benchmark frames")
    return parser.parse_args()


def build(args) -> dict:
    if (
        args.target_n < 0
        or not math.isfinite(args.clip_step_s)
        or args.clip_step_s <= 0
    ):
        raise ValueError("Target must be nonnegative and clip step positive and finite")
    if not math.isfinite(args.video_offset_s):
        raise ValueError("Video offset must be finite")
    root, output = args.rollouts.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Rollout directory not found: {root}")
    baseline = extract_entries(load_json(args.benchmark))
    baseline_scenes = [scene_id(entry["clip_id"]) for entry in baseline]
    if len(set(baseline_scenes)) != len(baseline_scenes):
        raise ValueError("Baseline contains multiple frames from one scene")
    excluded = set(baseline_scenes)
    for path in args.exclude_scenes:
        if path.suffix == ".json":
            excluded.update(
                scene_id(e["clip_id"]) for e in extract_entries(load_json(path))
            )
        else:
            excluded.update(SCENE_RE.findall(path.read_text()))
    counts = Counter()
    baseline_labels = defaultdict(list)
    missing_baseline = []
    for entry in baseline:
        categories = entry_categories(entry)
        if categories is None:
            missing_baseline.append(entry["clip_id"])
            continue
        counts.update(category_keys(categories))
        baseline_labels[clean_cot(entry["chain_of_thought"])].append(categories)
    proposals = load_proposals(args.cot_labels)
    scenes, requests, metadata_hashes = [], {}, []
    seen_scenes, seen_frames = set(), set()
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or SCENE_RE.search(path.name) is None:
            continue
        sid = scene_id(path.name)
        if sid in seen_scenes:
            raise ValueError(f"Duplicate source scene: {sid}")
        seen_scenes.add(sid)
        scene = {
            "scene_id": sid,
            "name": path.name,
            "source_dir": str(path.resolve()),
            "frames": [],
            "video": None,
            "video_path": None,
            "status": "",
            "skipped_frames": 0,
        }
        scenes.append(scene)
        if sid in excluded:
            scene["status"] = "Excluded: existing benchmark or exclusion list"
            continue
        videos = sorted((path / "rollouts").glob("**/*camera_front_wide_120fov*.mp4"))
        if len(videos) > 1:
            raise ValueError(
                f"Ambiguous front-camera rollout videos in {path}; use one rollout per scene"
            )
        if videos:
            video = videos[0]
            scene["video_path"] = str(video.resolve())
            scene["video"] = quote(
                "media/" + path.name + "/" + str(video.relative_to(path))
            )
            scene["video_duration_s"] = video_duration_s(video)
        candidates = sorted((path / "extracted_frames").rglob("metadata.json"))
        for metadata in candidates:
            raw_bytes = metadata.read_bytes()
            raw = json.loads(raw_bytes)
            frame_id = raw["clip_id"]
            if scene_id(frame_id) != sid or metadata.parent.name != frame_id:
                raise ValueError(f"Frame identity mismatch: {metadata}")
            if frame_id in seen_frames:
                raise ValueError(f"Duplicate decision frame: {frame_id}")
            seen_frames.add(frame_id)
            text = clean_cot(raw.get("chain_of_thought"))
            index = clip_index(frame_id)
            if index is None:
                raise ValueError(f"Missing decision index: {frame_id}")
            image = metadata.parent / FRAME_NAME
            if not text or not raw.get("trajectory_poses") or not image.is_file():
                scene["skipped_frames"] += 1
                continue
            key = cot_id(text)
            categories, label_source = None, "Unclassified"
            if key in proposals:
                categories, label_source = (
                    proposals[key]["categories"],
                    proposals[key]["source"],
                )
            elif entry_categories(raw) is not None:
                categories = entry_categories(raw)
                label_source = "Extracted metadata (provisional)"
            elif text in baseline_labels and all(
                v == baseline_labels[text][0] for v in baseline_labels[text]
            ):
                categories = baseline_labels[text][0]
                label_source = "Exact benchmark CoT text match (provisional)"
            request = requests.setdefault(
                key, {"cot_id": key, "chain_of_thought": text, "frame_ids": []}
            )
            request["frame_ids"].append(frame_id)
            frame = {
                "clip_id": frame_id,
                "cot_id": key,
                "chain_of_thought": text,
                "time_now_us": raw["time_now_us"],
                "video_time_s": round(
                    index * args.clip_step_s + args.video_offset_s, 6
                ),
                "image": quote(
                    "media/" + path.name + "/" + str(image.relative_to(path))
                ),
                "source_scene_file": str(metadata.resolve()),
                "categories": categories,
                "category_source": label_source,
            }
            scene["frames"].append(frame)
            metadata_hashes.append((frame_id, hashlib.sha256(raw_bytes).hexdigest()))
        scene["frames"].sort(key=lambda f: f["time_now_us"])
        times = [f["time_now_us"] for f in scene["frames"]]
        if len(times) != len(set(times)):
            raise ValueError(f"Duplicate decision timestamps in {path}")
        scene["status"] = "Ready" if times else "Needs frame extraction"
        if times and not videos:
            scene["status"] = "Frames available; video missing"
    if not scenes:
        raise ValueError("No scene folders found")
    unknown = proposals.keys() - requests.keys()
    if unknown:
        raise ValueError(f"Label file contains unknown CoT IDs: {sorted(unknown)[:5]}")
    benchmark_hash = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    identity = {
        "benchmark_sha256": benchmark_hash,
        "frames": sorted(metadata_hashes),
        "scenes": sorted(seen_scenes),
        "excluded_scenes": sorted(excluded),
        "root": str(root),
        "clip_step_s": args.clip_step_s,
        "video_offset_s": args.video_offset_s,
    }
    dataset_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    data = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "title": args.title,
        "taxonomy": TAXONOMY,
        "target_n": args.target_n,
        "baseline_counts": dict(counts),
        "baseline_cases": len(baseline),
        "baseline_missing_categories": missing_baseline,
        "benchmark_path": str(args.benchmark.resolve()),
        "benchmark_sha256": benchmark_hash,
        "scenes": scenes,
        "timing_note": "Video times use the extracted frame index; verify alignment against the exact camera image.",
    }
    output.mkdir(parents=True, exist_ok=True)
    media = output / "media"
    media.mkdir(exist_ok=True)
    for scene in scenes:
        if scene["status"].startswith("Excluded"):
            continue
        link = media / scene["name"]
        target = Path(scene["source_dir"])
        if os.path.lexists(link):
            if not link.is_symlink() or link.resolve() != target:
                raise ValueError(
                    f"Media destination conflicts with an existing file: {link}"
                )
        else:
            link.symlink_to(target, target_is_directory=True)
    script_root = Path(__file__).parent
    encoded = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    template = (script_root / "frame_selector_template.html").read_text()
    (output / "index.html").write_text(
        template.replace("__TITLE__", html.escape(args.title)).replace(
            "__DATA__", encoded
        )
    )
    shutil.copyfile(script_root / "frame_selector.js", output / "frame_selector.js")
    for asset in ["frame_selector_tour.js", "frame_selector_tour.css"]:
        shutil.copyfile(script_root / asset, output / asset)
    shutil.copyfile(
        script_root / "frame_selector_tutorial.html", output / "tutorial.html"
    )
    shutil.copyfile(
        script_root.parent / "docs/FRAME_SELECTION_TUTORIAL.md",
        output / "FRAME_SELECTION_TUTORIAL.md",
    )
    (output / "candidate_manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "cot_label_requests.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "taxonomy": TAXONOMY,
                "instruction": "Classify stated CoT intent and causal factors, not reliability or safety. Use none only when no decision is stated. Return schema_version=1 and labels=[{cot_id, source, categories:{longitudinal,lateral,causal_factors}}].",
                "requests": list(requests.values()),
            },
            indent=2,
        )
        + "\n"
    )
    (output / "build_manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(
        f"Built {output / 'index.html'}: {len(scenes)} scenes, "
        f"{sum(len(s['frames']) for s in scenes)} selectable decisions, "
        f"{sum(not s['frames'] for s in scenes)} scenes without selectable frames"
    )
    return data


if __name__ == "__main__":
    build(parse_args())
