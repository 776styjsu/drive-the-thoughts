#!/usr/bin/env python3
"""Export causal five-frame windows and inputs for the existing monitor runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_analysis import (
    clean_cot,
    cot_is_reliable,
    ground_truth_is_consistent,
    load_json,
)
from benchmark_analysis.loading import load_entries
from cot_analysis.benchmark_source import read_lane_center_lines


def resolve_source(entry: dict, roots: list[Path]) -> Path:
    source = Path(entry["source_scene_file"])
    for root in roots:
        candidate = root / source
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {source} under {roots}")


def prepare(
    benchmark: Path,
    source_benchmark: Path,
    source_roots: list[Path],
    *,
    step_us: int = 100_000,
    require_lane_geometry: bool = False,
    include_unreliable: bool = False,
) -> tuple[dict, list[dict]]:
    """Keep complete windows; never skip a missing decision to extend history."""
    sources = {}
    for entry in load_entries(source_benchmark):
        if entry["clip_id"] in sources:
            raise ValueError(f"Duplicate source clip_id: {entry['clip_id']}")
        sources[entry["clip_id"]] = entry
    roots = [source_benchmark.resolve().parent, *source_roots]
    windows, excluded, frames = [], [], {}
    seen = set()
    entries = load_entries(benchmark)
    targets = [
        entry for entry in entries if include_unreliable or cot_is_reliable(entry)
    ]
    for target in targets:
        clip_id = target["clip_id"]
        if clip_id in seen:
            raise ValueError(f"Duplicate target clip_id: {clip_id}")
        seen.add(clip_id)
        truth = ground_truth_is_consistent(target)
        try:
            if clip_id not in sources:
                raise ValueError("Target absent from source benchmark")
            source = resolve_source(sources[clip_id], roots)
            original = load_json(source)
            released_path = resolve_source(
                target, [benchmark.resolve().parent, Path.cwd()]
            )
            released = load_json(released_path)
            for key in (
                "clip_id",
                "time_now_us",
                "chain_of_thought",
                "trajectory_poses",
            ):
                if original[key] != released[key]:
                    raise ValueError(
                        f"Source target differs from benchmark metadata: {key}"
                    )
            scene_id, suffix = clip_id.rsplit("_t", 1)
            target_index = int(suffix)
            if target_index < 4:
                raise ValueError("Fewer than four preceding frame indices")
            window_frames = []
            for index in range(target_index - 4, target_index + 1):
                frame_id = f"{scene_id}_t{index:0{len(suffix)}d}"
                path = source.parent.parent / frame_id / "metadata.json"
                meta = load_json(path)
                if meta["clip_id"] != frame_id:
                    raise ValueError(f"Metadata clip_id mismatch: {path}")
                if (
                    not clean_cot(meta.get("chain_of_thought")).strip()
                    or len(meta.get("trajectory_poses", [])) < 3
                    or meta.get("time_query_us") is None
                ):
                    raise ValueError(
                        f"Missing CoT, trajectory, or driver request: {frame_id}"
                    )
                expected_time = (
                    original["time_now_us"] - (target_index - index) * step_us
                )
                if meta["time_now_us"] != expected_time:
                    raise ValueError(f"Nonconsecutive timestamps: {frame_id}")
                if require_lane_geometry:
                    geometry = path.with_name("trajectory_plot_geometry.json")
                    if not read_lane_center_lines(geometry):
                        raise ValueError(f"Missing usable lane geometry: {frame_id}")
                window_frames.append(
                    {
                        "clip_id": frame_id,
                        "source_scene_file": str(path.resolve()),
                        "chain_of_thought": meta["chain_of_thought"],
                        "time_now_us": meta["time_now_us"],
                    }
                )
        except (OSError, ValueError, KeyError) as exc:
            excluded.append(
                {"clip_id": clip_id, "is_consistent": truth, "reason": str(exc)}
            )
            continue
        for frame in window_frames:
            previous = frames.get(frame["clip_id"])
            if previous is not None and previous != frame:
                raise ValueError(f"Conflicting source for {frame['clip_id']}")
            frames[frame["clip_id"]] = frame
        windows.append(
            {
                "clip_id": clip_id,
                "scene_id": scene_id,
                "is_consistent": truth,
                "frame_ids": [frame["clip_id"] for frame in window_frames],
                "timestamps_us": [frame["time_now_us"] for frame in window_frames],
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "temporal_consistency_windows",
        "config": {
            "benchmark": str(benchmark.resolve()),
            "source_benchmark": str(source_benchmark.resolve()),
            "source_roots": [str(root.resolve()) for root in roots],
            "reliable_only": not include_unreliable,
            "window_size": 5,
            "step_us": step_us,
            "require_lane_geometry": require_lane_geometry,
        },
        "n_benchmark": len(entries),
        "n_targets": len(targets),
        "n_frames": len(frames),
        "windows": windows,
        "excluded": excluded,
    }
    # No target label or reliability annotation is copied to historical inputs.
    return manifest, list(frames.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark/benchmark.json")
    )
    parser.add_argument("--source-benchmark", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=[],
        help="Additional source path root; repeat as needed. The source benchmark's parent is searched first.",
    )
    parser.add_argument("--step-us", type=int, default=100_000)
    parser.add_argument("--require-lane-geometry", action="store_true")
    parser.add_argument(
        "--include-unreliable",
        action="store_true",
        help="Also build windows for targets whose CoT is labeled unreliable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.step_us <= 0:
        parser.error("--step-us must be positive")
    manifest, frames = prepare(
        args.benchmark,
        args.source_benchmark,
        args.source_root,
        step_us=args.step_us,
        require_lane_geometry=args.require_lane_geometry,
        include_unreliable=args.include_unreliable,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("windows.json", manifest), ("frames.json", frames)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Prepared {len(manifest['windows'])}/{manifest['n_targets']} targets, "
        f"{len(frames)} frames; {len(manifest['excluded'])} exclusions in windows.json."
    )
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
