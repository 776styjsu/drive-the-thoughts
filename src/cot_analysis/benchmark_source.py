# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Benchmark-driven input for the CoT-consistency judge.

The released benchmark stores curated ``clip_id`` entries plus
``source_scene_file`` pointers into per-clip scene directories
(``data/media/scenes/<clip_id>/``), not raw trajectories. This module resolves
those pointers, loads trajectories and lane geometry from the per-clip
``metadata.json`` / ``trajectory_plot_geometry.json`` / ``additional_info.json``
files, and reuses trajectory features cached in referenced judge reports when
recomputation is unnecessary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from alpasim_utils.lane_projection import lane_center_lines_from_geometry
from benchmark_analysis import (
    cot_is_unreliable,
    extract_entries,
    load_json,
    reliability_justification,
    unreliability_taxonomy,
)

logger = logging.getLogger(__name__)


def resolve_benchmark_source_path(
    source_scene_file: str | None,
    benchmark_path: Path,
    source_root: Path,
) -> Path | None:
    """Resolve a benchmark ``source_scene_file`` against likely local roots.

    Absolute paths are used as-is; relative paths are tried against the
    working directory, the ``--benchmark_source_root``, and the benchmark
    file's own directory.
    """
    if not source_scene_file:
        return None

    source_path = Path(source_scene_file)
    if source_path.is_absolute():
        candidates = [source_path]
    else:
        candidates = [
            Path.cwd() / source_path,
            source_root / source_path,
            benchmark_path.parent / source_path,
        ]

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def read_lane_center_lines(geometry_path: Path) -> list[np.ndarray] | None:
    """Read road lane-center polylines from one trajectory geometry JSON."""
    try:
        with geometry_path.open("r") as f:
            geometry = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping invalid geometry file %s: %s", geometry_path, exc)
        return None
    return lane_center_lines_from_geometry(geometry)


def _metadata_trajectory_to_xy(
    metadata_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """Load metadata.json trajectory poses as relative and map-frame XY points."""
    if not metadata_path.exists():
        return None, None, None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    poses = metadata.get("trajectory_poses") or []
    if len(poses) < 3:
        return None, None, metadata.get("time_now_us")

    map_xy = np.asarray([[pose["x"], pose["y"]] for pose in poses], dtype=float)
    relative_xy = map_xy - map_xy[0]
    return relative_xy, map_xy, metadata.get("time_now_us")


def _find_clip_file(
    source_path: Path | None,
    metadata_path: Path | None,
    clip_id: str | None,
    filename: str,
) -> Path | None:
    """Find a per-clip file adjacent to the source report or metadata.json.

    Also probes ``<parent>/rollouts/frames/<clip_id>/<filename>`` for source
    reports that live outside the extracted frame directory.
    """
    candidates = []
    if metadata_path is not None:
        candidates.append(metadata_path.with_name(filename))
    if source_path is not None:
        candidates.append(source_path.with_name(filename))
        if clip_id:
            for parent in source_path.parents:
                candidates.append(parent / "rollouts" / "frames" / clip_id / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_additional_info(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read additional trajectory info from %s: %s", path, exc
        )
        return None
    return payload if isinstance(payload, dict) else None


def _index_source_report(source_path: Path) -> dict[str, dict]:
    """Map clip_id -> result for every result in a referenced judge report."""
    source_report = load_json(source_path)
    if isinstance(source_report, dict) and source_report.get("clip_id"):
        source_results = [source_report]
    elif isinstance(source_report, dict):
        source_results = source_report.get("results", [])
    else:
        source_results = source_report
    if not isinstance(source_results, list):
        source_results = []
    return {
        result.get("clip_id"): result
        for result in source_results
        if isinstance(result, dict) and result.get("clip_id")
    }


def extract_entries_from_benchmark(
    benchmark_json: str,
    source_root: str = ".",
    load_raw_trajectory: bool = False,
) -> list[dict]:
    """Load benchmark-selected entries, reusing cached trajectory features.

    The source judge reports referenced by ``source_scene_file`` already
    include ``trajectory_features``, so those are reused by default. Set
    ``load_raw_trajectory`` when the requested feature frame may need
    recomputation, e.g. lane-center features from map geometry.
    """
    benchmark_path = Path(benchmark_json)
    benchmark_items = extract_entries(load_json(benchmark_path))

    source_root_path = Path(source_root)
    report_cache: dict[Path, dict[str, dict]] = {}
    entries = []

    for item in benchmark_items:
        clip_id = item.get("clip_id")
        source_path = resolve_benchmark_source_path(
            item.get("source_scene_file"),
            benchmark_path,
            source_root_path,
        )

        source_result = None
        error = None
        metadata_path = None
        if source_path is None:
            error = (
                f"could not resolve source_scene_file: {item.get('source_scene_file')}"
            )
        else:
            if source_path not in report_cache:
                report_cache[source_path] = _index_source_report(source_path)
            source_result = report_cache[source_path].get(clip_id)
            if source_result is None:
                error = f"clip_id not found in source report: {clip_id}"
                metadata_path = _find_clip_file(
                    source_path, None, clip_id, "metadata.json"
                )
                if metadata_path is not None:
                    source_result = {"clip_id": clip_id}
                    error = None

        trajectory_xy = None
        trajectory_map_xy = None
        lane_center_lines = None
        additional_info = None
        timestamp_us = source_result.get("timestamp_us") if source_result else None
        should_load_raw = load_raw_trajectory or (
            source_result is not None
            and source_result.get("trajectory_features") is None
        )
        if source_result and should_load_raw and source_path:
            try:
                metadata_path = metadata_path or _find_clip_file(
                    source_path, None, clip_id, "metadata.json"
                )
                trajectory_xy, trajectory_map_xy, metadata_timestamp_us = (
                    _metadata_trajectory_to_xy(
                        metadata_path or source_path.with_name("metadata.json")
                    )
                )
                timestamp_us = timestamp_us or metadata_timestamp_us
                if trajectory_xy is None:
                    error = "metadata.json has no usable trajectory_poses"
                geometry_path = _find_clip_file(
                    source_path,
                    metadata_path,
                    clip_id,
                    "trajectory_plot_geometry.json",
                )
                if geometry_path is not None:
                    lane_center_lines = read_lane_center_lines(geometry_path)
            except Exception as exc:
                error = f"failed to load trajectory from metadata.json: {exc}"
        if source_result and source_path:
            try:
                metadata_path = metadata_path or _find_clip_file(
                    source_path, None, clip_id, "metadata.json"
                )
                additional_info = _read_additional_info(
                    _find_clip_file(
                        source_path, metadata_path, clip_id, "additional_info.json"
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Could not load additional trajectory info for %s: %s",
                    clip_id,
                    exc,
                )

        entries.append(
            {
                "clip_id": clip_id,
                "chain_of_thought": item.get(
                    "chain_of_thought",
                    source_result.get("chain_of_thought") if source_result else "",
                ),
                "trajectory_features": (
                    source_result.get("trajectory_features") if source_result else None
                ),
                "trajectory_xy": trajectory_xy,
                "trajectory_map_xy": trajectory_map_xy,
                "lane_center_lines": lane_center_lines,
                "metadata_path": (
                    str(metadata_path) if metadata_path is not None else None
                ),
                "trajectory_meta_actions": (
                    additional_info.get("meta_actions") if additional_info else None
                ),
                "trajectory_dynamics_summary": (
                    additional_info.get("summary") if additional_info else None
                ),
                "timestamp_us": timestamp_us,
                "benchmark": item,
                "source_report": str(source_path) if source_path else None,
                "error": error,
            }
        )

    logger.info(
        "Loaded %d benchmark entries from %s using %d source report(s)",
        len(entries),
        benchmark_path,
        len(report_cache),
    )
    return entries


def _skip_record_for_unreliable_cot(entry: dict) -> dict:
    """Build a compact audit record for an unreliable benchmark CoT skip."""
    benchmark_item = entry.get("benchmark")
    if not isinstance(benchmark_item, dict):
        benchmark_item = {}

    taxonomy = unreliability_taxonomy(benchmark_item)
    if isinstance(taxonomy, dict):
        taxonomy = taxonomy.get("categories") or taxonomy.get("primary_category")

    return {
        "clip_id": entry.get("clip_id"),
        "reason": "benchmark CoT marked unreliable",
        "cot_reliability_justification": reliability_justification(benchmark_item),
        "cot_unreliability_taxonomy": taxonomy,
        "source_report": entry.get("source_report"),
    }


def filter_unreliable_benchmark_cots(
    entries: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Drop benchmark entries with explicitly unreliable CoT annotations."""
    kept_entries = []
    skipped_entries = []
    for entry in entries:
        if cot_is_unreliable(entry.get("benchmark")):
            skipped_entries.append(_skip_record_for_unreliable_cot(entry))
        else:
            kept_entries.append(entry)
    return kept_entries, skipped_entries
