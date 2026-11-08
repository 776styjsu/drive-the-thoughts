import json

import numpy as np
import pytest

from alpasim_utils.cot_consistency import compute_trajectory_features
from cot_analysis.benchmark_source import (
    _metadata_trajectory_to_xy,
    extract_entries_from_benchmark,
)
from cot_analysis.pipeline import process_entry


def _write_metadata(path, poses, rig=None):
    payload = {"time_now_us": 0, "trajectory_poses": poses}
    if rig is not None:
        payload["trajectory_xy_rig_frame"] = rig
    path.write_text(json.dumps(payload))
    return path


def _north_poses(n=5):
    # Ego at (10, 20) facing map +y, driving straight 1 m per step.
    return [{"x": 10.0, "y": 20.0 + i, "z": 0.0} for i in range(n)]


def test_uses_rig_frame_from_metadata(tmp_path):
    rig = [{"timestamp_us": i, "rx": float(i), "ry": 0.1 * i} for i in range(5)]
    path = _write_metadata(tmp_path / "metadata.json", _north_poses(), rig=rig)

    relative_xy, map_xy, _ = _metadata_trajectory_to_xy(path)

    np.testing.assert_allclose(relative_xy, [[p["rx"], p["ry"]] for p in rig])
    np.testing.assert_allclose(map_xy[:, 1], [20, 21, 22, 23, 24])


def test_fallback_rotates_map_xy_into_heading(tmp_path):
    path = _write_metadata(tmp_path / "metadata.json", _north_poses())

    relative_xy, _, _ = _metadata_trajectory_to_xy(path)

    # Straight-ahead driving must read as forward (+x), not leftward (+y).
    np.testing.assert_allclose(relative_xy[:, 0], np.arange(5), atol=1e-9)
    np.testing.assert_allclose(relative_xy[:, 1], 0.0, atol=1e-9)


def test_fallback_keeps_stationary_trajectory_at_origin(tmp_path):
    poses = [{"x": 3.0, "y": 4.0, "z": 0.0}] * 4
    path = _write_metadata(tmp_path / "metadata.json", poses)

    relative_xy, _, _ = _metadata_trajectory_to_xy(path)

    np.testing.assert_allclose(relative_xy, 0.0)


def _benchmark_with_cache(tmp_path, cache):
    report = tmp_path / "source.json"
    report.write_text(json.dumps({"clip_id": "clip", "trajectory_features": cache}))
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "clip_id": "clip",
                    "source_scene_file": str(report),
                    "chain_of_thought": "Keep lane.",
                }
            ]
        )
    )
    return benchmark


def test_lane_cache_is_recomputed_as_ego_from_raw_metadata(tmp_path):
    xy = np.column_stack((np.arange(5), np.zeros(5)))
    lane_cache = compute_trajectory_features(
        xy, route_xy=xy, reference_frame="lane_center", lane_reference="route"
    )
    benchmark = _benchmark_with_cache(tmp_path, lane_cache)
    _write_metadata(tmp_path / "metadata.json", _north_poses())

    entry = extract_entries_from_benchmark(str(benchmark))[0]
    assert entry["error"] is None
    assert entry["trajectory_xy"] is not None
    result = process_entry(None, "test-model", entry)

    assert (
        result["trajectory_features"]["summary_stats"]["reference_frame"] == "ego_rig"
    )
    assert "- s_lane:" not in result["judge_prompt"]
    assert "- x: 4.0m" in result["judge_prompt"]


def test_dual_cache_reuses_its_explicit_ego_view_without_raw_metadata(tmp_path):
    xy = np.column_stack((np.arange(5), np.zeros(5)))
    cache = compute_trajectory_features(xy, reference_frame="dual")
    benchmark = _benchmark_with_cache(tmp_path, cache)

    entry = extract_entries_from_benchmark(str(benchmark))[0]
    assert entry["error"] is None
    assert entry["trajectory_xy"] is None
    result = process_entry(None, "test-model", entry)
    assert result["trajectory_features"] == cache["ego_features"]
    assert (
        result["trajectory_features"]["summary_stats"]["reference_frame"] == "ego_rig"
    )


def test_incompatible_cache_without_raw_trajectory_cannot_be_judged(tmp_path):
    benchmark = _benchmark_with_cache(
        tmp_path, {"summary_stats": {"reference_frame": "lane_center"}}
    )
    entry = extract_entries_from_benchmark(str(benchmark))[0]
    assert "no usable trajectory" in entry["error"]
    with pytest.raises(ValueError, match="Cannot prepare trajectory evidence"):
        process_entry(object(), "test-model", entry)
