# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json

import pytest
from alpasim_utils.trajectory_additional_info import (
    build_additional_info,
    generate_additional_info_files,
)


def test_build_additional_info_from_rig_frame_segments():
    metadata = {
        "clip_id": "clip_t0001",
        "time_now_us": 10,
        "time_query_us": 20,
        "trajectory_xy_rig_frame": [
            {"timestamp_us": 0, "rx": 0.0, "ry": 0.0},
            {"timestamp_us": 1_000_000, "rx": 1.0, "ry": 0.0},
            {"timestamp_us": 2_000_000, "rx": 3.0, "ry": 0.5},
        ],
    }

    info = build_additional_info(metadata, precision=6)

    assert info["trajectory_source"] == "trajectory_xy_rig_frame"
    assert info["coordinate_frame"] == "ego_rig_frame"
    assert info["summary"]["waypoint_count"] == 3
    assert info["summary"]["segment_count"] == 2
    assert info["summary"]["total_distance_2d_m"] == pytest.approx(3.061553)

    first, second = info["segments"]
    assert first["velocity_mps"]["x"] == pytest.approx(1.0)
    assert first["velocity_mps"]["speed_2d"] == pytest.approx(1.0)
    assert first["acceleration_from_previous_mps2"] is None

    assert second["velocity_mps"]["x"] == pytest.approx(2.0)
    assert second["velocity_mps"]["y"] == pytest.approx(0.5)
    assert second["velocity_mps"]["speed_2d"] == pytest.approx(2.061553)
    assert second["acceleration_from_previous_mps2"]["x"] == pytest.approx(1.0)
    assert second["acceleration_from_previous_mps2"]["y"] == pytest.approx(0.5)
    assert second["acceleration_from_previous_mps2"]["longitudinal"] == pytest.approx(
        1.061553
    )
    assert second["motion_class"] == "accelerating"


def test_build_additional_info_falls_back_to_global_poses():
    metadata = {
        "clip_id": "clip_t0002",
        "trajectory_xy_rig_frame": [],
        "trajectory_poses": [
            {"timestamp_us": 0, "x": 4.0, "y": 5.0, "z": 0.25},
            {"timestamp_us": 500_000, "x": 5.0, "y": 5.0, "z": 0.25},
        ],
    }

    info = build_additional_info(metadata)

    assert info["trajectory_source"] == "trajectory_poses"
    assert info["coordinate_frame"] == "global_frame"
    assert info["segments"][0]["velocity_mps"]["speed_2d"] == pytest.approx(2.0)


def test_build_additional_info_handles_empty_trajectory():
    info = build_additional_info(
        {"clip_id": "empty", "trajectory_xy_rig_frame": []},
    )

    assert info["summary"]["trajectory_status"] == "insufficient_points"
    assert info["summary"]["waypoint_count"] == 0
    assert info["segments"] == []
    assert info["waypoints"] == []


def test_generate_additional_info_files_recursively(tmp_path):
    frame_dir = tmp_path / "extracted_frames" / "clip" / "clip_t0001"
    frame_dir.mkdir(parents=True)
    metadata_path = frame_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "clip_id": "clip_t0001",
                "trajectory_xy_rig_frame": [
                    {"timestamp_us": 0, "rx": 0.0, "ry": 0.0},
                    {"timestamp_us": 100_000, "rx": 1.0, "ry": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    results = generate_additional_info_files(tmp_path / "extracted_frames")

    assert len(results) == 1
    assert results[0].written
    output_path = frame_dir / "additional_info.json"
    assert output_path.exists()
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["summary"]["max_speed_mps"] == pytest.approx(10.0)
