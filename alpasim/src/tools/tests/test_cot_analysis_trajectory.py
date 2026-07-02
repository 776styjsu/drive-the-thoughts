# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np

from cot_analysis.trajectory import compute_trajectory_features


def test_map_graph_same_lane_uses_only_the_starting_lane() -> None:
    trajectory_xy = np.column_stack((np.linspace(5.0, 18.0, 14), np.zeros(14)))
    lane_center_lines = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[10.0, 0.0], [20.0, 0.0]]),
    ]

    features = compute_trajectory_features(
        trajectory_xy,
        trajectory_world_xy=trajectory_xy,
        lane_center_lines=lane_center_lines,
        reference_frame="lane_center",
        lane_reference="map_graph_same_lane",
    )

    stats = features["summary_stats"]
    assert stats["lane_reference"] == "map_graph_same_lane"
    assert stats["lane_segment_count_used"] == 1
