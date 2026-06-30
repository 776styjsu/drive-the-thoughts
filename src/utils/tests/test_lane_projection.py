# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np

from alpasim_utils.lane_projection import build_route_consistent_reference


def test_same_lane_reference_does_not_traverse_successors() -> None:
    trajectory_xy = np.column_stack((np.linspace(5.0, 18.0, 14), np.zeros(14)))
    lane_center_lines = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[10.0, 0.0], [20.0, 0.0]]),
    ]

    graph_reference, graph_lanes_used = build_route_consistent_reference(
        trajectory_xy,
        lane_center_lines,
    )
    same_lane_reference, same_lane_lanes_used = build_route_consistent_reference(
        trajectory_xy,
        lane_center_lines,
        stay_on_start_lane=True,
    )

    assert graph_lanes_used == 2
    assert graph_reference[-1, 0] == 20.0
    assert same_lane_lanes_used == 1
    assert same_lane_reference[-1, 0] == 10.0
