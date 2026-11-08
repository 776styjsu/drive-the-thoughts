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


def test_start_lanes_limited_to_search_radius() -> None:
    # Ego starts between a lane 2.5m left and one 3.4m right, then drifts
    # onto the right lane. The right lane fits the horizon better but starts
    # outside the 3m search radius, so it must not be a starting candidate.
    trajectory_xy = np.column_stack(
        (np.linspace(0.0, 30.0, 31), np.linspace(0.0, -3.4, 31))
    )
    left = np.array([[-5.0, 2.5], [40.0, 2.5]])
    right = np.array([[-5.0, -3.4], [40.0, -3.4]])

    reference, _ = build_route_consistent_reference(trajectory_xy, [left, right])
    np.testing.assert_allclose(reference[:, 1], 2.5)

    reference, _ = build_route_consistent_reference(
        trajectory_xy, [left, right], start_search_radius_m=4.0
    )
    np.testing.assert_allclose(reference[:, 1], -3.4)


def test_start_falls_back_to_nearest_lane_outside_radius() -> None:
    trajectory_xy = np.column_stack((np.linspace(0.0, 20.0, 21), np.zeros(21)))
    near = np.array([[-5.0, 4.0], [30.0, 4.0]])
    far = np.array([[-5.0, -6.0], [30.0, -6.0]])

    reference, _ = build_route_consistent_reference(trajectory_xy, [far, near])
    np.testing.assert_allclose(reference[:, 1], 4.0)
