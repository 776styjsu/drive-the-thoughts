# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon

from trajectory_safety.outcomes import SceneGeometry, compute_outcome


def _straight_lane_scene(actors: list[Polygon] | None = None) -> SceneGeometry:
    """A 4 m wide straight lane along +x with road edges 1 m outside it."""
    lane = Polygon([(0.0, -2.0), (50.0, -2.0), (50.0, 2.0), (0.0, 2.0)])
    edges = MultiLineString(
        [LineString([(0.0, 3.0), (50.0, 3.0)]), LineString([(0.0, -3.0), (50.0, -3.0)])]
    )
    return SceneGeometry(
        drivable=lane,
        road_edges=edges,
        actors=actors or [],
        ego_length_m=4.7,
        ego_width_m=1.9,
        n_lanes=1,
        timestamp_us=0,
    )


def test_in_lane_trajectory_is_not_a_departure() -> None:
    scene = _straight_lane_scene()
    traj = np.column_stack((np.linspace(0.0, 30.0, 31), np.zeros(31)))

    out = compute_outcome(traj, scene)

    assert out.max_offroad_dist_m < 0.5
    assert out.road_edge_crossings == 0
    assert not out.road_departure


def test_veering_trajectory_is_a_road_departure() -> None:
    scene = _straight_lane_scene()
    xs = np.linspace(0.0, 30.0, 31)
    ys = np.clip((xs - 10.0) * 0.5, 0.0, 6.0)  # stays in lane, then veers off the road
    traj = np.column_stack((xs, ys))

    out = compute_outcome(traj, scene)

    assert out.max_offroad_dist_m > 1.5
    assert out.road_edge_crossings > 0
    assert out.road_departure


def test_static_collision_proxy_detects_actor_on_path() -> None:
    actor = Polygon([(14.0, -1.0), (16.0, -1.0), (16.0, 1.0), (14.0, 1.0)])
    scene = _straight_lane_scene(actors=[actor])
    traj = np.column_stack((np.linspace(0.0, 30.0, 31), np.zeros(31)))

    out = compute_outcome(traj, scene)

    assert out.n_actors == 1
    assert out.static_collision
    assert out.min_actor_clearance_m == 0.0


def test_clear_path_has_positive_actor_clearance() -> None:
    actor = Polygon([(14.0, 8.0), (16.0, 8.0), (16.0, 10.0), (14.0, 10.0)])
    scene = _straight_lane_scene(actors=[actor])
    traj = np.column_stack((np.linspace(0.0, 30.0, 31), np.zeros(31)))

    out = compute_outcome(traj, scene)

    assert not out.static_collision
    assert out.min_actor_clearance_m > 0.0
