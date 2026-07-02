# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Geometric safety outcomes for a single planned ego trajectory.

This is the core of "Experiment A": instead of relying on a human label of
whether a planned trajectory is safe, we *play the plan out against the map*
and measure an objective outcome (road departure, plus a static collision
proxy). The computation mirrors the off-road definition in
``src/eval/src/eval/scorers/offroad.py`` (lane-polygon containment, union of
nearby lanes, road-edge boundary) but operates directly on the per-frame
``trajectory_plot_geometry.json`` that each benchmark frame already provides,
so it needs no neural driver and no full simulation loop.

Frames and units
----------------
``metadata.json`` stores the planned trajectory both in the world/map frame
(``trajectory_poses`` -> x, y, z) and the ego-rig frame
(``trajectory_xy_rig_frame``). ``trajectory_plot_geometry.json`` stores
``map_linestrings`` and ``actors`` in the *same world frame*; we therefore use
the world-frame ``trajectory_poses`` so trajectory, lanes, and actors overlay
directly. All distances are metres.

Scope and limitations
----------------------
* The map (lanes, road edges) is static, so the **road-departure** outcome is
  exact for an open-loop planned trajectory and is the headline signal.
* Other actors are only available as a single snapshot at ``t_now`` (the
  geometry file carries no future actor motion), so the **collision** signal is
  a STATIC-WORLD proxy: it asks whether the planned ego footprint sweeps through
  space that is occupied at ``t_now``. It is reported as a secondary, clearly
  caveated signal and is *not* part of the headline outcome.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, MultiLineString, Polygon

# Fallback ego footprint (length x width, metres) when the geometry file has no
# EGO actor to measure. Roughly a passenger car / robotaxi.
_DEFAULT_EGO_LENGTH_M = 4.7
_DEFAULT_EGO_WIDTH_M = 1.9

# A footprint is considered to touch a road edge when its distance to the edge
# is below this tolerance (metres); mirrors the 1e-3 used by the eval scorer.
_EDGE_TOUCH_TOL_M = 1e-6


@dataclass
class SceneGeometry:
    """Static world geometry extracted from ``trajectory_plot_geometry.json``."""

    drivable: shapely.geometry.base.BaseGeometry  # union of lane polygons
    road_edges: MultiLineString | None
    actors: list[Polygon]  # non-ego actor footprints at t_now (static)
    ego_length_m: float
    ego_width_m: float
    n_lanes: int
    timestamp_us: int | None


@dataclass
class TrajectoryOutcome:
    """Objective safety outcome of a single planned trajectory."""

    n_waypoints: int
    ego_length_m: float
    ego_width_m: float
    # --- road departure (headline) ---
    max_offroad_dist_m: float  # max distance the ego centre travels outside lanes
    frac_waypoints_off_lane: float  # diagnostic: share of waypoints off the lanes
    road_edge_crossings: int  # waypoints whose footprint touches a road edge
    road_departure: bool  # max_offroad_dist_m >= thresh AND a road edge is crossed
    # --- collision (secondary, static-world proxy) ---
    min_actor_clearance_m: float  # min footprint->actor distance (nan if no actors)
    static_collision: bool  # any footprint intersects an actor at t_now
    n_actors: int

    def to_dict(self) -> dict:
        return asdict(self)


def _coords_xy(linestring_entry: dict) -> np.ndarray:
    """Return the (N, 2) xy array of a ``map_linestrings`` entry."""
    return np.asarray(linestring_entry["coords"], dtype=float)[:, :2]


def _polygon_from_edges(left: np.ndarray, right: np.ndarray) -> Polygon | None:
    """Build a lane polygon from its left and right edge polylines.

    The two edges run in the same direction (verified: the lane centre equals
    the per-vertex midpoint of left/right), so left + reversed(right) closes a
    simple ring. Self-intersections from noisy geometry are healed with
    ``buffer(0)``.
    """
    try:
        poly = Polygon(np.concatenate([left, right[::-1]], axis=0))
    except (ValueError, AssertionError):
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0.0:
        return None
    return poly


def _ego_dims_from_polygon(coords: np.ndarray) -> tuple[float, float]:
    """Recover (length, width) from an EGO footprint polygon via its OBB."""
    poly = Polygon(coords[:, :2])
    if not poly.is_valid:
        poly = poly.buffer(0)
    obb = poly.minimum_rotated_rectangle
    edge_pts = np.asarray(obb.exterior.coords)
    edge_len = np.linalg.norm(np.diff(edge_pts, axis=0), axis=1)
    if len(edge_len) < 2:
        return _DEFAULT_EGO_LENGTH_M, _DEFAULT_EGO_WIDTH_M
    length, width = sorted((float(edge_len[0]), float(edge_len[1])), reverse=True)
    # Guard against degenerate footprints.
    if length <= 0.5 or width <= 0.5:
        return _DEFAULT_EGO_LENGTH_M, _DEFAULT_EGO_WIDTH_M
    return length, width


def load_scene_geometry(geometry_path: str | Path) -> SceneGeometry:
    """Load static lanes, road edges and actor footprints for one frame."""
    geom = json.loads(Path(geometry_path).read_text())
    linestrings = geom.get("map_linestrings", [])

    lefts = [
        _coords_xy(x) for x in linestrings if x.get("type") == "road_lane_left_edge"
    ]
    rights = [
        _coords_xy(x) for x in linestrings if x.get("type") == "road_lane_right_edge"
    ]
    lanes: list[Polygon] = []
    for left, right in zip(lefts, rights):
        poly = _polygon_from_edges(left, right)
        if poly is not None:
            lanes.append(poly)
    drivable = shapely.ops.unary_union(lanes) if lanes else Polygon()

    edge_lines = [
        LineString(_coords_xy(x))
        for x in linestrings
        if x.get("type") == "road_edge" and len(x.get("coords", [])) >= 2
    ]
    road_edges = MultiLineString(edge_lines) if edge_lines else None

    actors: list[Polygon] = []
    ego_length, ego_width = _DEFAULT_EGO_LENGTH_M, _DEFAULT_EGO_WIDTH_M
    for actor in geom.get("actors", []):
        coords = np.asarray(actor["polygon_coords"], dtype=float)
        if actor.get("id") == "EGO":
            ego_length, ego_width = _ego_dims_from_polygon(coords)
            continue
        poly = Polygon(coords[:, :2])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0.0:
            actors.append(poly)

    return SceneGeometry(
        drivable=drivable,
        road_edges=road_edges,
        actors=actors,
        ego_length_m=ego_length,
        ego_width_m=ego_width,
        n_lanes=len(lanes),
        timestamp_us=geom.get("timestamp_us"),
    )


def load_planned_trajectory(metadata_path: str | Path) -> np.ndarray:
    """Return the planned trajectory as world-frame (T, 2) xy waypoints."""
    meta = json.loads(Path(metadata_path).read_text())
    poses = meta.get("trajectory_poses") or []
    if not poses:
        return np.empty((0, 2), dtype=float)
    return np.asarray([[p["x"], p["y"]] for p in poses], dtype=float)


def _headings(xy: np.ndarray) -> np.ndarray:
    """Per-waypoint travel heading (radians) from finite differences."""
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=float)
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    return np.arctan2(dy, dx)


def _oriented_box(
    center: np.ndarray, heading: float, length: float, width: float
) -> Polygon:
    """Axis-aligned-then-rotated ego footprint centred at ``center``."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    fwd = np.array([cos_h, sin_h])
    left = np.array([-sin_h, cos_h])
    half_l, half_w = length / 2.0, width / 2.0
    return Polygon(
        [
            center + fwd * half_l + left * half_w,
            center + fwd * half_l - left * half_w,
            center - fwd * half_l - left * half_w,
            center - fwd * half_l + left * half_w,
        ]
    )


def compute_outcome(
    trajectory_xy: np.ndarray,
    geom: SceneGeometry,
    *,
    offroad_thresh_m: float = 1.5,
) -> TrajectoryOutcome:
    """Compute the geometric safety outcome of a planned trajectory.

    Args:
        trajectory_xy: World-frame (T, 2) planned waypoints.
        geom: Static scene geometry from :func:`load_scene_geometry`.
        offroad_thresh_m: Minimum off-lane excursion of the ego centre (metres)
            required, together with a road-edge crossing, to call the plan a
            road departure. Defaults to 1.5 m, which on the reliable benchmark
            subset separates the unsafe trajectories from the safe ones with the
            fewest false positives.
    """
    headings = _headings(trajectory_xy)
    footprints = [
        _oriented_box(
            trajectory_xy[i], float(headings[i]), geom.ego_length_m, geom.ego_width_m
        )
        for i in range(len(trajectory_xy))
    ]

    drivable = geom.drivable
    has_lanes = (drivable is not None) and (not drivable.is_empty)

    off_lane = np.zeros(len(footprints), dtype=bool)
    off_dist = np.zeros(len(footprints), dtype=float)
    edge_cross = np.zeros(len(footprints), dtype=bool)
    for i, fp in enumerate(footprints):
        if has_lanes:
            off_lane[i] = not drivable.contains(fp.centroid)
            off_dist[i] = float(drivable.distance(fp.centroid))
        if geom.road_edges is not None:
            edge_cross[i] = fp.distance(geom.road_edges) < _EDGE_TOUCH_TOL_M

    max_offroad_dist = float(off_dist.max()) if len(off_dist) else 0.0
    road_edge_crossings = int(edge_cross.sum())
    road_departure = bool(
        max_offroad_dist >= offroad_thresh_m and road_edge_crossings > 0
    )

    if geom.actors:
        clearances = [min(fp.distance(a) for a in geom.actors) for fp in footprints]
        min_clearance = float(min(clearances)) if clearances else float("nan")
        static_collision = bool(
            any(fp.intersects(a) for fp in footprints for a in geom.actors)
        )
    else:
        min_clearance = float("nan")
        static_collision = False

    return TrajectoryOutcome(
        n_waypoints=len(footprints),
        ego_length_m=round(geom.ego_length_m, 3),
        ego_width_m=round(geom.ego_width_m, 3),
        max_offroad_dist_m=round(max_offroad_dist, 3),
        frac_waypoints_off_lane=round(
            float(off_lane.mean()) if len(off_lane) else 0.0, 3
        ),
        road_edge_crossings=road_edge_crossings,
        road_departure=road_departure,
        min_actor_clearance_m=(
            round(min_clearance, 3) if min_clearance == min_clearance else float("nan")
        ),
        static_collision=static_collision,
        n_actors=len(geom.actors),
    )


def score_frame(
    metadata_path: str | Path,
    geometry_path: str | Path,
    *,
    offroad_thresh_m: float = 1.5,
) -> TrajectoryOutcome:
    """Convenience: load one frame's data and compute its outcome."""
    trajectory_xy = load_planned_trajectory(metadata_path)
    geom = load_scene_geometry(geometry_path)
    return compute_outcome(trajectory_xy, geom, offroad_thresh_m=offroad_thresh_m)
