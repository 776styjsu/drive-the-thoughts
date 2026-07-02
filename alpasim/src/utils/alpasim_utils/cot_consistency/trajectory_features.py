# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Trajectory feature computation for CoT consistency analysis.

Computes raw numerical trajectory features (positions, velocities,
accelerations, lateral deviation) from rig-frame trajectory poses.
The LLM interprets maneuvers from the raw data — no categorical
labels or magic thresholds.
"""

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

# Pure projection / lane-reference geometry lives in alpasim_utils so the CoT
# judge and the extract_frame visualization compute lane offset identically.
from alpasim_utils.lane_projection import as_xy_array as _as_xy_array
from alpasim_utils.lane_projection import (
    build_route_consistent_reference as _build_route_consistent_reference,
)
from alpasim_utils.lane_projection import (
    project_points_to_polyline as _project_points_to_polyline,
)
from alpasim_utils.lane_projection import (
    project_points_to_polyline_monotone as _project_points_to_polyline_monotone,
)


def _compute_ego_frame_features(xy: np.ndarray, dt: float) -> dict:
    """Compute trajectory features in the ego/rig frame."""
    T = xy.shape[0]

    # --- Per-step kinematics ---
    dxy = np.diff(xy, axis=0)  # (T-1, 2)
    velocities = dxy / dt  # m/s
    speeds = np.linalg.norm(velocities, axis=1)  # (T-1,)

    # Acceleration (from speed differences)
    if len(speeds) > 1:
        accel = np.diff(speeds) / dt  # m/s², (T-2,)
    else:
        accel = np.array([0.0])

    # Lateral velocity (Y component in rig/ego frame)
    lat_vel = velocities[:, 1]  # positive = leftward

    # --- Summary stats ---
    total_path_length = float(np.sum(np.linalg.norm(dxy, axis=1)))
    final_displacement = xy[-1] - xy[0]
    max_lat_deviation = float(xy[:, 1].max())
    min_lat_deviation = float(xy[:, 1].min())
    heading_rel_deg, traj_net_heading_deg, traj_abs_heading_deg = (
        _trajectory_heading_series(xy)
    )

    summary_stats = {
        "reference_frame": "ego_rig",
        "duration_s": round((T - 1) * dt, 1),
        "total_path_length_m": round(total_path_length, 2),
        "traj_net_heading_change_deg": round(traj_net_heading_deg, 1),
        "traj_total_heading_change_deg": round(traj_abs_heading_deg, 1),
        "final_longitudinal_m": round(float(final_displacement[0]), 2),
        "final_lateral_m": round(float(final_displacement[1]), 2),
        "max_lateral_m": round(max_lat_deviation, 2),
        "min_lateral_m": round(min_lat_deviation, 2),
        "mean_speed_ms": round(float(speeds.mean()), 2),
        "max_speed_ms": round(float(speeds.max()), 2),
        "min_speed_ms": round(float(speeds.min()), 2),
        "mean_accel_ms2": round(float(accel.mean()), 2),
    }

    # --- Downsample to ~1.25Hz for the LLM table ---
    step = max(1, int(0.8 / dt))  # every 8 steps at 10Hz
    sample_indices = list(range(0, T, step))
    if (T - 1) not in sample_indices:
        sample_indices.append(T - 1)

    table_rows = []
    for i in sample_indices:
        t_sec = round(i * dt, 1)
        x, y = round(float(xy[i, 0]), 2), round(float(xy[i, 1]), 2)
        if i < len(speeds):
            spd = round(float(speeds[i]), 2)
            lv = round(float(lat_vel[i]), 2)
        else:
            spd = round(float(speeds[-1]), 2)
            lv = round(float(lat_vel[-1]), 2)
        if i < len(accel):
            acc = round(float(accel[i]), 2)
        elif len(accel) > 0:
            acc = round(float(accel[-1]), 2)
        else:
            acc = 0.0
        table_rows.append(
            {
                "t": t_sec,
                "x": x,
                "y": y,
                "heading_deg": round(float(heading_rel_deg[i]), 1),
                "speed": spd,
                "lat_vel": lv,
                "accel": acc,
            }
        )

    # Format as markdown for the prompt
    lines = []
    for r in table_rows:
        lines.append(f"### t={r['t']}s")
        lines.append(f"- x: {r['x']}m")
        lines.append(f"- y: {r['y']}m")
        lines.append(f"- speed: {r['speed']} m/s")
        lines.append(f"- lat_vel: {r['lat_vel']} m/s")
        lines.append(f"- accel: {r['accel']} m/s²")

    return {
        "summary_stats": summary_stats,
        "table_rows": table_rows,
        "markdown_kv": "\n".join(lines),
    }


def _trajectory_heading_series(
    xy: np.ndarray, min_step_m: float = 0.05
) -> tuple[np.ndarray, float, float]:
    """Per-point heading change of a trajectory, plus net and total turning.

    Heading is the direction of motion (atan2 of each step), which is
    frame-independent for *changes*: a left turn of 90 deg reads as +90 deg
    whether the points are in the world or ego frame. This is the turn evidence
    the lane-relative offset cancels out — when the reference route curves
    through an intersection turn, offset_m stays ~0 even though the trajectory
    physically rotated. Near-stationary steps (norm < min_step_m) carry the last
    valid heading so a stop or crawl does not inject atan2 noise.

    Returns:
        Tuple of (heading_rel_deg, net_deg, total_abs_deg). heading_rel_deg has
        one entry per trajectory point (length T), measured relative to the
        initial heading; net_deg is the signed start->end change (+ = left/CCW);
        total_abs_deg is the cumulative absolute turning along the path.
    """
    steps = np.diff(xy, axis=0)
    if len(steps) == 0:
        return np.zeros(len(xy)), 0.0, 0.0
    norms = np.linalg.norm(steps, axis=1)
    valid = norms > min_step_m
    if not np.any(valid):
        return np.zeros(len(xy)), 0.0, 0.0

    headings = np.arctan2(steps[:, 1], steps[:, 0])
    # Forward-fill invalid (near-stationary) steps with the last good heading,
    # then back-fill any leading invalid steps with the first good heading.
    last = None
    for i in range(len(headings)):
        if valid[i]:
            last = headings[i]
        elif last is not None:
            headings[i] = last
    first_valid = headings[int(np.argmax(valid))]
    for i in range(len(headings)):
        if valid[i]:
            break
        headings[i] = first_valid

    unwrapped = np.unwrap(headings)
    rel = unwrapped - unwrapped[0]
    net = float(rel[-1])
    total_abs = float(np.sum(np.abs(np.diff(unwrapped)))) if len(unwrapped) > 1 else 0.0
    # Extend step-headings (length T-1) to one value per point (length T).
    per_point = np.concatenate([rel, [rel[-1]]])
    return np.degrees(per_point), float(np.degrees(net)), float(np.degrees(total_abs))


def _route_heading_change(
    route_xy: np.ndarray, s_start: float, s_end: float
) -> tuple[float, float]:
    """Net and total heading change of the reference route over [s_start, s_end].

    Lets the judge tell a genuine maneuver from road-following: if the reference
    route itself turns ~90 deg over the covered span, a trajectory that also
    turns ~90 deg while holding offset_m near zero is executing that turn, not
    drifting. Returns (net_deg, total_abs_deg), + = left/CCW.
    """
    seg = route_xy[1:] - route_xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1.0e-6
    if not np.any(keep):
        return 0.0, 0.0
    seg = seg[keep]
    seg_len = seg_len[keep]
    seg_start_s = np.concatenate(([0.0], np.cumsum(seg_len)))[:-1]
    seg_end_s = seg_start_s + seg_len
    lo, hi = (s_start, s_end) if s_start <= s_end else (s_end, s_start)
    in_span = (seg_end_s >= lo) & (seg_start_s <= hi)
    if not np.any(in_span):
        in_span = np.ones(len(seg), dtype=bool)
    headings = np.unwrap(np.arctan2(seg[in_span, 1], seg[in_span, 0]))
    if len(headings) < 2:
        return 0.0, 0.0
    net = float(headings[-1] - headings[0])
    total_abs = float(np.sum(np.abs(np.diff(headings))))
    return float(np.degrees(net)), float(np.degrees(total_abs))


def _compute_lane_center_features(
    xy: np.ndarray,
    lane_center_xy: np.ndarray,
    dt: float,
    lane_reference: str,
    lane_segments_used: int | None = None,
    route_consistent: bool = False,
) -> dict:
    """Compute trajectory features relative to a lane-center polyline.

    With route_consistent=True (map_graph mode), projection uses the
    arc-length continuity window, and speed/accel come from the raw
    trajectory steps — speed is frame-independent, so this avoids the
    Frenet ds distortion at large lateral offsets on curves (corner
    cutting amplifies ds by 1/(1-d*kappa)).
    """
    T = xy.shape[0]
    if route_consistent:
        s_lane, offsets = _project_points_to_polyline_monotone(xy, lane_center_xy)
    else:
        s_lane, offsets = _project_points_to_polyline(xy, lane_center_xy)
    s_progress = s_lane - s_lane[0]
    delta_offsets = offsets - offsets[0]

    ds = np.diff(s_progress)
    doffset = np.diff(offsets)
    lat_vel = doffset / dt
    if route_consistent:
        speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        accel_basis = speeds
    else:
        speeds = np.linalg.norm(np.column_stack((ds, doffset)), axis=1) / dt
        accel_basis = ds / dt  # legacy: lane-progress velocity

    if len(accel_basis) > 1:
        accel = np.diff(accel_basis) / dt
    else:
        accel = np.array([0.0])

    step = max(1, int(0.8 / dt))
    sample_indices = list(range(0, T, step))
    if (T - 1) not in sample_indices:
        sample_indices.append(T - 1)

    # Heading / turn evidence: the lateral offset cancels turns when the
    # reference route curves through them, so carry the raw heading change of
    # the trajectory and the route's own heading change as explicit signals.
    heading_rel_deg, traj_net_heading_deg, traj_abs_heading_deg = (
        _trajectory_heading_series(xy)
    )
    route_net_heading_deg, route_abs_heading_deg = _route_heading_change(
        lane_center_xy, float(s_lane[0]), float(s_lane[-1])
    )

    summary_stats = {
        "reference_frame": "lane_center",
        "lane_reference": lane_reference,
        "duration_s": round((T - 1) * dt, 1),
        "lane_path_length_m": round(float(s_progress[-1]), 2),
        "traj_net_heading_change_deg": round(traj_net_heading_deg, 1),
        "traj_total_heading_change_deg": round(traj_abs_heading_deg, 1),
        "route_net_heading_change_deg": round(route_net_heading_deg, 1),
        "route_total_heading_change_deg": round(route_abs_heading_deg, 1),
        "heading_minus_route_net_deg": round(
            traj_net_heading_deg - route_net_heading_deg, 1
        ),
        "initial_offset_m": round(float(offsets[0]), 2),
        "final_offset_m": round(float(offsets[-1]), 2),
        "delta_offset_m": round(float(delta_offsets[-1]), 2),
        "min_offset_m": round(float(offsets.min()), 2),
        "max_offset_m": round(float(offsets.max()), 2),
        "mean_speed_ms": round(float(speeds.mean()), 2),
        "max_speed_ms": round(float(speeds.max()), 2),
        "min_speed_ms": round(float(speeds.min()), 2),
        "mean_accel_ms2": round(float(accel.mean()), 2),
        "lane_sample_count": int(len(lane_center_xy)),
        "lane_sample_stride": step,
    }
    if lane_segments_used is not None:
        summary_stats["lane_segment_count_used"] = int(lane_segments_used)

    table_rows = []
    for i in sample_indices:
        if i < len(speeds):
            spd = round(float(speeds[i]), 2)
            lv = round(float(lat_vel[i]), 2)
        else:
            spd = round(float(speeds[-1]), 2)
            lv = round(float(lat_vel[-1]), 2)
        if i < len(accel):
            acc = round(float(accel[i]), 2)
        elif len(accel) > 0:
            acc = round(float(accel[-1]), 2)
        else:
            acc = 0.0

        table_rows.append(
            {
                "t": round(i * dt, 1),
                "s_lane": round(float(s_progress[i]), 2),
                "offset_m": round(float(offsets[i]), 2),
                "delta_offset_m": round(float(delta_offsets[i]), 2),
                "speed": spd,
                "lat_vel": lv,
                "accel": acc,
                "heading_deg": round(float(heading_rel_deg[i]), 1),
            }
        )

    lines = []
    for r in table_rows:
        lines.append(f"### t={r['t']}s")
        lines.append(f"- s_lane: {r['s_lane']}m")
        lines.append(f"- offset_m: {r['offset_m']}m")
        lines.append(f"- delta_offset_m: {r['delta_offset_m']}m")
        lines.append(f"- speed: {r['speed']} m/s")
        lines.append(f"- lat_vel: {r['lat_vel']} m/s")
        lines.append(f"- accel: {r['accel']} m/s²")

    return {
        "summary_stats": summary_stats,
        "table_rows": table_rows,
        "markdown_kv": "\n".join(lines),
    }


@dataclass(frozen=True)
class TrajectoryFeatureContext:
    """Normalized inputs shared by trajectory feature strategies."""

    xy: np.ndarray
    dt: float
    route_xy: np.ndarray | None = None
    trajectory_world_xy: np.ndarray | None = None
    lane_center_lines: list[np.ndarray] | None = None
    lane_reference: str = "auto"

    @property
    def route(self) -> np.ndarray | None:
        return _as_xy_array(self.route_xy)

    @property
    def world_xy(self) -> np.ndarray | None:
        return _as_xy_array(self.trajectory_world_xy)

    def with_lane_reference(self, lane_reference: str) -> "TrajectoryFeatureContext":
        return replace(self, lane_reference=lane_reference)


class TrajectoryFeatureComputer(Protocol):
    """Strategy interface for one trajectory reference-frame view."""

    reference_frame: str

    def compute(self, context: TrajectoryFeatureContext) -> dict:
        """Compute feature payload for this strategy."""
        ...


class EgoRigFeatureComputer:
    """Feature strategy for the raw ego/rig trajectory frame."""

    reference_frame = "ego_rig"

    def compute(self, context: TrajectoryFeatureContext) -> dict:
        return _compute_ego_frame_features(context.xy, context.dt)


class LaneCenterFeatureComputer:
    """Feature strategy for route/lane-relative trajectory views."""

    reference_frame = "lane_center"

    def compute(self, context: TrajectoryFeatureContext) -> dict:
        lane_errors = []
        world_xy = context.world_xy
        lane_center_lines = context.lane_center_lines
        lane_reference = context.lane_reference

        if lane_reference in {"auto", "map_graph", "map_graph_same_lane"}:
            graph_reference = (
                "map_graph" if lane_reference == "auto" else lane_reference
            )
            if world_xy is not None and lane_center_lines:
                try:
                    reference_xy, lanes_used = _build_route_consistent_reference(
                        world_xy,
                        lane_center_lines,
                        stay_on_start_lane=(graph_reference == "map_graph_same_lane"),
                    )
                    return _compute_lane_center_features(
                        world_xy,
                        reference_xy,
                        context.dt,
                        lane_reference=graph_reference,
                        lane_segments_used=lanes_used,
                        route_consistent=True,
                    )
                except ValueError as exc:
                    lane_errors.append(f"{graph_reference}: {exc}")
            else:
                lane_errors.append(
                    f"{graph_reference}: No map lane-center geometry available"
                )

            if lane_reference != "auto":
                return self._ego_fallback(context, lane_reference, lane_errors)

        route = context.route
        if (
            lane_reference in {"auto", "route"}
            and route is not None
            and len(route) >= 2
        ):
            try:
                return _compute_lane_center_features(
                    context.xy,
                    route,
                    context.dt,
                    lane_reference="route",
                )
            except ValueError as exc:
                lane_errors.append(f"route: {exc}")

        if not lane_errors:
            lane_errors.append("No lane-center reference available")

        return self._ego_fallback(context, lane_reference, lane_errors)

    @staticmethod
    def _ego_fallback(
        context: TrajectoryFeatureContext,
        lane_reference: str,
        lane_errors: list[str],
    ) -> dict:
        features = _compute_ego_frame_features(context.xy, context.dt)
        features["summary_stats"]["requested_reference_frame"] = "lane_center"
        features["summary_stats"]["lane_reference"] = lane_reference
        features["summary_stats"]["lane_center_error"] = "; ".join(lane_errors)
        return features


class DualFrameFeatureComputer:
    """Feature strategy that packages ego and lane-relative views together."""

    reference_frame = "dual"

    def compute(self, context: TrajectoryFeatureContext) -> dict:
        ego_features = _FEATURE_COMPUTERS["ego_rig"].compute(context)
        lane_reference = (
            "map_graph" if context.lane_reference == "auto" else context.lane_reference
        )
        lane_features = _FEATURE_COMPUTERS["lane_center"].compute(
            context.with_lane_reference(lane_reference)
        )
        lane_stats = lane_features.get("summary_stats", {})
        summary_stats = {
            "reference_frame": "dual",
            "lane_reference": lane_stats.get("lane_reference"),
            "duration_s": ego_features["summary_stats"]["duration_s"],
            "mean_speed_ms": ego_features["summary_stats"]["mean_speed_ms"],
            "ego": ego_features["summary_stats"],
            "lane": lane_stats,
        }
        if lane_stats.get("lane_center_error"):
            summary_stats["lane_center_error"] = lane_stats["lane_center_error"]
        # Top-level table/markdown mirror the ego view so prompts that are not
        # dual-aware still render something coherent.
        return {
            "summary_stats": summary_stats,
            "table_rows": ego_features["table_rows"],
            "markdown_kv": ego_features["markdown_kv"],
            "ego_features": ego_features,
            "lane_features": lane_features,
        }


_FEATURE_COMPUTERS: dict[str, TrajectoryFeatureComputer] = {}


def register_trajectory_feature_computer(
    computer: TrajectoryFeatureComputer,
) -> TrajectoryFeatureComputer:
    """Register a feature strategy by its ``reference_frame`` name."""
    _FEATURE_COMPUTERS[computer.reference_frame] = computer
    return computer


register_trajectory_feature_computer(EgoRigFeatureComputer())
register_trajectory_feature_computer(LaneCenterFeatureComputer())
register_trajectory_feature_computer(DualFrameFeatureComputer())


def compute_trajectory_features(
    trajectory_xy: np.ndarray,
    dt: float = 0.1,
    route_xy: np.ndarray | None = None,
    trajectory_world_xy: np.ndarray | None = None,
    lane_center_lines: list[np.ndarray] | None = None,
    reference_frame: str = "ego_rig",
    lane_reference: str = "auto",
) -> dict:
    """Compute raw numerical trajectory features for LLM evaluation.

    Args:
        trajectory_xy: Trajectory positions in rig frame, shape (T, 2) or (T, 3).
            X = forward, Y = left.
        dt: Time between steps in seconds (default 0.1 = 10Hz).
        route_xy: Optional route/lane-center waypoints in the same rig frame.
        trajectory_world_xy: Optional trajectory positions in map/world frame,
            used with lane_center_lines when available.
        lane_center_lines: Optional map lane-center polylines in the same frame
            as trajectory_world_xy.
        reference_frame: "ego_rig", "lane_center", or "dual". Lane-center mode
            projects trajectory points onto map lane centers or route_xy, and
            falls back to ego_rig if no lane reference is available. Dual mode
            returns both ego-frame and lane-center features (nested under
            "ego_features" and "lane_features") so a prompt can present the
            trajectory in two complementary frames.
        lane_reference: "auto", "map_graph", "map_graph_same_lane", or "route".
            map_graph walks the lane successor graph to build one continuous,
            route-consistent reference (no mid-horizon reference switching).
            map_graph_same_lane evaluates the same candidate starting lanes but
            does not traverse to predecessor or successor lanes. In auto mode,
            map_graph is preferred when map geometry is available; route
            waypoints are the fallback.

    Returns:
        Dict with:
            - summary_stats: Aggregate trajectory statistics
            - table_rows: Per-timestep data (downsampled)
            - markdown_kv: Formatted string for the LLM prompt
    """
    if trajectory_xy is None or len(trajectory_xy) < 3:
        return {"error": "Trajectory too short"}

    xy = _as_xy_array(trajectory_xy)
    if xy is None:
        shape = np.asarray(trajectory_xy).shape
        return {"error": f"Invalid trajectory shape: {shape}"}

    context = TrajectoryFeatureContext(
        xy=xy,
        dt=dt,
        route_xy=route_xy,
        trajectory_world_xy=trajectory_world_xy,
        lane_center_lines=lane_center_lines,
        lane_reference=lane_reference,
    )
    computer = _FEATURE_COMPUTERS.get(reference_frame, _FEATURE_COMPUTERS["ego_rig"])
    return computer.compute(context)
