# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Lane-relative trajectory projection.

Pure-geometry helpers that project a trajectory onto a lane-center reference
and compute the signed lateral ``offset_m`` and arc-length progress ``s_lane``.

This is the single source of truth for "how lane offset is computed":
``cot_analysis`` consumes these functions to build the lane-relative features
the CoT judge scores, and ``extract_frame`` consumes the same functions to
render the lane-relative visualization. Keeping them here (in ``alpasim_utils``,
the lower package both depend on) keeps the two in lock-step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def as_xy_array(points: np.ndarray | None) -> np.ndarray | None:
    """Validate and trim an array-like point sequence to XY coordinates."""
    if points is None:
        return None
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return None
    return xy[:, :2]


def project_points_to_polyline(
    points_xy: np.ndarray, route_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto a route polyline.

    Returns:
        Tuple of (s_lane, offset_m). offset_m is signed lateral distance from
        route centerline, positive to the left of route direction.
    """
    segments = route_xy[1:] - route_xy[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    keep = segment_lengths > 1.0e-6
    if not np.any(keep):
        raise ValueError("Route has no nonzero-length segments")

    starts = route_xy[:-1][keep]
    segments = segments[keep]
    segment_lengths = segment_lengths[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))

    s_lane = []
    offsets = []
    for point in points_xy:
        rel = point - starts
        t = np.sum(rel * segments, axis=1) / (segment_lengths**2)
        t = np.clip(t, 0.0, 1.0)
        projections = starts + t[:, None] * segments
        deltas = point - projections
        distances_sq = np.sum(deltas**2, axis=1)
        best = int(np.argmin(distances_sq))

        tangent = segments[best] / segment_lengths[best]
        offset = tangent[0] * deltas[best, 1] - tangent[1] * deltas[best, 0]
        s_lane.append(cumulative[best] + t[best] * segment_lengths[best])
        offsets.append(offset)

    return np.asarray(s_lane), np.asarray(offsets)


def project_points_to_polyline_monotone(
    points_xy: np.ndarray,
    route_xy: np.ndarray,
    backward_slack_m: float = 1.0,
    forward_window_m: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a point sequence onto a route with arc-length continuity.

    Like project_points_to_polyline, but each point only considers segments
    within a window of the previous point's arc length. Even on a single
    connected route, globally-nearest projection can jump where the route
    passes near itself (intersections, junction kinks) — the continuity window
    keeps s smooth, which is what the accel proxy differentiates.
    """
    segments = route_xy[1:] - route_xy[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    keep = segment_lengths > 1.0e-6
    if not np.any(keep):
        raise ValueError("Route has no nonzero-length segments")

    starts = route_xy[:-1][keep]
    segments = segments[keep]
    segment_lengths = segment_lengths[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))

    s_lane = []
    offsets = []
    prev_s = None
    for point in points_xy:
        rel = point - starts
        t = np.sum(rel * segments, axis=1) / (segment_lengths**2)
        t = np.clip(t, 0.0, 1.0)
        projections = starts + t[:, None] * segments
        deltas = point - projections
        distances_sq = np.sum(deltas**2, axis=1)

        if prev_s is None:
            best = int(np.argmin(distances_sq))
        else:
            seg_s = cumulative + t * segment_lengths
            in_window = (seg_s >= prev_s - backward_slack_m) & (
                seg_s <= prev_s + forward_window_m
            )
            if np.any(in_window):
                masked = np.where(in_window, distances_sq, np.inf)
                best = int(np.argmin(masked))
            else:
                best = int(np.argmin(distances_sq))

        tangent = segments[best] / segment_lengths[best]
        offset = tangent[0] * deltas[best, 1] - tangent[1] * deltas[best, 0]
        s = cumulative[best] + t[best] * segment_lengths[best]
        s_lane.append(s)
        offsets.append(offset)
        prev_s = s

    return np.asarray(s_lane), np.asarray(offsets)


def polyline_min_distance_and_tangent(
    line_xy: np.ndarray, point: np.ndarray
) -> tuple[float, np.ndarray]:
    """Minimum distance from a point to a polyline, plus the unit tangent there."""
    segments = line_xy[1:] - line_xy[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    keep = segment_lengths > 1.0e-6
    if not np.any(keep):
        return float("inf"), np.array([1.0, 0.0])

    starts = line_xy[:-1][keep]
    segments = segments[keep]
    segment_lengths = segment_lengths[keep]

    rel = point - starts
    t = np.clip(np.sum(rel * segments, axis=1) / (segment_lengths**2), 0.0, 1.0)
    projections = starts + t[:, None] * segments
    distances = np.linalg.norm(point - projections, axis=1)
    best = int(np.argmin(distances))
    tangent = segments[best] / segment_lengths[best]
    return float(distances[best]), tangent


def polyline_arclength_interpolate(
    line_xy: np.ndarray, s_values: np.ndarray
) -> np.ndarray:
    """Sample XY points along a polyline at the given arc lengths.

    Used to recover the foot of each trajectory point's projection (the point on
    the lane center the offset is measured to) from its ``s_lane`` value, so the
    visualization can draw the offset connector lines.
    """
    seg = line_xy[1:] - line_xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cumulative[-1])

    s = np.clip(np.asarray(s_values, dtype=float), 0.0, total)
    idx = np.clip(np.searchsorted(cumulative, s, side="right") - 1, 0, len(seg) - 1)
    seg_len_at = seg_len[idx]
    frac = np.where(seg_len_at > 1.0e-9, (s - cumulative[idx]) / seg_len_at, 0.0)
    return line_xy[:-1][idx] + frac[:, None] * seg[idx]


def build_route_consistent_reference(
    trajectory_xy: np.ndarray,
    lane_center_lines: list[np.ndarray],
    endpoint_tol_m: float = 3.0,
    heading_dot_min: float = 0.0,
    coverage_margin_m: float = 5.0,
    max_lanes_per_path: int = 12,
    max_candidate_paths: int = 64,
    max_start_candidates: int = 3,
    stay_on_start_lane: bool = False,
) -> tuple[np.ndarray, int]:
    """Build a single route-consistent lane-center reference (Werling-style).

    Instead of snapping each trajectory point to its nearest centerline (which
    can switch lanes mid-horizon and create spurious jumps in s/accel), this
    walks the lane successor graph from the ego's current lane to build one
    continuous reference path. Connectivity is inferred geometrically: lane B
    succeeds lane A when B's start endpoint lies within endpoint_tol_m of A's
    end endpoint with continuing heading. Where the graph forks (merge/split),
    the branch minimizing total lateral deviation over the full horizon wins.
    When ``stay_on_start_lane`` is true, the same starting-lane candidates are
    evaluated without traversing to predecessor or successor lanes.

    Returns:
        Tuple of (reference_xy, lanes_used) for the best path.
    """
    lanes = []
    for line in lane_center_lines:
        line_xy = as_xy_array(line)
        if line_xy is not None and len(line_xy) >= 2:
            lanes.append(line_xy)
    if not lanes:
        raise ValueError("No valid map lane-center lines available")

    # Lane direction is not guaranteed in the geometry dump, so each node is a
    # (lane index, orientation) pair.
    def node_coords(lane_idx: int, flipped: bool) -> np.ndarray:
        return lanes[lane_idx][::-1] if flipped else lanes[lane_idx]

    nodes = []
    for lane_idx in range(len(lanes)):
        for flipped in (False, True):
            coords = node_coords(lane_idx, flipped)
            start_seg = coords[1] - coords[0]
            end_seg = coords[-1] - coords[-2]
            start_norm = np.linalg.norm(start_seg)
            end_norm = np.linalg.norm(end_seg)
            if start_norm < 1.0e-6 or end_norm < 1.0e-6:
                continue
            nodes.append(
                {
                    "lane": lane_idx,
                    "flipped": flipped,
                    "start": coords[0],
                    "end": coords[-1],
                    "start_tangent": start_seg / start_norm,
                    "end_tangent": end_seg / end_norm,
                }
            )
    if not nodes:
        raise ValueError("No usable lane-center polylines")

    # Ego heading disambiguates the orientation of the starting lane.
    traj_start = trajectory_xy[0]
    heading = trajectory_xy[-1] - trajectory_xy[0]
    heading_norm = np.linalg.norm(heading)
    if heading_norm < 0.5:
        steps = np.diff(trajectory_xy, axis=0)
        step_norms = np.linalg.norm(steps, axis=1)
        moving = np.nonzero(step_norms > 0.05)[0]
        if len(moving) > 0:
            heading = steps[moving[0]]
            heading_norm = np.linalg.norm(heading)
    heading_unit = heading / heading_norm if heading_norm > 1.0e-6 else None

    # Start candidates: lanes nearest to the ego, oriented along its heading.
    start_candidates = []
    for node in nodes:
        coords = node_coords(node["lane"], node["flipped"])
        dist, tangent = polyline_min_distance_and_tangent(coords, traj_start)
        if heading_unit is not None and float(np.dot(tangent, heading_unit)) <= 0.0:
            continue
        start_candidates.append((dist, node))
    if not start_candidates:
        raise ValueError("Could not match trajectory start to any lane center")
    start_candidates.sort(key=lambda item: item[0])
    min_start_dist = start_candidates[0][0]
    start_nodes = [
        node
        for dist, node in start_candidates[:max_start_candidates]
        if dist <= max(3.0, min_start_dist + 1.0)
    ]

    traj_steps = np.diff(trajectory_xy, axis=0)
    traj_length = float(np.sum(np.linalg.norm(traj_steps, axis=1)))

    def _connects(tail: dict, head: dict) -> bool:
        """Whether head's start continues from tail's end."""
        connector = head["start"] - tail["end"]
        gap = float(np.linalg.norm(connector))
        if gap > endpoint_tol_m:
            return False
        if float(np.dot(tail["end_tangent"], head["start_tangent"])) <= heading_dot_min:
            return False
        # Reject laterally offset connections (adjacent lane heads): for a real
        # successor the connector itself continues along the lane direction.
        if gap > 0.5 and float(np.dot(connector / gap, tail["end_tangent"])) <= 0.5:
            return False
        return True

    def successors(node: dict, visited_lanes: set) -> list:
        return [
            cand
            for cand in nodes
            if cand["lane"] not in visited_lanes and _connects(node, cand)
        ]

    def predecessors(node: dict, visited_lanes: set) -> list:
        return [
            cand
            for cand in nodes
            if cand["lane"] not in visited_lanes and _connects(cand, node)
        ]

    def merged_coords(path: list) -> np.ndarray:
        points = []
        for node in path:
            coords = node_coords(node["lane"], node["flipped"])
            if points:
                # Trim successor points that overlap backward past the junction,
                # which would fold the reference back on itself.
                prev_end = np.asarray(points[-1])
                prev_tangent = node["start_tangent"]
                if len(points) >= 2:
                    prev_seg = prev_end - np.asarray(points[-2])
                    prev_norm = np.linalg.norm(prev_seg)
                    if prev_norm > 1.0e-6:
                        prev_tangent = prev_seg / prev_norm
                while len(coords) >= 2 and (
                    float(np.dot(coords[0] - prev_end, prev_tangent)) <= 0.0
                ):
                    coords = coords[1:]
                if np.linalg.norm(coords[0] - prev_end) < 1.0e-6:
                    coords = coords[1:]
                if len(coords) == 0:
                    continue
            points.extend(coords.tolist())
        return np.asarray(points, dtype=float)

    def path_covers_trajectory(path_xy: np.ndarray) -> bool:
        s_start, _ = project_points_to_polyline(traj_start[None, :], path_xy)
        total_length = float(np.sum(np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)))
        return total_length - float(s_start[0]) >= traj_length + coverage_margin_m

    def extend_backward(node: dict) -> list:
        """Prepend predecessor lanes so the trajectory start projects strictly
        inside the path instead of clamping to its first vertex (which would
        zero out s-progress and spike the accel proxy)."""
        path = [node]
        for _ in range(3):
            first = node_coords(path[0]["lane"], path[0]["flipped"])
            s_start, _ = project_points_to_polyline(traj_start[None, :], first)
            if float(s_start[0]) > 2.0:
                break
            preds = predecessors(path[0], {n["lane"] for n in path})
            if not preds:
                break
            # Backward branch choice barely matters (it only anchors the first
            # moments of the horizon); take the closest-gap predecessor.
            preds.sort(
                key=lambda cand: float(np.linalg.norm(path[0]["start"] - cand["end"]))
            )
            path.insert(0, preds[0])
        return path

    # Depth-first enumeration of maximal paths through the successor graph.
    # The same-lane variant intentionally skips backward and forward traversal:
    # every candidate reference consists only of its oriented starting lane.
    complete_paths: list[list]
    if stay_on_start_lane:
        complete_paths = [[node] for node in start_nodes]
    else:
        complete_paths = []
        stack = [extend_backward(node) for node in start_nodes]
        while stack and len(complete_paths) < max_candidate_paths:
            path = stack.pop()
            path_xy = merged_coords(path)
            if path_covers_trajectory(path_xy) or len(path) >= max_lanes_per_path:
                complete_paths.append(path)
                continue
            succ = successors(path[-1], {n["lane"] for n in path})
            if not succ:
                # Dead end (map clipped to a radius): still a valid candidate.
                complete_paths.append(path)
                continue
            for cand in succ:
                stack.append(path + [cand])

    # Branch selection: minimize total lateral deviation over the full horizon.
    best_path_xy = None
    best_lanes = 0
    best_cost = float("inf")
    for path in complete_paths:
        path_xy = merged_coords(path)
        if len(path_xy) < 2:
            continue
        try:
            _, offsets = project_points_to_polyline_monotone(trajectory_xy, path_xy)
        except ValueError:
            continue
        cost = float(np.sum(offsets**2))
        if cost < best_cost:
            best_cost = cost
            best_path_xy = path_xy
            best_lanes = len(path)
    if best_path_xy is None:
        raise ValueError("No connected lane path matched the trajectory")
    return best_path_xy, best_lanes


def merge_map_lane_centers_for_trajectory(
    trajectory_xy: np.ndarray,
    lane_center_lines: list[np.ndarray],
) -> tuple[np.ndarray, int]:
    """Build a continuous map lane-center reference near the trajectory.

    This mirrors the tutorial diagnostic script: find the closest map lane-center
    line for each trajectory point, keep those lines in trajectory order, and
    reverse each added line if needed for continuity.
    """
    valid_lines = []
    for line in lane_center_lines:
        line_xy = as_xy_array(line)
        if line_xy is not None and len(line_xy) >= 2:
            valid_lines.append(line_xy)
    if not valid_lines:
        raise ValueError("No valid map lane-center lines available")

    closest_line_indices = []
    for point in trajectory_xy:
        best_idx = None
        best_dist = float("inf")
        for idx, line_xy in enumerate(valid_lines):
            dists = np.linalg.norm(line_xy - point, axis=1)
            min_dist = float(np.min(dists))
            if min_dist < best_dist:
                best_dist = min_dist
                best_idx = idx
        if best_idx is not None:
            closest_line_indices.append(best_idx)

    seen = []
    for idx in closest_line_indices:
        if idx not in seen:
            seen.append(idx)
    if not seen:
        raise ValueError("Could not match trajectory points to map lane centers")

    merged_points = []
    for idx in seen:
        coords = valid_lines[idx]
        if merged_points:
            last_pt = np.asarray(merged_points[-1])
            dist_to_first = np.linalg.norm(coords[0] - last_pt)
            dist_to_last = np.linalg.norm(coords[-1] - last_pt)
            if dist_to_last < dist_to_first:
                coords = coords[::-1]
        merged_points.extend(coords.tolist())

    merged_xy = np.asarray(merged_points, dtype=float)
    if len(merged_xy) < 2:
        raise ValueError("Merged map lane-center polyline is too short")
    return merged_xy, len(seen)


def lane_center_lines_from_geometry(
    geometry: dict[str, Any],
) -> list[np.ndarray] | None:
    """Extract ``road_lane_center`` polylines from an extracted-frame geometry dict.

    ``geometry`` is the parsed contents of a ``trajectory_plot_geometry.json``
    sidecar (as written by the extract_frame eval visualization). Returns a list
    of (N, 2) world-frame polylines, or None when none are present.
    """
    lane_centers = []
    for linestring in geometry.get("map_linestrings", []):
        if linestring.get("type") != "road_lane_center":
            continue
        coords = np.asarray(linestring.get("coords", []), dtype=float)
        if coords.ndim == 2 and coords.shape[0] >= 2 and coords.shape[1] >= 2:
            lane_centers.append(coords[:, :2])
    return lane_centers or None


@dataclass(frozen=True)
class LaneRelativeResult:
    """Lane-relative projection of a trajectory, ready for plotting/inspection."""

    reference_xy: np.ndarray  # (M, 2) merged lane-center reference polyline
    s_lane: np.ndarray  # (T,) arc length of each point's projection along reference
    offsets: np.ndarray  # (T,) signed lateral offset, + = left of lane direction
    projections: np.ndarray  # (T, 2) foot of each projection on the reference
    method: str  # "map_graph" or "map_geometry"
    lane_segments_used: int


def compute_lane_relative_trajectory(
    trajectory_xy: np.ndarray,
    lane_center_lines: list[np.ndarray],
    prefer: str = "map_graph",
) -> LaneRelativeResult:
    """Project a world-frame trajectory onto a lane-center reference.

    This is the high-level entry point shared by the CoT analysis features and
    the extract_frame visualization. With ``prefer="map_graph"`` it builds the
    route-consistent lane-graph reference the judge uses (dual mode default) and
    projects with arc-length continuity; it falls back to the nearest-lane merge
    if the graph cannot be built. ``prefer="map_geometry"`` uses the merge
    directly.

    Args:
        trajectory_xy: (T, >=2) trajectory positions in the world/map frame.
        lane_center_lines: world-frame ``road_lane_center`` polylines.
        prefer: "map_graph" (route-consistent, matches the judge) or
            "map_geometry" (nearest-lane merge).

    Returns:
        A LaneRelativeResult. Raises ValueError if no reference can be built.
    """
    xy = as_xy_array(trajectory_xy)
    if xy is None or len(xy) < 2:
        raise ValueError("Trajectory must have at least 2 XY points")
    if not lane_center_lines:
        raise ValueError("No lane-center lines provided")

    reference_xy: np.ndarray | None = None
    lanes_used = 0
    method = ""
    if prefer == "map_graph":
        try:
            reference_xy, lanes_used = build_route_consistent_reference(
                xy, lane_center_lines
            )
            method = "map_graph"
        except ValueError:
            reference_xy = None
    if reference_xy is None:
        reference_xy, lanes_used = merge_map_lane_centers_for_trajectory(
            xy, lane_center_lines
        )
        method = "map_geometry"

    monotone = method == "map_graph"
    if monotone:
        s_lane, offsets = project_points_to_polyline_monotone(xy, reference_xy)
    else:
        s_lane, offsets = project_points_to_polyline(xy, reference_xy)
    projections = polyline_arclength_interpolate(reference_xy, s_lane)

    return LaneRelativeResult(
        reference_xy=reference_xy,
        s_lane=s_lane,
        offsets=offsets,
        projections=projections,
        method=method,
        lane_segments_used=lanes_used,
    )
