# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Generate trajectory dynamics sidecar files for extracted frame metadata."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .meta_actions_types import (
    LateralAction,
    LongitudinalAction,
    MetaActionThresholds,
)


METADATA_FILENAME = "metadata.json"
DEFAULT_OUTPUT_FILENAME = "additional_info.json"
DEFAULT_PRECISION = 6
_EPS = 1e-9


@dataclass(frozen=True)
class GenerationResult:
    metadata_path: Path
    output_path: Path
    written: bool
    skipped: bool
    error: str | None = None


@dataclass(frozen=True)
class _TrajectorySpec:
    source_key: str
    coordinate_frame: str
    position_keys: tuple[str, str, str | None]
    axis_descriptions: dict[str, str]
    raw_points: list[dict[str, Any]]


def _round(value: float | None, precision: int | None) -> float | None:
    if value is None:
        return None
    if precision is None:
        return float(value)
    return round(float(value), precision)


def _wrap_angle_rad(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _max_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values, key=abs)


def _classify_acceleration(accel: float | None) -> str | None:
    if accel is None:
        return None
    if accel > 0.15:
        return "accelerating"
    if accel < -0.15:
        return "braking"
    return "coasting"


def _select_trajectory(metadata: dict[str, Any]) -> _TrajectorySpec:
    rig_points = metadata.get("trajectory_xy_rig_frame")
    if isinstance(rig_points, list) and rig_points:
        return _TrajectorySpec(
            source_key="trajectory_xy_rig_frame",
            coordinate_frame="ego_rig_frame",
            position_keys=("rx", "ry", None),
            axis_descriptions={
                "x": "rx, longitudinal offset forward from ego at t0, meters",
                "y": "ry, lateral offset left from ego at t0, meters",
                "z": "not present in rig-frame trajectory",
            },
            raw_points=rig_points,
        )

    global_points = metadata.get("trajectory_poses")
    if isinstance(global_points, list) and global_points:
        return _TrajectorySpec(
            source_key="trajectory_poses",
            coordinate_frame="global_frame",
            position_keys=("x", "y", "z"),
            axis_descriptions={
                "x": "global x position, meters",
                "y": "global y position, meters",
                "z": "global z position, meters",
            },
            raw_points=global_points,
        )

    if isinstance(rig_points, list):
        return _TrajectorySpec(
            source_key="trajectory_xy_rig_frame",
            coordinate_frame="ego_rig_frame",
            position_keys=("rx", "ry", None),
            axis_descriptions={
                "x": "rx, longitudinal offset forward from ego at t0, meters",
                "y": "ry, lateral offset left from ego at t0, meters",
                "z": "not present in rig-frame trajectory",
            },
            raw_points=[],
        )

    return _TrajectorySpec(
        source_key="trajectory_poses",
        coordinate_frame="global_frame",
        position_keys=("x", "y", "z"),
        axis_descriptions={
            "x": "global x position, meters",
            "y": "global y position, meters",
            "z": "global z position, meters",
        },
        raw_points=[],
    )


def _normalize_points(
    spec: _TrajectorySpec,
    precision: int | None,
) -> list[dict[str, Any]]:
    x_key, y_key, z_key = spec.position_keys
    points: list[dict[str, Any]] = []
    cumulative_distance_2d_m = 0.0
    first_timestamp_us: int | None = None

    for index, raw_point in enumerate(spec.raw_points):
        timestamp_us = int(raw_point["timestamp_us"])
        if first_timestamp_us is None:
            first_timestamp_us = timestamp_us
        x_m = float(raw_point[x_key])
        y_m = float(raw_point[y_key])
        z_m = float(raw_point[z_key]) if z_key is not None else 0.0

        if points:
            prev = points[-1]["position_m"]
            cumulative_distance_2d_m += math.hypot(
                x_m - prev["x"],
                y_m - prev["y"],
            )

        points.append(
            {
                "index": index,
                "timestamp_us": timestamp_us,
                "relative_time_s": _round(
                    (timestamp_us - first_timestamp_us) / 1e6,
                    precision,
                ),
                "position_m": {
                    "x": _round(x_m, precision),
                    "y": _round(y_m, precision),
                    "z": _round(z_m, precision) if z_key is not None else None,
                },
                "cumulative_distance_2d_m": _round(
                    cumulative_distance_2d_m,
                    precision,
                ),
            }
        )

    return points


def _build_segments(
    points: list[dict[str, Any]],
    precision: int | None,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []

    for index, (start, end) in enumerate(zip(points, points[1:])):
        start_pos = start["position_m"]
        end_pos = end["position_m"]
        dt_s = (end["timestamp_us"] - start["timestamp_us"]) / 1e6
        if dt_s <= 0.0:
            raise ValueError("trajectory timestamps must be strictly increasing")

        dx = end_pos["x"] - start_pos["x"]
        dy = end_pos["y"] - start_pos["y"]
        start_z = start_pos["z"] or 0.0
        end_z = end_pos["z"] or 0.0
        dz = end_z - start_z
        distance_2d_m = math.hypot(dx, dy)
        distance_3d_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        vx_mps = dx / dt_s
        vy_mps = dy / dt_s
        vz_mps = dz / dt_s
        speed_mps = distance_2d_m / dt_s
        speed_3d_mps = distance_3d_m / dt_s
        heading_rad = math.atan2(dy, dx) if distance_2d_m > _EPS else None
        mid_timestamp_us = (start["timestamp_us"] + end["timestamp_us"]) // 2

        segments.append(
            {
                "index": index,
                "start_index": start["index"],
                "end_index": end["index"],
                "start_timestamp_us": start["timestamp_us"],
                "end_timestamp_us": end["timestamp_us"],
                "mid_timestamp_us": mid_timestamp_us,
                "relative_mid_time_s": _round(
                    (
                        (start["relative_time_s"] + end["relative_time_s"])
                        / 2.0
                    ),
                    precision,
                ),
                "dt_s": _round(dt_s, precision),
                "delta_m": {
                    "x": _round(dx, precision),
                    "y": _round(dy, precision),
                    "z": _round(dz, precision),
                },
                "distance_2d_m": _round(distance_2d_m, precision),
                "distance_3d_m": _round(distance_3d_m, precision),
                "cumulative_distance_start_2d_m": start[
                    "cumulative_distance_2d_m"
                ],
                "cumulative_distance_end_2d_m": end["cumulative_distance_2d_m"],
                "velocity_mps": {
                    "x": _round(vx_mps, precision),
                    "y": _round(vy_mps, precision),
                    "z": _round(vz_mps, precision),
                    "speed_2d": _round(speed_mps, precision),
                    "speed_3d": _round(speed_3d_mps, precision),
                    "speed_kph": _round(speed_mps * 3.6, precision),
                    "speed_mph": _round(speed_mps * 2.2369362920544, precision),
                },
                "heading_rad": _round(heading_rad, precision),
                "heading_deg": _round(
                    math.degrees(heading_rad) if heading_rad is not None else None,
                    precision,
                ),
                "acceleration_from_previous_mps2": None,
                "heading_rate_from_previous_radps": None,
                "curvature_from_previous_1pm": None,
                "jerk_from_previous_mps3": None,
                "motion_class": None,
            }
        )

    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        center_dt_s = (
            current["mid_timestamp_us"] - previous["mid_timestamp_us"]
        ) / 1e6
        if center_dt_s <= 0.0:
            raise ValueError("segment midpoints must be strictly increasing")

        prev_velocity = previous["velocity_mps"]
        curr_velocity = current["velocity_mps"]
        ax_mps2 = (curr_velocity["x"] - prev_velocity["x"]) / center_dt_s
        ay_mps2 = (curr_velocity["y"] - prev_velocity["y"]) / center_dt_s
        az_mps2 = (curr_velocity["z"] - prev_velocity["z"]) / center_dt_s
        accel_2d_mps2 = math.hypot(ax_mps2, ay_mps2)
        longitudinal_accel_mps2 = (
            curr_velocity["speed_2d"] - prev_velocity["speed_2d"]
        ) / center_dt_s

        heading_delta_rad: float | None = None
        heading_rate_radps: float | None = None
        lateral_accel_mps2: float | None = None
        curvature_1pm: float | None = None
        if (
            previous["heading_rad"] is not None
            and current["heading_rad"] is not None
        ):
            heading_delta_rad = _wrap_angle_rad(
                current["heading_rad"] - previous["heading_rad"]
            )
            heading_rate_radps = heading_delta_rad / center_dt_s
            mean_speed_mps = (
                curr_velocity["speed_2d"] + prev_velocity["speed_2d"]
            ) / 2.0
            lateral_accel_mps2 = mean_speed_mps * heading_rate_radps

            prev_mid_x = (
                previous["cumulative_distance_start_2d_m"]
                + previous["cumulative_distance_end_2d_m"]
            ) / 2.0
            curr_mid_x = (
                current["cumulative_distance_start_2d_m"]
                + current["cumulative_distance_end_2d_m"]
            ) / 2.0
            distance_between_segment_centers_m = curr_mid_x - prev_mid_x
            if abs(distance_between_segment_centers_m) > _EPS:
                curvature_1pm = heading_delta_rad / distance_between_segment_centers_m

        acceleration = {
            "x": _round(ax_mps2, precision),
            "y": _round(ay_mps2, precision),
            "z": _round(az_mps2, precision),
            "magnitude_2d": _round(accel_2d_mps2, precision),
            "longitudinal": _round(longitudinal_accel_mps2, precision),
            "lateral": _round(lateral_accel_mps2, precision),
        }
        current["acceleration_from_previous_mps2"] = acceleration
        current["heading_rate_from_previous_radps"] = _round(
            heading_rate_radps,
            precision,
        )
        current["curvature_from_previous_1pm"] = _round(curvature_1pm, precision)
        current["motion_class"] = _classify_acceleration(longitudinal_accel_mps2)

        if index > 1:
            prev_accel = previous["acceleration_from_previous_mps2"]
            if prev_accel is not None:
                jerk_x_mps3 = (acceleration["x"] - prev_accel["x"]) / center_dt_s
                jerk_y_mps3 = (acceleration["y"] - prev_accel["y"]) / center_dt_s
                jerk_z_mps3 = (acceleration["z"] - prev_accel["z"]) / center_dt_s
                current["jerk_from_previous_mps3"] = {
                    "x": _round(jerk_x_mps3, precision),
                    "y": _round(jerk_y_mps3, precision),
                    "z": _round(jerk_z_mps3, precision),
                    "magnitude_2d": _round(
                        math.hypot(jerk_x_mps3, jerk_y_mps3),
                        precision,
                    ),
                }

    return segments


def _build_summary(
    metadata: dict[str, Any],
    points: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    precision: int | None,
) -> dict[str, Any]:
    speed_values = [segment["velocity_mps"]["speed_2d"] for segment in segments]
    distance_values = [segment["distance_2d_m"] for segment in segments]
    dt_values = [segment["dt_s"] for segment in segments]
    acceleration_values = [
        segment["acceleration_from_previous_mps2"]["longitudinal"]
        for segment in segments
        if segment["acceleration_from_previous_mps2"] is not None
    ]
    lateral_accel_values = [
        segment["acceleration_from_previous_mps2"]["lateral"]
        for segment in segments
        if segment["acceleration_from_previous_mps2"] is not None
        and segment["acceleration_from_previous_mps2"]["lateral"] is not None
    ]
    jerk_values = [
        segment["jerk_from_previous_mps3"]["magnitude_2d"]
        for segment in segments
        if segment["jerk_from_previous_mps3"] is not None
    ]
    curvature_values = [
        segment["curvature_from_previous_1pm"]
        for segment in segments
        if segment["curvature_from_previous_1pm"] is not None
    ]

    first_timestamp_us = points[0]["timestamp_us"] if points else None
    last_timestamp_us = points[-1]["timestamp_us"] if points else None
    time_horizon_s = (
        (last_timestamp_us - first_timestamp_us) / 1e6
        if first_timestamp_us is not None and last_timestamp_us is not None
        else 0.0
    )
    final_position = points[-1]["position_m"] if points else None
    first_position = points[0]["position_m"] if points else None
    net_displacement_2d_m = None
    if first_position is not None and final_position is not None:
        net_displacement_2d_m = math.hypot(
            final_position["x"] - first_position["x"],
            final_position["y"] - first_position["y"],
        )

    headings = [
        segment["heading_rad"]
        for segment in segments
        if segment["heading_rad"] is not None
    ]
    heading_change_rad = None
    if len(headings) >= 2:
        heading_change_rad = _wrap_angle_rad(headings[-1] - headings[0])

    lateral_offsets = [
        abs(point["position_m"]["y"])
        for point in points
        if point["position_m"]["y"] is not None
    ]
    monotonic_timestamps = all(
        later["timestamp_us"] > earlier["timestamp_us"]
        for earlier, later in zip(points, points[1:])
    )
    dt_uniform = False
    if dt_values:
        dt_uniform = max(dt_values) - min(dt_values) <= 1e-6

    return {
        "clip_id": metadata.get("clip_id"),
        "time_now_us": metadata.get("time_now_us"),
        "time_query_us": metadata.get("time_query_us"),
        "trajectory_status": "ok" if len(points) >= 2 else "insufficient_points",
        "waypoint_count": len(points),
        "segment_count": len(segments),
        "time_horizon_s": _round(time_horizon_s, precision),
        "total_distance_2d_m": _round(sum(distance_values), precision),
        "net_displacement_2d_m": _round(net_displacement_2d_m, precision),
        "final_position_m": final_position,
        "final_forward_offset_m": final_position["x"] if final_position else None,
        "final_lateral_offset_m": final_position["y"] if final_position else None,
        "max_abs_lateral_offset_m": _round(
            max(lateral_offsets) if lateral_offsets else None,
            precision,
        ),
        "average_dt_s": _round(_mean(dt_values), precision),
        "min_dt_s": _round(min(dt_values) if dt_values else None, precision),
        "max_dt_s": _round(max(dt_values) if dt_values else None, precision),
        "dt_is_uniform": dt_uniform,
        "timestamps_strictly_increasing": monotonic_timestamps,
        "min_speed_mps": _round(min(speed_values) if speed_values else None, precision),
        "mean_speed_mps": _round(_mean(speed_values), precision),
        "max_speed_mps": _round(max(speed_values) if speed_values else None, precision),
        "max_speed_kph": _round(
            max(speed_values) * 3.6 if speed_values else None,
            precision,
        ),
        "min_longitudinal_acceleration_mps2": _round(
            min(acceleration_values) if acceleration_values else None,
            precision,
        ),
        "mean_longitudinal_acceleration_mps2": _round(
            _mean(acceleration_values),
            precision,
        ),
        "max_longitudinal_acceleration_mps2": _round(
            max(acceleration_values) if acceleration_values else None,
            precision,
        ),
        "max_abs_longitudinal_acceleration_mps2": _round(
            abs(_max_abs(acceleration_values))
            if acceleration_values and _max_abs(acceleration_values) is not None
            else None,
            precision,
        ),
        "max_abs_lateral_acceleration_mps2": _round(
            abs(_max_abs(lateral_accel_values))
            if lateral_accel_values and _max_abs(lateral_accel_values) is not None
            else None,
            precision,
        ),
        "max_abs_jerk_mps3": _round(
            max(jerk_values) if jerk_values else None,
            precision,
        ),
        "heading_change_rad": _round(heading_change_rad, precision),
        "heading_change_deg": _round(
            math.degrees(heading_change_rad)
            if heading_change_rad is not None
            else None,
            precision,
        ),
        "max_abs_curvature_1pm": _round(
            abs(_max_abs(curvature_values))
            if curvature_values and _max_abs(curvature_values) is not None
            else None,
            precision,
        ),
    }


_DEFAULT_THRESHOLDS = MetaActionThresholds()


def _classify_longitudinal(
    speed_mps: float,
    forward_velocity_mps: float,
    longitudinal_accel_mps2: float | None,
    thresholds: MetaActionThresholds,
) -> LongitudinalAction:
    if speed_mps < thresholds.stop_speed_mps:
        return LongitudinalAction.STOP
    if forward_velocity_mps < -thresholds.reverse_speed_mps:
        return LongitudinalAction.REVERSE
    if longitudinal_accel_mps2 is None:
        return LongitudinalAction.MAINTAIN_SPEED
    if abs(longitudinal_accel_mps2) < thresholds.maintain_accel_mps2:
        return LongitudinalAction.MAINTAIN_SPEED
    if longitudinal_accel_mps2 >= thresholds.strong_accel_mps2:
        return LongitudinalAction.STRONG_ACCELERATE
    if longitudinal_accel_mps2 > 0.0:
        return LongitudinalAction.GENTLE_ACCELERATE
    if longitudinal_accel_mps2 <= -thresholds.strong_accel_mps2:
        return LongitudinalAction.STRONG_DECELERATE
    return LongitudinalAction.GENTLE_DECELERATE


def _classify_lateral(
    forward_velocity_mps: float,
    curvature_1pm: float | None,
    thresholds: MetaActionThresholds,
    speed_mps: float = 0.0,
) -> LateralAction:
    # Sign convention: positive curvature / heading-rate = left turn
    # (counterclockwise yaw, consistent with rig-frame ry pointing left).
    # Gate on speed: at near-stop speeds tiny xy jitter produces large fake
    # curvatures, so anything below the stop threshold is forced to "straight".
    if speed_mps < thresholds.stop_speed_mps:
        return LateralAction.GO_STRAIGHT
    if curvature_1pm is None or abs(curvature_1pm) < thresholds.straight_curvature_1pm:
        return LateralAction.GO_STRAIGHT
    is_left = curvature_1pm > 0.0
    is_sharp = abs(curvature_1pm) >= thresholds.sharp_curvature_1pm
    if forward_velocity_mps < -thresholds.reverse_speed_mps:
        return LateralAction.REVERSE_LEFT if is_left else LateralAction.REVERSE_RIGHT
    if is_sharp:
        return (
            LateralAction.SHARP_STEER_LEFT
            if is_left
            else LateralAction.SHARP_STEER_RIGHT
        )
    return LateralAction.STEER_LEFT if is_left else LateralAction.STEER_RIGHT


def _segment_speed(segment: dict[str, Any]) -> float:
    velocity = segment.get("velocity_mps") or {}
    speed = velocity.get("speed_2d")
    return float(speed) if speed is not None else 0.0


def _segment_forward_velocity(segment: dict[str, Any]) -> float:
    velocity = segment.get("velocity_mps") or {}
    vx = velocity.get("x")
    return float(vx) if vx is not None else 0.0


def _segment_longitudinal_accel(segment: dict[str, Any]) -> float | None:
    accel = segment.get("acceleration_from_previous_mps2")
    if not accel:
        return None
    value = accel.get("longitudinal")
    return float(value) if value is not None else None


def _segment_curvature(segment: dict[str, Any]) -> float | None:
    value = segment.get("curvature_from_previous_1pm")
    return float(value) if value is not None else None


def _build_meta_actions(
    segments: list[dict[str, Any]],
    thresholds: MetaActionThresholds,
    coordinate_frame: str,
) -> dict[str, Any]:
    per_segment: list[dict[str, Any]] = []
    for segment in segments:
        speed = _segment_speed(segment)
        forward_v = _segment_forward_velocity(segment)
        a_long = _segment_longitudinal_accel(segment)
        curvature = _segment_curvature(segment)

        longitudinal = _classify_longitudinal(speed, forward_v, a_long, thresholds)
        lateral = _classify_lateral(forward_v, curvature, thresholds, speed_mps=speed)

        per_segment.append(
            {
                "index": segment["index"],
                "start_timestamp_us": segment["start_timestamp_us"],
                "end_timestamp_us": segment["end_timestamp_us"],
                "mid_timestamp_us": segment["mid_timestamp_us"],
                "longitudinal": longitudinal.value,
                "lateral": lateral.value,
            }
        )

    transitions: list[dict[str, Any]] = []
    for previous, current in zip(per_segment, per_segment[1:]):
        if (
            previous["longitudinal"] == current["longitudinal"]
            and previous["lateral"] == current["lateral"]
        ):
            continue
        transitions.append(
            {
                "at_index": current["index"],
                "at_timestamp_us": current["mid_timestamp_us"],
                "from": {
                    "longitudinal": previous["longitudinal"],
                    "lateral": previous["lateral"],
                },
                "to": {
                    "longitudinal": current["longitudinal"],
                    "lateral": current["lateral"],
                },
            }
        )

    longitudinal_counter = Counter(p["longitudinal"] for p in per_segment)
    lateral_counter = Counter(p["lateral"] for p in per_segment)

    dominant_longitudinal = (
        longitudinal_counter.most_common(1)[0][0] if longitudinal_counter else None
    )
    dominant_lateral = (
        lateral_counter.most_common(1)[0][0] if lateral_counter else None
    )

    lateral_reliable = coordinate_frame == "ego_rig_frame"

    return {
        "vocabulary": {
            "longitudinal": [a.value for a in LongitudinalAction if a is not LongitudinalAction.UNKNOWN],
            "lateral": [a.value for a in LateralAction if a is not LateralAction.UNKNOWN],
        },
        "thresholds": {
            "stop_speed_mps": thresholds.stop_speed_mps,
            "reverse_speed_mps": thresholds.reverse_speed_mps,
            "maintain_accel_mps2": thresholds.maintain_accel_mps2,
            "strong_accel_mps2": thresholds.strong_accel_mps2,
            "straight_curvature_1pm": thresholds.straight_curvature_1pm,
            "sharp_curvature_1pm": thresholds.sharp_curvature_1pm,
        },
        "coordinate_frame": coordinate_frame,
        # Lateral side ("left" vs "right") is only meaningful in the ego rig frame;
        # in global frame the curvature sign gives yaw direction, not vehicle-frame side.
        "lateral_side_reliable": lateral_reliable,
        "dominant": {
            "longitudinal": dominant_longitudinal,
            "lateral": dominant_lateral,
        },
        "longitudinal_distribution": dict(longitudinal_counter),
        "lateral_distribution": dict(lateral_counter),
        "transition_count": len(transitions),
        "transitions": transitions,
        "per_segment": per_segment,
    }


def build_additional_info(
    metadata: dict[str, Any],
    *,
    metadata_path: str | Path | None = None,
    precision: int | None = DEFAULT_PRECISION,
    meta_action_thresholds: MetaActionThresholds | None = None,
) -> dict[str, Any]:
    """Build a trajectory dynamics sidecar payload from a metadata dictionary."""

    spec = _select_trajectory(metadata)
    points = _normalize_points(spec, precision)
    segments = _build_segments(points, precision)
    summary = _build_summary(metadata, points, segments, precision)
    thresholds = meta_action_thresholds or _DEFAULT_THRESHOLDS
    meta_actions = _build_meta_actions(segments, thresholds, spec.coordinate_frame)

    return {
        "schema_version": 2,
        "kind": "alpasim_trajectory_additional_info",
        "metadata_file": str(metadata_path) if metadata_path is not None else None,
        "trajectory_source": spec.source_key,
        "coordinate_frame": spec.coordinate_frame,
        "units": {
            "position": "m",
            "time": "s",
            "timestamp": "us",
            "velocity": "m/s",
            "acceleration": "m/s^2",
            "jerk": "m/s^3",
            "heading": "rad",
            "curvature": "1/m",
        },
        "axis_descriptions": spec.axis_descriptions,
        "summary": summary,
        "meta_actions": meta_actions,
        "waypoints": points,
        "segments": segments,
    }


def find_metadata_files(
    extracted_frames_dir: str | Path,
    *,
    metadata_name: str = METADATA_FILENAME,
) -> list[Path]:
    root = Path(extracted_frames_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")
    return sorted(path for path in root.rglob(metadata_name) if path.is_file())


def generate_additional_info_files(
    extracted_frames_dir: str | Path,
    *,
    output_name: str = DEFAULT_OUTPUT_FILENAME,
    overwrite: bool = True,
    dry_run: bool = False,
    strict: bool = False,
    precision: int | None = DEFAULT_PRECISION,
) -> list[GenerationResult]:
    """Generate one sidecar JSON file next to each discovered metadata file."""

    if output_name == METADATA_FILENAME:
        raise ValueError("output_name must not overwrite metadata.json")

    results: list[GenerationResult] = []
    for metadata_path in find_metadata_files(extracted_frames_dir):
        output_path = metadata_path.with_name(output_name)
        if output_path.exists() and not overwrite:
            results.append(
                GenerationResult(
                    metadata_path=metadata_path,
                    output_path=output_path,
                    written=False,
                    skipped=True,
                )
            )
            continue

        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            additional_info = build_additional_info(
                metadata,
                metadata_path=metadata_path,
                precision=precision,
            )
            if not dry_run:
                with output_path.open("w", encoding="utf-8") as file:
                    json.dump(additional_info, file, indent=2)
                    file.write("\n")
            results.append(
                GenerationResult(
                    metadata_path=metadata_path,
                    output_path=output_path,
                    written=not dry_run,
                    skipped=False,
                )
            )
        except Exception as exc:
            if strict:
                raise
            results.append(
                GenerationResult(
                    metadata_path=metadata_path,
                    output_path=output_path,
                    written=False,
                    skipped=False,
                    error=str(exc),
                )
            )

    return results


def _parse_precision(value: str) -> int | None:
    if value.lower() == "none":
        return None
    precision = int(value)
    if precision < 0:
        raise argparse.ArgumentTypeError("precision must be >= 0 or 'none'")
    return precision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate additional_info.json sidecars with velocity, acceleration, "
            "and summary trajectory dynamics for extracted frame metadata."
        )
    )
    parser.add_argument(
        "extracted_frames_dir",
        type=Path,
        help="Directory to recursively search for metadata.json files.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Sidecar filename to write next to each metadata file. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip sidecars that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be generated without writing files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid metadata file.",
    )
    parser.add_argument(
        "--precision",
        type=_parse_precision,
        default=DEFAULT_PRECISION,
        help="Decimal places for float values, or 'none'. Default: 6",
    )
    args = parser.parse_args(argv)

    results = generate_additional_info_files(
        args.extracted_frames_dir,
        output_name=args.output_name,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
        strict=args.strict,
        precision=args.precision,
    )
    written = sum(result.written for result in results)
    skipped = sum(result.skipped for result in results)
    errors = [result for result in results if result.error is not None]

    action = "Would write" if args.dry_run else "Wrote"
    print(
        f"{action} {written if not args.dry_run else len(results) - skipped - len(errors)} "
        f"{args.output_name} files; skipped {skipped}; errors {len(errors)}."
    )
    for result in errors[:10]:
        print(f"ERROR {result.metadata_path}: {result.error}")
    if len(errors) > 10:
        print(f"... {len(errors) - 10} more errors omitted")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
