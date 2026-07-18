# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from dataclasses import dataclass

import numpy as np

from rollout_monitoring.analysis import (
    OutcomeSeries,
    analyze_rollout,
)
from rollout_monitoring.monitor import (
    MonitorStep,
    _flatten_cot,
    _poses_to_metadata,
)

US = 1_000_000


def _series(
    metrics: dict[str, list[tuple[float, float]]], *, max_dist_to_gt_m: float = 4.0
):
    """Build an OutcomeSeries from {metric: [(t_s, value), ...]} like load_outcome_series."""
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_ts: list[int] = []
    for name, pairs in metrics.items():
        ts = np.array([int(t * US) for t, _ in pairs], dtype=np.int64)
        val = np.array([v for _, v in pairs], dtype=float)
        order = np.argsort(ts)
        raw[name] = (ts[order], val[order])
        all_ts.extend(ts.tolist())
    eval_rel = raw.get("eval_relevant")
    if eval_rel is not None and (eval_rel[1] > 0).any():
        eval_start = int(eval_rel[0][eval_rel[1] > 0].min())
    else:
        eval_start = min(all_ts)
    eval_end = max(all_ts)
    base = OutcomeSeries(eval_start, eval_end, False, raw)
    ends = [eval_end]
    for m in ("collision_any", "offroad"):
        t = base._first_event(m, 0.0)
        if t is not None:
            ends.append(t)
    td = base._first_event("dist_to_gt_trajectory", max_dist_to_gt_m, ge=True)
    if td is not None:
        ends.append(td)
    s = OutcomeSeries(eval_start, min(ends), False, raw)
    s.img_black = s._fires_in_window("img_is_black")
    return s


def _flag_step(t_s: float, *, flag: bool) -> MonitorStep:
    label = "inconsistent" if flag else "consistent"
    return MonitorStep(
        now_us=int(t_s * US),
        n_poses=10,
        has_cot=True,
        scored=True,
        score=0.0 if flag else 1.0,
        label=label,
        longitudinal_verdict="x",
        lateral_verdict="y",
        flag=flag,
    )


def test_rear_collision_is_not_at_fault_due_to_window_truncation():
    # First any-collision is a rear hit at t=10; an at-fault front hit at t=12 is
    # after the window end and must not count (matches the evaluator).
    s = _series(
        {
            "eval_relevant": [(1.0, 1.0), (15.0, 1.0)],
            "collision_any": [(10.0, 1.0)],
            "collision_rear": [(10.0, 1.0)],
            "collision_front": [(12.0, 1.0)],
            "dist_to_gt_trajectory": [(1.0, 0.5), (20.0, 1.0)],
        }
    )
    assert s.failed("collision_any")
    assert s.failure_time_us("collision_any") == 10 * US
    assert not s.failed("offroad_or_collision_at_fault")
    assert not s.failed("collision_at_fault")


def test_dist_to_gt_cutoff_excludes_later_failure():
    # Ego drifts >=4 m from GT at t=8, before a collision at t=12 -> safe.
    s = _series(
        {
            "eval_relevant": [(1.0, 1.0)],
            "dist_to_gt_trajectory": [(1.0, 0.0), (8.0, 4.5), (12.0, 6.0)],
            "collision_any": [(12.0, 1.0)],
        }
    )
    assert s.window_end_us == 8 * US
    assert not s.failed("collision_any")


def test_leading_indicator_lead_time_and_horizon():
    s = _series(
        {
            "eval_relevant": [(1.0, 1.0), (15.0, 1.0)],
            "offroad": [(10.0, 1.0)],
        }
    )
    steps = [
        _flag_step(2.0, flag=True),
        _flag_step(5.0, flag=True),
        _flag_step(11.0, flag=True),
    ]
    r = analyze_rollout(
        steps, s, clip_id="c", rollout_id="r", outcome="offroad", horizon_s=4.0
    )
    assert r.failed and r.failure_time_us == 10 * US
    assert r.flag_before_failure
    assert r.first_flag_us == 2 * US  # earliest pre-failure flag
    assert r.lead_time_s == 8.0
    # flags at 2,5 are before t-H=6 -> not within the 4 s horizon
    assert not r.flag_within_horizon
    # the flag at t=11 is after the failure -> not counted as leading
    assert r.n_flags_followed_by_failure == 0


def test_within_horizon_flag_counts():
    s = _series({"eval_relevant": [(1.0, 1.0), (15.0, 1.0)], "offroad": [(10.0, 1.0)]})
    steps = [_flag_step(8.0, flag=True), _flag_step(9.0, flag=True)]
    r = analyze_rollout(
        steps, s, clip_id="c", rollout_id="r", outcome="offroad", horizon_s=4.0
    )
    assert r.flag_within_horizon
    assert r.n_flags_followed_by_failure == 2
    assert r.lead_time_s == 2.0


def test_safe_rollout_flag_is_false_alarm():
    s = _series({"eval_relevant": [(1.0, 1.0), (15.0, 1.0)]})  # no failure
    steps = [_flag_step(5.0, flag=True)]
    r = analyze_rollout(
        steps, s, clip_id="c", rollout_id="r", outcome="offroad", horizon_s=4.0
    )
    assert not r.failed
    assert r.false_alarm
    assert not r.flag_before_failure


# --- monitor helpers ---


@dataclass
class _Vec:
    x: float
    y: float
    z: float = 0.0


@dataclass
class _Quat:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class _Pose:
    vec: _Vec
    quat: _Quat


@dataclass
class _Waypoint:
    pose: _Pose
    timestamp_us: int


def test_flatten_cot_handles_str_list_and_none():
    assert _flatten_cot(None) is None
    assert _flatten_cot("a") == "a"
    assert _flatten_cot(["a", "b"]) == "a b"
    assert _flatten_cot([["a"], "b"]) == "a b"


def test_poses_to_metadata_rig_frame_is_anchored():
    # Straight trajectory along +x starting at (5, 0); rig frame should put the
    # anchor at the origin and the motion along +rx.
    poses = [
        _Waypoint(_Pose(_Vec(5.0, 0.0), _Quat()), 0),
        _Waypoint(_Pose(_Vec(7.0, 0.0), _Quat()), 500_000),
    ]
    meta = _poses_to_metadata(poses)
    assert meta is not None
    rig = meta["trajectory_xy_rig_frame"]
    assert abs(rig[0]["rx"]) < 1e-9 and abs(rig[0]["ry"]) < 1e-9
    assert abs(rig[1]["rx"] - 2.0) < 1e-9 and abs(rig[1]["ry"]) < 1e-9
    assert meta["trajectory_poses"][1]["x"] == 7.0


def test_poses_to_metadata_needs_two_points():
    assert _poses_to_metadata([]) is None
    assert _poses_to_metadata([_Waypoint(_Pose(_Vec(0.0, 0.0), _Quat()), 0)]) is None
