# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Per-step online consistency monitor over a closed-loop ``rollout.asl``.

For each driver decision step recorded in the ASL log this module recovers the
chain-of-thought (CoT) text and the planned trajectory the policy returned,
derives the trajectory meta-actions, and runs the deterministic rule-based
consistency monitor (:func:`alpasim_utils.consistency.match_cot_to_trajectory`,
the same exact-label monitor used for the static benchmark). It returns one
:class:`MonitorStep` per decision step.

The extraction of ``(CoT, trajectory_poses)`` from the ASL mirrors
``alpasim_utils.extract_frame`` -- it reads the ``driver_request`` /
``driver_return`` protobuf stream -- but yields *every* decision step rather
than a single target frame, which is exactly what the online-monitoring
analysis needs.
"""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from alpasim_utils.consistency import match_cot_to_trajectory
from alpasim_utils.logs import async_read_pb_log
from alpasim_utils.trajectory_additional_info import build_additional_info

#: The rule monitor abstains (no intent expressed on either axis) on this label,
#: so it is *not* counted as a safety flag.
_ABSTAIN_LABEL = "invalid_parse"
#: The label the rule monitor scores 0 on while still expressing a (mis)intent.
_FLAG_LABELS = ("inconsistent",)


@dataclass
class MonitorStep:
    """Online monitor verdict for a single closed-loop decision step."""

    now_us: int  # simulation time at which the policy was queried
    n_poses: int  # waypoints in the planned trajectory the policy returned
    has_cot: bool  # whether a CoT string was logged for this step
    scored: bool  # whether the monitor produced a usable verdict
    score: float | None  # rule monitor binary score (1 consistent, 0 otherwise)
    label: str | None  # consistent | inconsistent | invalid_parse
    longitudinal_verdict: str | None
    lateral_verdict: str | None
    flag: bool  # inconsistency flag (label == inconsistent)

    def to_dict(self) -> dict:
        return asdict(self)


def _flatten_cot(value: Any) -> str | None:
    """Mirror the CoT flattening used by the offline tooling."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(part for item in value if (part := _flatten_cot(item)))
    return str(value)


def _poses_to_metadata(poses: list) -> dict | None:
    """Build a metadata-like dict (rig + world trajectory) from ASL poses.

    Reproduces the rig-frame projection used by ``extract_frame``: the first
    pose is the anchor (origin, heading aligned to +x), later poses are rotated
    into that ego-rig frame. ``build_additional_info`` prefers the rig frame for
    meta-action derivation, with world poses kept for completeness.
    """
    if len(poses) < 2:
        return None
    anchor = poses[0]
    ax, ay = anchor.pose.vec.x, anchor.pose.vec.y
    anchor_yaw = np.arctan2(
        2.0
        * (
            anchor.pose.quat.w * anchor.pose.quat.z
            + anchor.pose.quat.x * anchor.pose.quat.y
        ),
        1.0 - 2.0 * (anchor.pose.quat.y**2 + anchor.pose.quat.z**2),
    )
    cos_yaw, sin_yaw = np.cos(-anchor_yaw), np.sin(-anchor_yaw)
    rig: list[dict] = []
    world: list[dict] = []
    for pt in poses:
        dx = pt.pose.vec.x - ax
        dy = pt.pose.vec.y - ay
        rig.append(
            {
                "timestamp_us": pt.timestamp_us,
                "rx": cos_yaw * dx - sin_yaw * dy,
                "ry": sin_yaw * dx + cos_yaw * dy,
            }
        )
        world.append(
            {
                "timestamp_us": pt.timestamp_us,
                "x": pt.pose.vec.x,
                "y": pt.pose.vec.y,
                "z": pt.pose.vec.z,
            }
        )
    return {"trajectory_xy_rig_frame": rig, "trajectory_poses": world}


def _cot_from_response(driver_return: Any) -> str | None:
    """Unpickle the driver debug ``extra`` and return its reasoning text."""
    dbg = driver_return.debug_info.unstructured_debug_info
    if not dbg:
        return None
    try:
        extra = pickle.loads(dbg)
    except Exception:
        return None
    if isinstance(extra, dict):
        return _flatten_cot(extra.get("reasoning_text"))
    return None


def _score_step(now_us: int, cot_text: str | None, poses: list) -> MonitorStep:
    """Run the rule monitor on one decision step's CoT and planned trajectory."""
    n_poses = len(poses)
    has_cot = bool(cot_text)
    meta = _poses_to_metadata(poses)
    if meta is None or not has_cot:
        return MonitorStep(
            now_us=now_us,
            n_poses=n_poses,
            has_cot=has_cot,
            scored=False,
            score=None,
            label=None,
            longitudinal_verdict=None,
            lateral_verdict=None,
            flag=False,
        )
    meta["chain_of_thought"] = cot_text
    try:
        additional_info = build_additional_info(meta, metadata_path="<asl>")
        meta_actions = additional_info.get("meta_actions") or {}
        report = match_cot_to_trajectory(cot_text or "", meta_actions)
    except Exception:
        return MonitorStep(
            now_us=now_us,
            n_poses=n_poses,
            has_cot=has_cot,
            scored=False,
            score=None,
            label=None,
            longitudinal_verdict=None,
            lateral_verdict=None,
            flag=False,
        )
    label = report.label
    return MonitorStep(
        now_us=now_us,
        n_poses=n_poses,
        has_cot=True,
        scored=True,
        score=float(report.score),
        label=label,
        longitudinal_verdict=report.longitudinal.verdict,
        lateral_verdict=report.lateral.verdict,
        flag=label in _FLAG_LABELS,
    )


async def _monitor_steps_async(asl_path: str | Path) -> list[MonitorStep]:
    pending_now_us: int | None = None
    steps: list[MonitorStep] = []
    async for message in async_read_pb_log(str(asl_path)):
        msg_type = message.WhichOneof("log_entry")
        if msg_type == "driver_request":
            pending_now_us = message.driver_request.time_now_us
        elif msg_type == "driver_return":
            if pending_now_us is None:
                continue
            poses = list(message.driver_return.trajectory.poses)
            cot = _cot_from_response(message.driver_return)
            steps.append(_score_step(pending_now_us, cot, poses))
            pending_now_us = None
    return steps


def monitor_steps(asl_path: str | Path) -> list[MonitorStep]:
    """Score every decision step in a ``rollout.asl`` with the rule monitor.

    Matching is exact-label (the configuration evaluated in the paper); see
    :func:`alpasim_utils.consistency.match_cot_to_trajectory`.

    Args:
        asl_path: Path to the rollout's ``rollout.asl`` protobuf log.

    Returns:
        One :class:`MonitorStep` per ``driver_return`` in log order.
    """
    return asyncio.run(_monitor_steps_async(asl_path))
