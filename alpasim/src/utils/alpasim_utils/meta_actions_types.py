# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared meta-action vocabulary for CoT/trajectory consistency analysis.

Both the trajectory parser (``trajectory_additional_info``) and the CoT parser
(``cot_meta_actions``) emit labels from the enums defined here; the consistency
matcher (``consistency``) compares those labels exactly. Keeping the enums in
one tiny module avoids circular imports and gives one place to extend the
vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LongitudinalAction(str, Enum):
    STOP = "stop"
    REVERSE = "reverse"
    MAINTAIN_SPEED = "maintain_speed"
    GENTLE_ACCELERATE = "gentle_accelerate"
    STRONG_ACCELERATE = "strong_accelerate"
    GENTLE_DECELERATE = "gentle_decelerate"
    STRONG_DECELERATE = "strong_decelerate"
    UNKNOWN = "unknown"


class LateralAction(str, Enum):
    GO_STRAIGHT = "go_straight"
    STEER_LEFT = "steer_left"
    STEER_RIGHT = "steer_right"
    SHARP_STEER_LEFT = "sharp_steer_left"
    SHARP_STEER_RIGHT = "sharp_steer_right"
    REVERSE_LEFT = "reverse_left"
    REVERSE_RIGHT = "reverse_right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MetaActionThresholds:
    """Cutoffs used by the trajectory-side classifier.

    The Alpamayo-R1 paper's Table 5 names these categories but does not publish
    exact numeric cutoffs; defaults below are AV-literature reasonable.
    """

    stop_speed_mps: float = 0.2
    reverse_speed_mps: float = 0.2
    maintain_accel_mps2: float = 0.3
    strong_accel_mps2: float = 1.5
    straight_curvature_1pm: float = 0.005
    sharp_curvature_1pm: float = 0.05
