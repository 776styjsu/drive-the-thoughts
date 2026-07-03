# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Rule-based CoT/trajectory consistency reward.

This parses generated CoT text into intended ego meta-actions, compares those
intents against the trajectory-derived meta-action sequence on both
longitudinal and lateral axes, and assigns ``score = 1`` only when both axes are
consistent. Matching is exact: a CoT label is satisfied only when the trajectory
contains that same label in order. Magnitude-sensitive labels such as
``gentle_decelerate`` and ``strong_decelerate`` are distinct.

Invalid parses, unknown trajectory labels, and exact-label mismatches score
``0``. A silent axis imposes no requirement, but a CoT with no parsed intent on
either axis is labelled ``invalid_parse``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from .cot_meta_actions import CotParseResult, parse_cot
from .meta_actions_types import LateralAction, LongitudinalAction

ActionT = TypeVar("ActionT", LongitudinalAction, LateralAction)


@dataclass
class MatchedAction:
    cot_label: str
    trajectory_label: str
    trajectory_index: int

    def to_dict(self) -> dict:
        return {
            "cot_label": self.cot_label,
            "trajectory_label": self.trajectory_label,
            "trajectory_index": self.trajectory_index,
        }


@dataclass
class AxisVerdict:
    axis: str
    cot_label: str
    trajectory_label: str
    cot_family: str | None
    trajectory_family: str | None
    verdict: str
    reward: float
    cot_sequence: list[str] = field(default_factory=list)
    trajectory_sequence: list[str] = field(default_factory=list)
    matched_actions: list[MatchedAction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "cot_label": self.cot_label,
            "trajectory_label": self.trajectory_label,
            "cot_family": self.cot_family,
            "trajectory_family": self.trajectory_family,
            "verdict": self.verdict,
            "reward": self.reward,
            "cot_sequence": self.cot_sequence,
            "trajectory_sequence": self.trajectory_sequence,
            "matched_actions": [m.to_dict() for m in self.matched_actions],
        }


@dataclass
class ConsistencyReport:
    longitudinal: AxisVerdict
    lateral: AxisVerdict
    score: float
    label: str
    cot_parse: CotParseResult

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "longitudinal": self.longitudinal.to_dict(),
            "lateral": self.lateral.to_dict(),
            "cot_parse": self.cot_parse.to_dict(),
        }


@dataclass(frozen=True)
class _AxisConfig:
    """Static, per-axis vocabulary for the consistency check.

    Bundling the axis-specific knobs lets the longitudinal and lateral passes
    share one ``_axis_verdict`` call site instead of repeating these argument
    lists everywhere a verdict is computed.
    """

    axis: str


_LONGITUDINAL_AXIS = _AxisConfig(axis="longitudinal")
_LATERAL_AXIS = _AxisConfig(axis="lateral")


def _finalize_report(
    cot_parse: CotParseResult,
    longitudinal_axis: AxisVerdict,
    lateral_axis: AxisVerdict,
) -> ConsistencyReport:
    """Merge two axis verdicts into the paper-style binary score and label.

    This is the single source of truth for the score/label collapse shared by
    ``match_cot_to_trajectory`` and ``match_actions``.
    """

    parsed_any_intent = bool(
        cot_parse.longitudinal_sequence or cot_parse.lateral_sequence
    )
    score = 1.0 if (
        parsed_any_intent
        and longitudinal_axis.reward == 1.0
        and lateral_axis.reward == 1.0
    ) else 0.0

    if not parsed_any_intent:
        label = "invalid_parse"
    elif score == 1.0:
        label = "consistent"
    else:
        label = "inconsistent"

    return ConsistencyReport(
        longitudinal=longitudinal_axis,
        lateral=lateral_axis,
        score=score,
        label=label,
        cot_parse=cot_parse,
    )


def match_cot_to_trajectory(
    cot_text: str,
    trajectory_meta_actions: dict[str, Any],
    *,
    reward_table: dict[str, float] | None = None,
) -> ConsistencyReport:
    """Parse CoT text and compute the paper-style binary consistency reward.

    ``reward_table`` is accepted for backward API compatibility and ignored:
    Alpamayo-R1 defines this reward as binary.
    """

    _ = reward_table
    cot_parse = parse_cot(cot_text)
    trajectory_longitudinal, trajectory_lateral = _trajectory_sequences(
        trajectory_meta_actions or {}
    )

    longitudinal_axis = _axis_verdict(
        _LONGITUDINAL_AXIS, cot_parse.longitudinal_sequence, trajectory_longitudinal
    )
    lateral_axis = _axis_verdict(
        _LATERAL_AXIS, cot_parse.lateral_sequence, trajectory_lateral
    )
    return _finalize_report(cot_parse, longitudinal_axis, lateral_axis)


def match_actions(
    cot_longitudinal: LongitudinalAction,
    cot_lateral: LateralAction,
    trajectory_longitudinal: LongitudinalAction,
    trajectory_lateral: LateralAction,
    *,
    reward_table: dict[str, float] | None = None,
    cot_parse: CotParseResult | None = None,
) -> ConsistencyReport:
    """Build a binary consistency report from single-step labels.

    Kept for callers that do not have a full trajectory sequence. New code
    should prefer ``match_cot_to_trajectory`` with ``meta_actions["per_segment"]``.
    """

    _ = reward_table
    parse = cot_parse or CotParseResult(
        longitudinal=cot_longitudinal,
        lateral=cot_lateral,
        longitudinal_sequence=(
            [] if cot_longitudinal is LongitudinalAction.UNKNOWN else [cot_longitudinal]
        ),
        lateral_sequence=[] if cot_lateral is LateralAction.UNKNOWN else [cot_lateral],
    )

    longitudinal_axis = _axis_verdict(
        _LONGITUDINAL_AXIS, parse.longitudinal_sequence, [trajectory_longitudinal]
    )
    lateral_axis = _axis_verdict(
        _LATERAL_AXIS, parse.lateral_sequence, [trajectory_lateral]
    )
    return _finalize_report(parse, longitudinal_axis, lateral_axis)


def _trajectory_sequences(
    trajectory_meta_actions: dict[str, Any],
) -> tuple[list[LongitudinalAction], list[LateralAction]]:
    per_segment = trajectory_meta_actions.get("per_segment") or []
    longitudinal: list[LongitudinalAction] = []
    lateral: list[LateralAction] = []

    if isinstance(per_segment, list) and per_segment:
        for segment in per_segment:
            if not isinstance(segment, dict):
                continue
            longitudinal.append(_safe_longitudinal(segment.get("longitudinal")))
            lateral.append(_safe_lateral(segment.get("lateral")))
    else:
        dominant = trajectory_meta_actions.get("dominant") or {}
        longitudinal.append(_safe_longitudinal(dominant.get("longitudinal")))
        lateral.append(_safe_lateral(dominant.get("lateral")))

    return _compress_sequence(longitudinal), _compress_sequence(lateral)


def _axis_verdict(
    config: _AxisConfig,
    cot_sequence: list[ActionT],
    trajectory_sequence: list[ActionT],
) -> AxisVerdict:
    axis = config.axis

    primary_cot = cot_sequence[0] if cot_sequence else None
    primary_traj = trajectory_sequence[0] if trajectory_sequence else None

    # No stated intent on this axis => no requirement: the CoT makes no claim
    # here, so the trajectory cannot contradict it and nothing needs to match
    # (e.g. a CoT that states only a turn need not also narrate the
    # deceleration the turn physically requires). When BOTH axes are silent the
    # clip is still labelled invalid_parse via the parsed_any_intent gate in
    # match_cot_to_trajectory.
    if not cot_sequence:
        return _build_axis_verdict(
            axis,
            primary_cot,
            primary_traj,
            cot_sequence,
            trajectory_sequence,
            "no_intent",
            1.0,
            [],
        )

    if not trajectory_sequence or any(a.value == "unknown" for a in trajectory_sequence):
        return _build_axis_verdict(
            axis,
            primary_cot,
            primary_traj,
            cot_sequence,
            trajectory_sequence,
            "trajectory_unparsed",
            0.0,
            [],
        )

    matched = _match_sequence(cot_sequence, trajectory_sequence)
    # Exact matching tolerates trajectory extras: the CoT action sequence only
    # has to appear in order within the trajectory action sequence.
    reward = 1.0 if matched is not None else 0.0
    return _build_axis_verdict(
        axis,
        primary_cot,
        primary_traj,
        cot_sequence,
        trajectory_sequence,
        "consistent" if reward == 1.0 else "mismatch",
        reward,
        matched or [],
    )


def _build_axis_verdict(
    axis: str,
    primary_cot,
    primary_traj,
    cot_sequence,
    trajectory_sequence,
    verdict: str,
    reward: float,
    matched: list[MatchedAction],
) -> AxisVerdict:
    return AxisVerdict(
        axis=axis,
        cot_label=primary_cot.value if primary_cot is not None else "unknown",
        trajectory_label=primary_traj.value if primary_traj is not None else "unknown",
        cot_family=None,
        trajectory_family=None,
        verdict=verdict,
        reward=reward,
        cot_sequence=[a.value for a in cot_sequence],
        trajectory_sequence=[a.value for a in trajectory_sequence],
        matched_actions=matched,
    )


def _match_sequence(
    cot_sequence: list[ActionT],
    trajectory_sequence: list[ActionT],
) -> list[MatchedAction] | None:
    matches: list[MatchedAction] = []
    search_start = 0
    for cot_action in cot_sequence:
        matched_index = None
        for index in range(search_start, len(trajectory_sequence)):
            trajectory_action = trajectory_sequence[index]
            if _actions_compatible(cot_action, trajectory_action):
                matched_index = index
                break
        if matched_index is None:
            return None
        trajectory_action = trajectory_sequence[matched_index]
        matches.append(
            MatchedAction(
                cot_label=cot_action.value,
                trajectory_label=trajectory_action.value,
                trajectory_index=matched_index,
            )
        )
        search_start = matched_index + 1
    return matches


def _actions_compatible(
    cot_action: ActionT,
    trajectory_action: ActionT,
) -> bool:
    return cot_action is trajectory_action


def _compress_sequence(sequence: list[ActionT]) -> list[ActionT]:
    compressed = []
    for action in sequence:
        if compressed and compressed[-1] is action:
            continue
        compressed.append(action)
    return compressed


def _safe_longitudinal(value: Any) -> LongitudinalAction:
    if value is None:
        return LongitudinalAction.UNKNOWN
    try:
        return LongitudinalAction(value)
    except ValueError:
        return LongitudinalAction.UNKNOWN


def _safe_lateral(value: Any) -> LateralAction:
    if value is None:
        return LateralAction.UNKNOWN
    try:
        return LateralAction(value)
    except ValueError:
        return LateralAction.UNKNOWN
