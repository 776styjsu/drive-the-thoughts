# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Rule-based CoT/trajectory consistency reward.

This implements the Alpamayo-R1 paper's binary CoC-action consistency reward:
parse the generated CoT into intended ego meta-actions, compare those intents
against the trajectory-derived meta-action sequence on both longitudinal and
lateral axes, and assign ``score = 1`` only when both axes are consistent.
Invalid parses, unknown trajectory labels, mismatches, and contradictions score
``0``.

Two matching modes are supported via ``match_mode``:

* ``"family"`` (default): the Alpamayo-R1 behaviour. Labels are grouped into
  direction families (e.g. ``gentle_decelerate``/``strong_decelerate`` -> the
  ``decel`` family); a CoT label matches any trajectory label in the same
  family, opposite families contradict, and neutral/supportive trajectory
  extras are tolerated.
* ``"exact"``: ignore families entirely and match by exact action label. A CoT
  label is consistent only when the trajectory contains that exact label (e.g.
  CoT ``gentle_decelerate`` is satisfied by trajectory ``gentle_decelerate`` but
  not by ``strong_decelerate``); anything else is ``inconsistent`` (there is no
  ``contradictory`` label in this mode). The ``invalid_parse`` rule (both axes
  silent) and the single-silent-channel no-requirement rule are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .cot_meta_actions import CotParseResult, parse_cot
from .meta_actions_types import (
    LATERAL_CONTRADICTIONS,
    LONGITUDINAL_CONTRADICTIONS,
    LateralAction,
    LongitudinalAction,
    lateral_family,
    longitudinal_family,
)

ActionT = TypeVar("ActionT", LongitudinalAction, LateralAction)

#: Supported values for the ``match_mode`` argument of the public matchers.
MATCH_MODES = ("family", "exact")


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
    family_fn: Callable[[Any], str | None]
    contradiction_pairs: frozenset[frozenset[str]]
    neutral_actions: frozenset[Any]
    supportive_actions: dict[Any, frozenset[Any]]
    #: When True, match by exact action label and ignore the family groupings
    #: above: only identical labels are compatible, opposite families never
    #: contradict (a non-match is a plain mismatch), and any trajectory extras
    #: are tolerated (the CoT label just has to appear in the trajectory).
    exact: bool = False


_LONGITUDINAL_AXIS = _AxisConfig(
    axis="longitudinal",
    family_fn=longitudinal_family,
    contradiction_pairs=LONGITUDINAL_CONTRADICTIONS,
    neutral_actions=frozenset({LongitudinalAction.MAINTAIN_SPEED}),
    supportive_actions={
        LongitudinalAction.STOP: frozenset(
            {
                LongitudinalAction.GENTLE_DECELERATE,
                LongitudinalAction.STRONG_DECELERATE,
            }
        ),
    },
)

_LATERAL_AXIS = _AxisConfig(
    axis="lateral",
    family_fn=lateral_family,
    contradiction_pairs=LATERAL_CONTRADICTIONS,
    neutral_actions=frozenset({LateralAction.GO_STRAIGHT}),
    supportive_actions={},
)

# Exact-match counterparts: no families, no contradictions, no neutral/
# supportive tolerance beyond "the trajectory contains the stated label".
_LONGITUDINAL_AXIS_EXACT = _AxisConfig(
    axis="longitudinal",
    family_fn=longitudinal_family,
    contradiction_pairs=frozenset(),
    neutral_actions=frozenset(),
    supportive_actions={},
    exact=True,
)

_LATERAL_AXIS_EXACT = _AxisConfig(
    axis="lateral",
    family_fn=lateral_family,
    contradiction_pairs=frozenset(),
    neutral_actions=frozenset(),
    supportive_actions={},
    exact=True,
)


def _axis_configs(match_mode: str) -> tuple[_AxisConfig, _AxisConfig]:
    """Return the (longitudinal, lateral) axis configs for a match mode."""
    if match_mode == "family":
        return _LONGITUDINAL_AXIS, _LATERAL_AXIS
    if match_mode == "exact":
        return _LONGITUDINAL_AXIS_EXACT, _LATERAL_AXIS_EXACT
    raise ValueError(
        f"Unknown match_mode {match_mode!r}; expected one of {MATCH_MODES}"
    )


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
    elif "contradict" in {longitudinal_axis.verdict, lateral_axis.verdict}:
        label = "contradictory"
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
    match_mode: str = "family",
) -> ConsistencyReport:
    """Parse CoT text and compute the paper-style binary consistency reward.

    ``reward_table`` is accepted for backward API compatibility and ignored:
    Alpamayo-R1 defines this reward as binary.

    ``match_mode`` selects how a CoT label matches a trajectory label: the
    family-based Alpamayo-R1 default (``"family"``) or exact label matching
    (``"exact"``); see the module docstring.
    """

    _ = reward_table
    longitudinal_config, lateral_config = _axis_configs(match_mode)
    cot_parse = parse_cot(cot_text)
    trajectory_longitudinal, trajectory_lateral = _trajectory_sequences(
        trajectory_meta_actions or {}
    )

    longitudinal_axis = _axis_verdict(
        longitudinal_config, cot_parse.longitudinal_sequence, trajectory_longitudinal
    )
    lateral_axis = _axis_verdict(
        lateral_config, cot_parse.lateral_sequence, trajectory_lateral
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
    match_mode: str = "family",
) -> ConsistencyReport:
    """Build a binary consistency report from single-step labels.

    Kept for callers that do not have a full trajectory sequence. New code
    should prefer ``match_cot_to_trajectory`` with ``meta_actions["per_segment"]``.
    ``match_mode`` behaves as in ``match_cot_to_trajectory``.
    """

    _ = reward_table
    longitudinal_config, lateral_config = _axis_configs(match_mode)
    parse = cot_parse or CotParseResult(
        longitudinal=cot_longitudinal,
        lateral=cot_lateral,
        longitudinal_sequence=(
            [] if cot_longitudinal is LongitudinalAction.UNKNOWN else [cot_longitudinal]
        ),
        lateral_sequence=[] if cot_lateral is LateralAction.UNKNOWN else [cot_lateral],
    )

    longitudinal_axis = _axis_verdict(
        longitudinal_config, parse.longitudinal_sequence, [trajectory_longitudinal]
    )
    lateral_axis = _axis_verdict(
        lateral_config, parse.lateral_sequence, [trajectory_lateral]
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
    # match_cot_to_trajectory. This single-silent-channel rule holds in both
    # family and exact modes.
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
            config,
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
            config,
        )

    # Exact mode has no notion of families, so opposite-direction labels never
    # "contradict" -- they are just a plain mismatch handled below.
    if not config.exact and _has_contradiction(
        cot_sequence,
        trajectory_sequence,
        config.family_fn,
        config.contradiction_pairs,
        config.neutral_actions,
        config.supportive_actions,
    ):
        return _build_axis_verdict(
            axis,
            primary_cot,
            primary_traj,
            cot_sequence,
            trajectory_sequence,
            "contradict",
            0.0,
            [],
            config,
        )

    matched = _match_sequence(
        cot_sequence,
        trajectory_sequence,
        config,
    )
    # Exact mode tolerates any trajectory extras: the CoT label only has to
    # appear somewhere in the trajectory ("traj has gentle_decelerate").
    extras_ok = config.exact or _trajectory_extras_are_allowed(
        cot_sequence,
        trajectory_sequence,
        config.family_fn,
        config.neutral_actions,
        config.supportive_actions,
    )
    reward = 1.0 if matched is not None and extras_ok else 0.0
    return _build_axis_verdict(
        axis,
        primary_cot,
        primary_traj,
        cot_sequence,
        trajectory_sequence,
        "consistent" if reward == 1.0 else "mismatch",
        reward,
        matched or [],
        config,
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
    config: _AxisConfig,
) -> AxisVerdict:
    # Family annotations are meaningless in exact mode (matching ignores them),
    # so report them as None to avoid implying a family-based verdict.
    family_fn = (lambda _action: None) if config.exact else config.family_fn
    return AxisVerdict(
        axis=axis,
        cot_label=primary_cot.value if primary_cot is not None else "unknown",
        trajectory_label=primary_traj.value if primary_traj is not None else "unknown",
        cot_family=family_fn(primary_cot) if primary_cot is not None else None,
        trajectory_family=family_fn(primary_traj) if primary_traj is not None else None,
        verdict=verdict,
        reward=reward,
        cot_sequence=[a.value for a in cot_sequence],
        trajectory_sequence=[a.value for a in trajectory_sequence],
        matched_actions=matched,
    )


def _match_sequence(
    cot_sequence: list[ActionT],
    trajectory_sequence: list[ActionT],
    config: _AxisConfig,
) -> list[MatchedAction] | None:
    matches: list[MatchedAction] = []
    search_start = 0
    for cot_action in cot_sequence:
        matched_index = None
        for index in range(search_start, len(trajectory_sequence)):
            trajectory_action = trajectory_sequence[index]
            if _actions_compatible(
                cot_action,
                trajectory_action,
                config,
            ):
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


def _trajectory_extras_are_allowed(
    cot_sequence: list[ActionT],
    trajectory_sequence: list[ActionT],
    family_fn,
    neutral_actions: set[ActionT],
    supportive_actions: dict[ActionT, set[ActionT]],
) -> bool:
    allowed_families = {
        family
        for action in cot_sequence
        if (family := family_fn(action)) is not None
    }
    allowed_actions = set(neutral_actions)
    for action in cot_sequence:
        allowed_actions.add(action)
        allowed_actions.update(supportive_actions.get(action, set()))

    for trajectory_action in trajectory_sequence:
        if trajectory_action in allowed_actions:
            continue
        family = family_fn(trajectory_action)
        if family is not None and family in allowed_families:
            continue
        return False
    return True


def _actions_compatible(
    cot_action: ActionT,
    trajectory_action: ActionT,
    config: _AxisConfig,
) -> bool:
    if cot_action is trajectory_action:
        return True
    # Exact mode: only the identical label matches; gentle_* and strong_* are
    # distinct even though they share a family.
    if config.exact:
        return False
    cot_family = config.family_fn(cot_action)
    trajectory_family = config.family_fn(trajectory_action)
    return cot_family is not None and cot_family == trajectory_family


def _has_contradiction(
    cot_sequence: list[ActionT],
    trajectory_sequence: list[ActionT],
    family_fn,
    contradiction_pairs: frozenset[frozenset[str]],
    neutral_actions: set[ActionT],
    supportive_actions: dict[ActionT, set[ActionT]],
) -> bool:
    cot_families = {
        family
        for action in cot_sequence
        if (family := family_fn(action)) is not None
    }
    allowed_families = set(cot_families)
    for action in neutral_actions:
        if (family := family_fn(action)) is not None:
            allowed_families.add(family)
    for action in cot_sequence:
        for supportive_action in supportive_actions.get(action, set()):
            if (family := family_fn(supportive_action)) is not None:
                allowed_families.add(family)

    trajectory_families = set()
    for action in trajectory_sequence:
        family = family_fn(action)
        if family is not None and family not in allowed_families:
            trajectory_families.add(family)

    for cot_family in cot_families:
        for trajectory_family in trajectory_families:
            if frozenset({cot_family, trajectory_family}) in contradiction_pairs:
                return True
    return False


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
