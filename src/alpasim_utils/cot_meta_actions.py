# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Parse driving chain-of-thought reasoning text into Alpamayo-R1 meta-actions.

The parser is deterministic: keyword + modifier matching with simple negation
handling. It maps a free-text CoT (e.g. "Decelerate because the lead vehicle
is braking") onto the same ``LongitudinalAction`` and ``LateralAction``
vocabulary that the trajectory parser emits, so the two can be compared by
``consistency.match_cot_to_trajectory``.

Design notes:
  * Driving CoTs typically follow ``ACTION because/since/due to CAUSE``. We
    bias toward action verbs that appear before causal connectives.
  * Magnitude (gentle vs strong, steer vs sharp steer) is carried by the phrase
    pattern itself: phrases that denote a forceful manoeuvre map directly to a
    strong/sharp label (e.g. "brake hard" -> ``strong_decelerate``). There is no
    separate adjacent-adverb promotion step, so the matched phrase alone
    determines the emitted label. The consistency reward compares these emitted
    labels exactly, so magnitude-specific wording should map to the
    corresponding magnitude-specific label.
  * Multiple sentences are supported; ordered action sequences are retained,
    while the first action per axis is exposed as a compact summary.
  * No LLM dependency. For richer parsing, a thin LLM-backed parser can be
    added with the same return shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .meta_actions_types import LateralAction, LongitudinalAction


_NEGATION_TOKENS = frozenset(
    {"not", "no", "never", "without", "avoid", "don't", "dont", "doesn't", "doesnt"}
)
_CAUSAL_CONNECTIVES = (
    " because ",
    " since ",
    " as ",
    " due to ",
    " given ",
    " so that ",
)


# Phrase patterns for P_c -> (phrase, label, magnitude annotation).
# The label already encodes magnitude (e.g. STRONG_DECELERATE vs
# GENTLE_DECELERATE); the third element is a retained provenance annotation
# ("strong" for intrinsically forceful phrases, None otherwise) and is not read
# by the parser. Order matters: longest/most-specific phrases first within each
# list so they pre-empt shorter overlaps (e.g. "hard brake" before "brake").
#
# Provenance summary for the inactive source distribution used to design the
# compact lexicon below. These counts are documentation only; parse_cot() uses
# only _LONGITUDINAL_PATTERNS and _LATERAL_PATTERNS.
_SOURCE_PATTERN_PROVENANCE = (
    ("mined recurring development-set surface forms", 100),
    ("close paraphrases of mined forms", 50),
    ("additional rubric-guided paraphrases", 50),
)


# Compact runtime lexicon.  Prefer action-bearing phrases that generalize
# across causes and scenes: "keep lane" should match whether the explanation
# mentions cones, traffic, a clear road, or something else.  Specific
# magnitude phrases must precede their generic forms because _scan_patterns()
# lets an earlier, longer phrase claim overlapping text.
_LONGITUDINAL_PATTERNS: list[tuple[str, LongitudinalAction, str | None]] = [
    # Stop intent and stop preparation.
    # This compact compound prevents the subordinate following-distance goal
    # from being emitted as a second maintain-speed action.
    ("stop to keep distance", LongitudinalAction.STOP, None),
    ("stop to yield", LongitudinalAction.STOP, None),
    ("prepare to stop", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow to a stop", LongitudinalAction.STOP, None),
    ("come to a full stop", LongitudinalAction.STOP, None),
    ("come to a stop", LongitudinalAction.STOP, None),
    ("bring the vehicle to a halt", LongitudinalAction.STOP, None),
    ("bring to a stop", LongitudinalAction.STOP, None),
    ("full stop", LongitudinalAction.STOP, None),
    ("remain stopped", LongitudinalAction.STOP, None),
    ("stay stopped", LongitudinalAction.STOP, None),
    ("hold position", LongitudinalAction.STOP, None),
    ("stop", LongitudinalAction.STOP, None),
    ("halt", LongitudinalAction.STOP, None),
    # Reverse.
    ("reverse slowly", LongitudinalAction.REVERSE, None),
    ("back up", LongitudinalAction.REVERSE, None),
    ("back out", LongitudinalAction.REVERSE, None),
    ("reverse", LongitudinalAction.REVERSE, None),
    # Strong deceleration before generic braking phrases.
    ("perform an emergency stop", LongitudinalAction.STRONG_DECELERATE, "strong"),
    ("emergency brake", LongitudinalAction.STRONG_DECELERATE, "strong"),
    ("apply a firm brake", LongitudinalAction.STRONG_DECELERATE, "strong"),
    ("slam the brakes", LongitudinalAction.STRONG_DECELERATE, "strong"),
    ("brake heavily", LongitudinalAction.STRONG_DECELERATE, "strong"),
    ("brake hard", LongitudinalAction.STRONG_DECELERATE, "strong"),
    ("hard brake", LongitudinalAction.STRONG_DECELERATE, "strong"),
    ("decelerate hard", LongitudinalAction.STRONG_DECELERATE, "strong"),
    # Ordinary deceleration and yielding.
    ("slow down", LongitudinalAction.GENTLE_DECELERATE, None),
    ("reduce speed", LongitudinalAction.GENTLE_DECELERATE, None),
    ("reduce velocity", LongitudinalAction.GENTLE_DECELERATE, None),
    ("ease off", LongitudinalAction.GENTLE_DECELERATE, None),
    ("back off", LongitudinalAction.GENTLE_DECELERATE, None),
    ("bleed speed", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adjust speed", LongitudinalAction.GENTLE_DECELERATE, None),
    ("create a usable gap", LongitudinalAction.GENTLE_DECELERATE, None),
    ("create a gap", LongitudinalAction.GENTLE_DECELERATE, None),
    ("open a gap", LongitudinalAction.GENTLE_DECELERATE, None),
    ("make room", LongitudinalAction.GENTLE_DECELERATE, None),
    ("yield space", LongitudinalAction.GENTLE_DECELERATE, None),
    ("decelerate", LongitudinalAction.GENTLE_DECELERATE, None),
    ("brake", LongitudinalAction.GENTLE_DECELERATE, None),
    ("yield", LongitudinalAction.GENTLE_DECELERATE, None),
    # Strong acceleration before generic acceleration phrases.
    ("accelerate strongly", LongitudinalAction.STRONG_ACCELERATE, "strong"),
    ("accelerate briskly", LongitudinalAction.STRONG_ACCELERATE, "strong"),
    ("accelerate hard", LongitudinalAction.STRONG_ACCELERATE, "strong"),
    ("speed up quickly", LongitudinalAction.STRONG_ACCELERATE, "strong"),
    ("rapid acceleration", LongitudinalAction.STRONG_ACCELERATE, "strong"),
    ("floor it", LongitudinalAction.STRONG_ACCELERATE, "strong"),
    # Ordinary acceleration.
    ("pull away", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("pick up speed", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("increase speed", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("gain speed", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("resume speed", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("resume from stop", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("creep forward", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("inch forward", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("roll forward", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("speed up", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("accelerate", LongitudinalAction.GENTLE_ACCELERATE, None),
    # Maintain speed / following intent.
    ("maintain a safe following gap", LongitudinalAction.MAINTAIN_SPEED, None),
    ("maintain a safe distance", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep a safe distance", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep distance", LongitudinalAction.MAINTAIN_SPEED, None),
    ("match the flow of traffic", LongitudinalAction.MAINTAIN_SPEED, None),
    ("match speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("maintain steady speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("maintain speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("hold speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("steady pace", LongitudinalAction.MAINTAIN_SPEED, None),
    ("constant speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("traffic speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("wait for a gap", LongitudinalAction.MAINTAIN_SPEED, None),
    ("coast", LongitudinalAction.MAINTAIN_SPEED, None),
    ("cruise", LongitudinalAction.MAINTAIN_SPEED, None),
]


_LATERAL_PATTERNS: list[tuple[str, LateralAction, str | None]] = [
    # Reverse-with-direction must precede ordinary steering phrases.
    ("reverse left", LateralAction.REVERSE_LEFT, None),
    ("reverse right", LateralAction.REVERSE_RIGHT, None),
    ("back left", LateralAction.REVERSE_LEFT, None),
    ("back right", LateralAction.REVERSE_RIGHT, None),
    # Sharp steering before generic left/right steering.
    ("steer sharply left", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("steer sharply right", LateralAction.SHARP_STEER_RIGHT, "strong"),
    ("sharp steer left", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("sharp steer right", LateralAction.SHARP_STEER_RIGHT, "strong"),
    ("sharp left", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("sharp right", LateralAction.SHARP_STEER_RIGHT, "strong"),
    ("hard left", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("hard right", LateralAction.SHARP_STEER_RIGHT, "strong"),
    ("make a u-turn", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("u-turn", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("uturn", LateralAction.SHARP_STEER_LEFT, "strong"),
    # Left motion.  Variants differ only where English word order requires it.
    ("change one lane to the left", LateralAction.STEER_LEFT, None),
    ("change lanes to the left", LateralAction.STEER_LEFT, None),
    ("change lane to the left", LateralAction.STEER_LEFT, None),
    ("change to the left lane", LateralAction.STEER_LEFT, None),
    ("lane change to the left", LateralAction.STEER_LEFT, None),
    ("change lanes left", LateralAction.STEER_LEFT, None),
    ("lane change left", LateralAction.STEER_LEFT, None),
    ("move one lane left", LateralAction.STEER_LEFT, None),
    ("move left", LateralAction.STEER_LEFT, None),
    ("merge left", LateralAction.STEER_LEFT, None),
    ("nudge to the left", LateralAction.STEER_LEFT, None),
    ("nudge left", LateralAction.STEER_LEFT, None),
    ("shift left", LateralAction.STEER_LEFT, None),
    ("edge left", LateralAction.STEER_LEFT, None),
    ("drift left", LateralAction.STEER_LEFT, None),
    ("swerve left", LateralAction.STEER_LEFT, None),
    ("bear left", LateralAction.STEER_LEFT, None),
    ("veer left", LateralAction.STEER_LEFT, None),
    ("split to the left", LateralAction.STEER_LEFT, None),
    ("steer left", LateralAction.STEER_LEFT, None),
    ("turn left", LateralAction.STEER_LEFT, None),
    # Right motion.
    ("change one lane to the right", LateralAction.STEER_RIGHT, None),
    ("change lanes to the right", LateralAction.STEER_RIGHT, None),
    ("change lane to the right", LateralAction.STEER_RIGHT, None),
    ("change to the right lane", LateralAction.STEER_RIGHT, None),
    ("lane change to the right", LateralAction.STEER_RIGHT, None),
    ("change lanes right", LateralAction.STEER_RIGHT, None),
    ("lane change right", LateralAction.STEER_RIGHT, None),
    ("move one lane right", LateralAction.STEER_RIGHT, None),
    ("move right", LateralAction.STEER_RIGHT, None),
    ("merge right", LateralAction.STEER_RIGHT, None),
    ("nudge to the right", LateralAction.STEER_RIGHT, None),
    ("nudge right", LateralAction.STEER_RIGHT, None),
    ("shift right", LateralAction.STEER_RIGHT, None),
    ("edge right", LateralAction.STEER_RIGHT, None),
    ("drift right", LateralAction.STEER_RIGHT, None),
    ("swerve right", LateralAction.STEER_RIGHT, None),
    ("bear right", LateralAction.STEER_RIGHT, None),
    ("veer right", LateralAction.STEER_RIGHT, None),
    ("split to the right", LateralAction.STEER_RIGHT, None),
    ("pull over", LateralAction.STEER_RIGHT, None),
    ("steer right", LateralAction.STEER_RIGHT, None),
    ("turn right", LateralAction.STEER_RIGHT, None),
    # Lane keeping / straight travel.
    ("keep at the center of the lane", LateralAction.GO_STRAIGHT, None),
    ("keep the vehicle centered", LateralAction.GO_STRAIGHT, None),
    ("continue centered", LateralAction.GO_STRAIGHT, None),
    ("stay centered", LateralAction.GO_STRAIGHT, None),
    ("track the lane center", LateralAction.GO_STRAIGHT, None),
    ("hold lane position", LateralAction.GO_STRAIGHT, None),
    ("maintain lane", LateralAction.GO_STRAIGHT, None),
    ("keeping lane", LateralAction.GO_STRAIGHT, None),
    ("keep lane", LateralAction.GO_STRAIGHT, None),
    ("stay in the lane", LateralAction.GO_STRAIGHT, None),
    ("stay in lane", LateralAction.GO_STRAIGHT, None),
    ("hold the lane", LateralAction.GO_STRAIGHT, None),
    ("hold lane", LateralAction.GO_STRAIGHT, None),
    ("lane keeping", LateralAction.GO_STRAIGHT, None),
    ("lane keep", LateralAction.GO_STRAIGHT, None),
    ("continue straight", LateralAction.GO_STRAIGHT, None),
    ("go straight", LateralAction.GO_STRAIGHT, None),
    ("proceed straight", LateralAction.GO_STRAIGHT, None),
    ("straight ahead", LateralAction.GO_STRAIGHT, None),
    ("follow the lane", LateralAction.GO_STRAIGHT, None),
    ("follow the road", LateralAction.GO_STRAIGHT, None),
]



@dataclass(frozen=True)
class CotMatch:
    """One pattern match inside the CoT text."""

    phrase: str
    label: str
    start: int
    end: int
    negated: bool


@dataclass
class CotParseResult:
    longitudinal: LongitudinalAction = LongitudinalAction.UNKNOWN
    lateral: LateralAction = LateralAction.UNKNOWN
    longitudinal_sequence: list[LongitudinalAction] = field(default_factory=list)
    lateral_sequence: list[LateralAction] = field(default_factory=list)
    longitudinal_evidence: list[CotMatch] = field(default_factory=list)
    lateral_evidence: list[CotMatch] = field(default_factory=list)
    all_matches: list[CotMatch] = field(default_factory=list)
    text: str = ""
    parser: str = "keyword"

    def to_dict(self) -> dict:
        return {
            "parser": self.parser,
            "longitudinal": self.longitudinal.value,
            "lateral": self.lateral.value,
            "longitudinal_sequence": [a.value for a in self.longitudinal_sequence],
            "lateral_sequence": [a.value for a in self.lateral_sequence],
            "longitudinal_evidence": [
                _match_to_dict(m) for m in self.longitudinal_evidence
            ],
            "lateral_evidence": [_match_to_dict(m) for m in self.lateral_evidence],
            "all_matches": [_match_to_dict(m) for m in self.all_matches],
            "text": self.text,
        }


def _match_to_dict(m: CotMatch) -> dict:
    return {
        "phrase": m.phrase,
        "label": m.label,
        "start": m.start,
        "end": m.end,
        "negated": m.negated,
    }


def _normalize(text: str) -> str:
    text = text.lower()
    # Collapse whitespace & strip apostrophe-only contractions for negation matching.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_negation_before(text: str, start: int, window_chars: int = 24) -> bool:
    left = text[max(0, start - window_chars) : start]
    tokens = re.findall(r"[a-z']+", left)
    return any(tok in _NEGATION_TOKENS for tok in tokens)


def _scan_patterns(
    text: str,
    patterns: list[tuple[str, "LongitudinalAction | LateralAction", str | None]],
    axis: str,
) -> list[CotMatch]:
    # ``axis`` is retained for signature symmetry with the trajectory side; the
    # emitted label now comes verbatim from the matched phrase's pattern label,
    # so no per-axis magnitude post-processing is applied here.
    _ = axis
    seen_spans: list[tuple[int, int]] = []
    matches: list[CotMatch] = []
    for phrase, base_label, _explicit_magnitude in patterns:
        # Match as a whole phrase, anchored at word boundaries on the outer ends.
        pattern = r"\b" + re.escape(phrase) + r"\b"
        for found in re.finditer(pattern, text):
            start, end = found.span()
            # Skip if a longer (earlier-listed) pattern already covered this span.
            if any(s <= start and end <= e for s, e in seen_spans):
                continue
            seen_spans.append((start, end))
            matches.append(
                CotMatch(
                    phrase=phrase,
                    label=base_label.value,
                    start=start,
                    end=end,
                    negated=_has_negation_before(text, start),
                )
            )
    matches.sort(key=lambda m: m.start)
    return matches


def _select_primary(matches: list[CotMatch], text: str) -> CotMatch | None:
    """Pick the action match most likely to be the stated decision.

    Heuristic: prefer non-negated matches that occur before the first causal
    connective (ACTION because CAUSE pattern). Otherwise the first non-negated
    match. If all are negated, return None.
    """
    causal_pos = len(text)
    for connective in _CAUSAL_CONNECTIVES:
        idx = text.find(connective)
        if idx != -1 and idx < causal_pos:
            causal_pos = idx

    pre_causal = [m for m in matches if not m.negated and m.end <= causal_pos]
    if pre_causal:
        return pre_causal[0]
    not_negated = [m for m in matches if not m.negated]
    if not_negated:
        return not_negated[0]
    return None


def _select_action_matches(matches: list[CotMatch], text: str) -> list[CotMatch]:
    """Return ordered action-side matches, excluding causal-factor matches.

    CoC traces usually read as ``ACTION because CAUSE``. If at least one action
    appears before the first causal connective, later matches are treated as
    causal evidence rather than ego intent.
    """

    causal_pos = len(text)
    for connective in _CAUSAL_CONNECTIVES:
        idx = text.find(connective)
        if idx != -1 and idx < causal_pos:
            causal_pos = idx

    not_negated = [m for m in matches if not m.negated]
    pre_causal = [m for m in not_negated if m.end <= causal_pos]
    return pre_causal or not_negated


def _dedupe_consecutive_labels(
    matches: list[CotMatch],
    action_type: type[LongitudinalAction] | type[LateralAction],
) -> list[LongitudinalAction] | list[LateralAction]:
    sequence = []
    for match in matches:
        action = action_type(match.label)
        if sequence and sequence[-1] is action:
            continue
        sequence.append(action)
    return sequence


def parse_cot(text: str) -> CotParseResult:
    """Parse a chain-of-thought string into longitudinal/lateral meta-actions."""

    if not isinstance(text, str) or not text.strip():
        return CotParseResult(text=text or "")

    normalized = _normalize(text)

    longitudinal_matches = _scan_patterns(
        normalized, _LONGITUDINAL_PATTERNS, "longitudinal"
    )
    lateral_matches = _scan_patterns(normalized, _LATERAL_PATTERNS, "lateral")

    longitudinal_action_matches = _select_action_matches(
        longitudinal_matches, normalized
    )
    lateral_action_matches = _select_action_matches(lateral_matches, normalized)

    primary_long = _select_primary(longitudinal_action_matches, normalized)
    primary_lat = _select_primary(lateral_action_matches, normalized)

    longitudinal_sequence = _dedupe_consecutive_labels(
        longitudinal_action_matches,
        LongitudinalAction,
    )
    lateral_sequence = _dedupe_consecutive_labels(
        lateral_action_matches,
        LateralAction,
    )

    longitudinal = (
        LongitudinalAction(primary_long.label)
        if primary_long is not None
        else LongitudinalAction.UNKNOWN
    )
    lateral = (
        LateralAction(primary_lat.label)
        if primary_lat is not None
        else LateralAction.UNKNOWN
    )

    return CotParseResult(
        longitudinal=longitudinal,
        lateral=lateral,
        longitudinal_sequence=longitudinal_sequence,
        lateral_sequence=lateral_sequence,
        longitudinal_evidence=longitudinal_matches,
        lateral_evidence=lateral_matches,
        all_matches=sorted(
            longitudinal_matches + lateral_matches, key=lambda m: m.start
        ),
        text=normalized,
        parser="keyword",
    )
