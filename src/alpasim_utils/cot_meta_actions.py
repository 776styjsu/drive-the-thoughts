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
    determines the emitted label. This keeps the parser aligned with the paper's
    coarse rule-based matching, where the consistency reward is magnitude-
    agnostic (gentle and strong share a direction family).
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
# Construction protocol:
#   Stage 1: 100 recurring surface forms mined from the development split
#     tutorial_alpamayo_upstream_2601/scene_1_clipgt-e66ebf31-... through
#     tutorial_alpamayo_upstream_2601/scene_30_clipgt-ee6908a0-....
#   Stage 2: 50 close paraphrase variants for those mined forms.
#   Stage 3: 50 additional paraphrases guided by, but distinct from, the
#     existing example phrase-action pairs and rubric examples.
#
# This mined/verbose set is intentionally commented out as a multiline
# provenance block.  It is not imported, allocated as hundreds of tuples, or
# consulted by parse_cot().  The compact runtime lexicon follows the block.
_VERBOSE_PATTERN_PROVENANCE = r"""
_P_C_CANDIDATE_LONGITUDINAL_PATTERNS: list[
    tuple[str, LongitudinalAction, str | None]
] = [
    # --- Stage 1: mined recurring surface forms (50 longitudinal) ---
    (
        "stop to keep distance to the stopped lead vehicle ahead",
        LongitudinalAction.STOP,
        None,
    ),
    ("stop to keep distance to the lead vehicle", LongitudinalAction.STOP, None),
    ("stop for the red right-turn traffic light", LongitudinalAction.STOP, None),
    ("stop for the red traffic light ahead", LongitudinalAction.STOP, None),
    ("stop for the red traffic light", LongitudinalAction.STOP, None),
    ("stop at the stop line", LongitudinalAction.STOP, None),
    ("stop due to the red traffic light", LongitudinalAction.STOP, None),
    ("slow down to stop for the red traffic light", LongitudinalAction.STOP, None),
    ("slow to a stop", LongitudinalAction.STOP, None),
    ("prepare to stop", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "yield to the cut-in vehicle from the right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "yield to the cut-in vehicle from the left",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("yield to the cut-in vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "slow down to yield to the cut-in vehicle from the left",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "decelerate to maintain a safe distance from the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "decelerate to maintain a safe distance to the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "decelerate to create a usable gap for a lane change to the right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "decelerate to create a gap for merging right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed to maintain a safe distance from the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed to maintain a safe distance",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the narrowed work-zone lane marked by cones",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the narrowed work-zone lane",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the narrowed lane marked by construction cones ahead",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the narrowed lane marked by construction cones",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("adapt speed for the narrowed lane", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "adapt speed for the raised crosswalk",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the school crossing ahead",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the school crossing zone",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("adapt speed for the school crossing", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the school zone", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the speed bumps", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the speed bump", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the speed hump", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the off-ramp", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the right curve", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the left curve", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "adjust speed for the left curve ahead",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to maintain a safe distance to the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to keep distance to the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down for the red traffic light ahead",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("slow down for the red traffic light", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "slow down to create a gap for merging right into the exit lane",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for merging into the right lane",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for vehicles merging from the right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for a merge into the left lane",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for a right-lane merge",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for merging right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for merging left",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("slow down and maintain lane", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow down", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "accelerate to match left-lane traffic and create a usable gap for a left lane change",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "accelerate to proceed through the intersection",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "accelerate to proceed through the right turn",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    ("accelerate to turn left", LongitudinalAction.GENTLE_ACCELERATE, None),
    (
        "change one lane to the left and accelerate",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "change to the left lane and accelerate",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "change lanes to the left and accelerate to overtake",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "change lanes to the left and accelerate",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "maintain lane and keep speed to proceed through the intersection",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("keep speed through the intersection", LongitudinalAction.MAINTAIN_SPEED, None),
    (
        "keep distance to the stopped lead vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    (
        "keep distance to the cut-in lead vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("keep distance to the cut-in motorcycle", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep distance to the cut-in vehicle", LongitudinalAction.MAINTAIN_SPEED, None),
    (
        "keep distance to the stopped school bus",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("keep distance to the stopped bus", LongitudinalAction.MAINTAIN_SPEED, None),
    (
        "keep distance to the stopped van blocking the lane ahead",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("stop", LongitudinalAction.STOP, None),
    (
        "keep distance to the pedestrian walking near the edge of the lane",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("keep distance to the pedestrian", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep distance to the lead truck", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep distance to the lead vehicle", LongitudinalAction.MAINTAIN_SPEED, None),
    ("match speed with right-lane traffic", LongitudinalAction.MAINTAIN_SPEED, None),
    ("match speed with left-lane traffic", LongitudinalAction.MAINTAIN_SPEED, None),
    ("match speed with traffic", LongitudinalAction.MAINTAIN_SPEED, None),
    ("resume speed to the speed limit", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("resume speed", LongitudinalAction.GENTLE_ACCELERATE, None),
    (
        "create a usable gap to merge into the right lane",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "create a usable gap for a right lane change",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("create a usable gap", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adjust speed and keep safe distance", LongitudinalAction.GENTLE_DECELERATE, None),
    ("wait for a gap", LongitudinalAction.MAINTAIN_SPEED, None),
    # --- Stage 2: close paraphrase variants (25 longitudinal) ---
    (
        "hold a safe following distance to the lead vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    (
        "maintain a safe following gap to the lead vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("preserve space behind the lead vehicle", LongitudinalAction.MAINTAIN_SPEED, None),
    ("pace the vehicle ahead", LongitudinalAction.MAINTAIN_SPEED, None),
    ("follow the lead vehicle at a safe gap", LongitudinalAction.MAINTAIN_SPEED, None),
    (
        "keep a safe buffer to the cut-in vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    (
        "match the flow of traffic after merging",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("hold speed through the green light", LongitudinalAction.MAINTAIN_SPEED, None),
    ("continue at traffic speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep rolling at the current speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("ease off for the lead vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "back off from the slowing lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("reduce speed for the red light", LongitudinalAction.GENTLE_DECELERATE, None),
    ("bleed speed before the curve", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow for the narrowed work zone", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow for the speed hump", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow for the school crossing", LongitudinalAction.GENTLE_DECELERATE, None),
    ("make room for the merging vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    ("open a gap for the merge", LongitudinalAction.GENTLE_DECELERATE, None),
    ("yield space to the merging vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    ("come to a full stop at the red light", LongitudinalAction.STOP, None),
    ("stop behind the stopped lead vehicle", LongitudinalAction.STOP, None),
    ("settle to a stop at the stop line", LongitudinalAction.STOP, None),
    (
        "pull away when the light turns green",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    ("accelerate with the through traffic", LongitudinalAction.GENTLE_ACCELERATE, None),
    # --- Stage 3: additional distinct paraphrases (25 longitudinal) ---
    (
        "brake hard to avoid the queue of stopped traffic ahead",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "apply a firm brake for stopped traffic",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "perform an emergency stop for the obstacle ahead",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "hard brake for the sudden cut-in",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "accelerate briskly to merge ahead of approaching traffic",
        LongitudinalAction.STRONG_ACCELERATE,
        "strong",
    ),
    (
        "speed up quickly to enter the gap",
        LongitudinalAction.STRONG_ACCELERATE,
        "strong",
    ),
    (
        "accelerate strongly to clear the intersection",
        LongitudinalAction.STRONG_ACCELERATE,
        "strong",
    ),
    ("pick up speed to complete the merge", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("gain speed after the turn", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("increase speed once the lane opens", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("creep forward from the stop line", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("inch forward into the gap", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("roll forward cautiously", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("coast at a steady pace", LongitudinalAction.MAINTAIN_SPEED, None),
    ("maintain steady speed in lane", LongitudinalAction.MAINTAIN_SPEED, None),
    ("hold position while waiting for the gap", LongitudinalAction.STOP, None),
    ("remain stopped for the red signal", LongitudinalAction.STOP, None),
    ("stay stopped behind the lead vehicle", LongitudinalAction.STOP, None),
    ("bring the vehicle to a halt", LongitudinalAction.STOP, None),
    ("come to a stop for the stop sign", LongitudinalAction.STOP, None),
    ("reverse slowly out of the space", LongitudinalAction.REVERSE, None),
    ("back up to reposition", LongitudinalAction.REVERSE, None),
    ("back out of the driveway", LongitudinalAction.REVERSE, None),
    ("reduce velocity for the ramp", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "decelerate smoothly before the crosswalk",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
]


_P_C_CANDIDATE_LATERAL_PATTERNS: list[tuple[str, LateralAction, str | None]] = [
    # --- Stage 1: mined recurring surface forms (50 lateral) ---
    (
        "lane change to the right and slow to pull into the curbside stop",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right and slow to pull into the curbside bay behind the stopped school bus",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right and pull over to the curb to stop behind the stopped school bus",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right and pull over to the curb to stop behind the school bus",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right and turn into the parking-lot entrance",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right and pull into the curbside entrance",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right and pull over to the curb",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right to prepare for the upcoming right turn",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the left to bypass slow traffic blocking the current lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to avoid slow traffic in the same lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to avoid the slow truck blocking the current lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to pass slow lead vehicles ahead",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to pass slow traffic ahead in our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to pass slow traffic ahead",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to pass the slow lead vehicle ahead",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "change to the left lane to overtake slower vehicles ahead while a safe gap opens",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "change to the left lane to overtake slower traffic ahead and maintain flow",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "change to the left lane to stay out of the curbside bus stop bay",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "change one lane to the left to stay on the mainline and avoid the right-hand off-ramp",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("change one lane to the left and proceed", LateralAction.STEER_LEFT, None),
    ("change one lane to the left", LateralAction.STEER_LEFT, None),
    (
        "change lane to the left to pass slower lead traffic and use an available gap",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("change lane to the left", LateralAction.STEER_LEFT, None),
    (
        "change lanes to the left to overtake slower traffic ahead and maintain flow",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "change lanes to the left and accelerate to overtake",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("change lanes to the left and accelerate", LateralAction.STEER_LEFT, None),
    ("change lanes to the left", LateralAction.STEER_LEFT, None),
    ("change to the left lane and slow to a stop", LateralAction.STEER_LEFT, None),
    ("change to the left lane and queue", LateralAction.STEER_LEFT, None),
    ("change to the left lane and accelerate", LateralAction.STEER_LEFT, None),
    ("change to the left lane", LateralAction.STEER_LEFT, None),
    ("merge left", LateralAction.STEER_LEFT, None),
    (
        "nudge to the left in the same lane to clear the traffic cones blocking the center of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to increase clearance from the parked vehicle blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to increase clearance from the stopped truck blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to increase clearance to the stopped vehicle with open door on the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to increase clearance to the stopped vehicle on the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to increase clearance to the van encroaching from the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to increase clearance to the vehicle merging from the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the construction barricade blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the construction barrier blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the roadwork barricade blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the construction equipment blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the construction truck blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the construction cones blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the parked vehicle blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the parked car blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped maintenance truck blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped truck blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped vehicle blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped vehicle blocking the lane ahead",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped van blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the cones blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to pass the stopped van blocking the lane ahead",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to pass the stopped van blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to pass the stopped van blocking the lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to a vehicle merging from the right into our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to a car merging from the right into our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to the vehicle merging from the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to the vehicle encroaching from the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to the vehicle encroaching from the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to parked vehicles blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to stopped vehicles blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to cones blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to debris blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("nudge left", LateralAction.STEER_LEFT, None),
    (
        "change lanes to the right and take the off-ramp",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change lanes to the right and merge into the right-turn lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change lanes to the right and turn into the driveway",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change lanes to the right and pull into the curbside bay to stop behind the stopped school buses",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change lanes to the right to bypass slower lead traffic",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change lanes to the right to use a clear right lane and maintain progress",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("change lanes to the right", LateralAction.STEER_RIGHT, None),
    (
        "change to the right-turn-only lane and decelerate to a stop",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change to the right-turn lane and slow to a stop",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change to the right lane and merge behind right-lane traffic",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("change to the right lane and merge into a gap", LateralAction.STEER_RIGHT, None),
    ("change to the right lane and prepare to stop", LateralAction.STEER_RIGHT, None),
    ("change to the right lane and slow to a stop", LateralAction.STEER_RIGHT, None),
    ("change to the right lane", LateralAction.STEER_RIGHT, None),
    ("change lane to the right", LateralAction.STEER_RIGHT, None),
    ("lane change to the right", LateralAction.STEER_RIGHT, None),
    ("split to the right to take the off-ramp", LateralAction.STEER_RIGHT, None),
    (
        "nudge to the right to create clearance for the oncoming van encroaching into our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to create clearance for the oncoming bus",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to create clearance for the oncoming van",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to create space for the oncoming van",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right to increase clearance to the oncoming vehicle",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to increase clearance from the oncoming bus",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to the vehicle encroaching from the left side of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to the vehicle encroaching from the left",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to the concrete barrier encroaching from the left",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to the construction barrels blocking the left side of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to the cones blocking the left side of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to debris blocking the left side of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to clear the construction cones blocking the left side of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to clear the cones blocking the center of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to clear the construction cones blocking the center of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to clear the traffic cones blocking the center of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to clear the cone blocking the center of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to avoid the traffic cones blocking the lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("nudge right", LateralAction.STEER_RIGHT, None),
    ("turn right to enter the parking lot", LateralAction.STEER_RIGHT, None),
    ("turn right to enter the driveway", LateralAction.STEER_RIGHT, None),
    ("turn right to enter the main road", LateralAction.STEER_RIGHT, None),
    ("turn right at the intersection", LateralAction.STEER_RIGHT, None),
    ("turn right", LateralAction.STEER_RIGHT, None),
    ("turn left to enter the main road", LateralAction.STEER_LEFT, None),
    ("turn left through the intersection", LateralAction.STEER_LEFT, None),
    ("turn left at the intersection", LateralAction.STEER_LEFT, None),
    (
        "maintain lane and wait for a suitable gap to merge right",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    (
        "maintain lane and wait for a gap for a left-lane merge",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    (
        "maintain lane and wait for a gap to merge right",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    ("maintain lane and wait for a gap", LateralAction.GO_STRAIGHT, None),
    (
        "keep lane to stay clear of the construction cones ahead",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    ("keep lane", LateralAction.GO_STRAIGHT, None),
    # --- Stage 2: close paraphrase variants (25 lateral) ---
    (
        "hold the current lane while waiting for a merge gap",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    ("stay in the lane until the adjacent car clears", LateralAction.GO_STRAIGHT, None),
    ("continue centered in the lane", LateralAction.GO_STRAIGHT, None),
    ("track the lane center through the curve", LateralAction.GO_STRAIGHT, None),
    (
        "keep the vehicle centered between the lane markings",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    ("move one lane left to pass slower traffic", LateralAction.STEER_LEFT, None),
    ("merge into the left lane once the gap opens", LateralAction.STEER_LEFT, None),
    ("shift left to avoid the blocked right side", LateralAction.STEER_LEFT, None),
    ("edge left around the stopped vehicle", LateralAction.STEER_LEFT, None),
    ("give the parked car on the right more room", LateralAction.STEER_LEFT, None),
    ("move left within the lane for clearance", LateralAction.STEER_LEFT, None),
    ("take the left lane to overtake", LateralAction.STEER_LEFT, None),
    ("bear left to stay on the mainline", LateralAction.STEER_LEFT, None),
    ("turn left through the green arrow", LateralAction.STEER_LEFT, None),
    ("merge right toward the exit lane", LateralAction.STEER_RIGHT, None),
    ("move one lane right for the off-ramp", LateralAction.STEER_RIGHT, None),
    ("shift right away from the left-side cones", LateralAction.STEER_RIGHT, None),
    ("edge right around the barrels", LateralAction.STEER_RIGHT, None),
    (
        "give the oncoming vehicle more room on the right",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("move right within the lane for clearance", LateralAction.STEER_RIGHT, None),
    ("take the right-turn lane", LateralAction.STEER_RIGHT, None),
    ("bear right onto the ramp", LateralAction.STEER_RIGHT, None),
    ("turn right through the green arrow", LateralAction.STEER_RIGHT, None),
    ("pull over toward the curbside bay", LateralAction.STEER_RIGHT, None),
    ("enter the driveway on the right", LateralAction.STEER_RIGHT, None),
    # --- Stage 3: additional distinct paraphrases (25 lateral) ---
    (
        "nudge slightly right within the lane to give the cyclist more clearance",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "make a small rightward adjustment for clearance",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("make a small leftward adjustment for clearance", LateralAction.STEER_LEFT, None),
    ("drift gently left within the lane", LateralAction.STEER_LEFT, None),
    ("drift gently right within the lane", LateralAction.STEER_RIGHT, None),
    ("swerve left around the obstruction", LateralAction.STEER_LEFT, None),
    ("swerve right around the obstruction", LateralAction.STEER_RIGHT, None),
    (
        "steer sharply left to avoid the hazard",
        LateralAction.SHARP_STEER_LEFT,
        "strong",
    ),
    (
        "steer sharply right to avoid the hazard",
        LateralAction.SHARP_STEER_RIGHT,
        "strong",
    ),
    ("hard left around the obstacle", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("hard right around the obstacle", LateralAction.SHARP_STEER_RIGHT, "strong"),
    ("make a u-turn", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("reverse left out of the spot", LateralAction.REVERSE_LEFT, None),
    ("reverse right out of the spot", LateralAction.REVERSE_RIGHT, None),
    ("back left into the driveway", LateralAction.REVERSE_LEFT, None),
    ("back right into the driveway", LateralAction.REVERSE_RIGHT, None),
    ("continue straight through the intersection", LateralAction.GO_STRAIGHT, None),
    ("go straight through the green light", LateralAction.GO_STRAIGHT, None),
    ("proceed straight ahead", LateralAction.GO_STRAIGHT, None),
    ("follow the lane through the bend", LateralAction.GO_STRAIGHT, None),
    ("hold lane position", LateralAction.GO_STRAIGHT, None),
    ("stay centered in lane", LateralAction.GO_STRAIGHT, None),
    ("keep at the center of the lane", LateralAction.GO_STRAIGHT, None),
    ("change lanes left", LateralAction.STEER_LEFT, None),
    ("keep lane", LateralAction.GO_STRAIGHT, None),
]


# Inactive verbose set retained for provenance.  The parser uses the compact
# compositional lexicon defined below instead.  Keeping this named copy makes
# it easy to compare against the mined phrases without accidentally activating
# hundreds of scenario-specific strings again.
_VERBOSE_LONGITUDINAL_PATTERNS: list[tuple[str, LongitudinalAction, str | None]] = [
    # --- Stage 1: mined recurring surface forms (50 longitudinal) ---
    (
        "stop to keep distance to the stopped lead vehicle ahead",
        LongitudinalAction.STOP,
        None,
    ),
    ("stop to keep distance to the lead vehicle", LongitudinalAction.STOP, None),
    ("stop for the red right-turn traffic light", LongitudinalAction.STOP, None),
    ("stop for the red traffic light ahead", LongitudinalAction.STOP, None),
    ("stop for the red traffic light", LongitudinalAction.STOP, None),
    ("stop at the stop line", LongitudinalAction.STOP, None),
    ("stop due to the red traffic light", LongitudinalAction.STOP, None),
    ("slow down to stop for the red traffic light", LongitudinalAction.STOP, None),
    ("slow to a stop", LongitudinalAction.STOP, None),
    ("prepare to stop", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "yield to the cut-in vehicle from the right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "yield to the cut-in vehicle from the left",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("yield to the cut-in vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "slow down to yield to the cut-in vehicle from the left",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "decelerate to maintain a safe distance from the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "decelerate to maintain a safe distance to the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "decelerate to create a gap for merging right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed to maintain a safe distance from the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the narrowed lane marked by construction cones",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("adapt speed for the narrowed lane", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "adapt speed for the raised crosswalk",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "adapt speed for the school crossing ahead",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("adapt speed for the speed bump", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the speed hump", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the off-ramp", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the right curve", LongitudinalAction.GENTLE_DECELERATE, None),
    ("adapt speed for the left curve", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "slow down to maintain a safe distance to the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to keep distance to the lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down for the red traffic light ahead",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("slow down for the red traffic light", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "slow down to create a gap for vehicles merging from the right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for merging right",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    (
        "slow down to create a gap for merging left",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("slow down and maintain lane", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow down", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "accelerate to match left-lane traffic and create a usable gap for a left lane change",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "accelerate to proceed through the intersection",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "accelerate to proceed through the right turn",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    ("accelerate to turn left", LongitudinalAction.GENTLE_ACCELERATE, None),
    (
        "change to the left lane and accelerate",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "change lanes to the left and accelerate",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    (
        "maintain lane and keep speed to proceed through the intersection",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("keep speed through the intersection", LongitudinalAction.MAINTAIN_SPEED, None),
    (
        "keep distance to the stopped lead vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("keep distance to the cut-in vehicle", LongitudinalAction.MAINTAIN_SPEED, None),
    (
        "keep distance to the stopped school bus",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("stop", LongitudinalAction.STOP, None),
    ("keep distance to the lead vehicle", LongitudinalAction.MAINTAIN_SPEED, None),
    ("resume speed to the speed limit", LongitudinalAction.GENTLE_ACCELERATE, None),
    # --- Stage 2: close paraphrase variants (25 longitudinal) ---
    (
        "hold a safe following distance to the lead vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    (
        "maintain a safe following gap to the lead vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("preserve space behind the lead vehicle", LongitudinalAction.MAINTAIN_SPEED, None),
    ("pace the vehicle ahead", LongitudinalAction.MAINTAIN_SPEED, None),
    ("follow the lead vehicle at a safe gap", LongitudinalAction.MAINTAIN_SPEED, None),
    (
        "keep a safe buffer to the cut-in vehicle",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    (
        "match the flow of traffic after merging",
        LongitudinalAction.MAINTAIN_SPEED,
        None,
    ),
    ("hold speed through the green light", LongitudinalAction.MAINTAIN_SPEED, None),
    ("continue at traffic speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("keep rolling at the current speed", LongitudinalAction.MAINTAIN_SPEED, None),
    ("ease off for the lead vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    (
        "back off from the slowing lead vehicle",
        LongitudinalAction.GENTLE_DECELERATE,
        None,
    ),
    ("reduce speed for the red light", LongitudinalAction.GENTLE_DECELERATE, None),
    ("bleed speed before the curve", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow for the narrowed work zone", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow for the speed hump", LongitudinalAction.GENTLE_DECELERATE, None),
    ("slow for the school crossing", LongitudinalAction.GENTLE_DECELERATE, None),
    ("make room for the merging vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    ("open a gap for the merge", LongitudinalAction.GENTLE_DECELERATE, None),
    ("yield space to the merging vehicle", LongitudinalAction.GENTLE_DECELERATE, None),
    ("come to a full stop at the red light", LongitudinalAction.STOP, None),
    ("stop behind the stopped lead vehicle", LongitudinalAction.STOP, None),
    ("settle to a stop at the stop line", LongitudinalAction.STOP, None),
    (
        "pull away when the light turns green",
        LongitudinalAction.GENTLE_ACCELERATE,
        None,
    ),
    ("accelerate with the through traffic", LongitudinalAction.GENTLE_ACCELERATE, None),
    # --- Stage 3: additional distinct paraphrases (25 longitudinal) ---
    (
        "brake hard to avoid the queue of stopped traffic ahead",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "apply a firm brake for stopped traffic",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "perform an emergency stop for the obstacle ahead",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "hard brake for the sudden cut-in",
        LongitudinalAction.STRONG_DECELERATE,
        "strong",
    ),
    (
        "accelerate briskly to merge ahead of approaching traffic",
        LongitudinalAction.STRONG_ACCELERATE,
        "strong",
    ),
    (
        "speed up quickly to enter the gap",
        LongitudinalAction.STRONG_ACCELERATE,
        "strong",
    ),
    (
        "accelerate strongly to clear the intersection",
        LongitudinalAction.STRONG_ACCELERATE,
        "strong",
    ),
    ("pick up speed to complete the merge", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("gain speed after the turn", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("increase speed once the lane opens", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("creep forward from the stop line", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("inch forward into the gap", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("accelerate", LongitudinalAction.GENTLE_ACCELERATE, None),
    ("coast at a steady pace", LongitudinalAction.MAINTAIN_SPEED, None),
    ("maintain steady speed in lane", LongitudinalAction.MAINTAIN_SPEED, None),
    ("hold position while waiting for the gap", LongitudinalAction.STOP, None),
    ("remain stopped for the red signal", LongitudinalAction.STOP, None),
    ("stay stopped behind the lead vehicle", LongitudinalAction.STOP, None),
    ("bring the vehicle to a halt", LongitudinalAction.STOP, None),
    ("come to a stop", LongitudinalAction.STOP, None),
    ("reverse slowly out of the space", LongitudinalAction.REVERSE, None),
    ("back up to reposition", LongitudinalAction.REVERSE, None),
    ("back out of the driveway", LongitudinalAction.REVERSE, None),
    ("reduce velocity for the ramp", LongitudinalAction.GENTLE_DECELERATE, None),
    ("decelerate", LongitudinalAction.GENTLE_DECELERATE, None),
]


_VERBOSE_LATERAL_PATTERNS: list[tuple[str, LateralAction, str | None]] = [
    # --- Stage 1: mined recurring surface forms (50 lateral) ---
    (
        "lane change to the right and slow to pull into the curbside stop",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right and pull over to the curb to stop behind the school bus",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the right to prepare for the upcoming right turn",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "lane change to the left to bypass slow traffic blocking the current lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to avoid slow traffic in the same lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to pass slow lead vehicles ahead",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "lane change to the left to pass the slow lead vehicle ahead",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "change to the left lane to overtake slower vehicles ahead while a safe gap opens",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "change one lane to the left to stay on the mainline and avoid the right-hand off-ramp",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("change one lane to the left", LateralAction.STEER_LEFT, None),
    (
        "change lane to the left to pass slower lead traffic and use an available gap",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("change lane to the left", LateralAction.STEER_LEFT, None),
    (
        "change lanes to the left to overtake slower traffic ahead and maintain flow",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("change lanes to the left and accelerate", LateralAction.STEER_LEFT, None),
    ("change lanes to the left", LateralAction.STEER_LEFT, None),
    ("change to the left lane and slow to a stop", LateralAction.STEER_LEFT, None),
    ("change to the left lane", LateralAction.STEER_LEFT, None),
    ("merge left", LateralAction.STEER_LEFT, None),
    (
        "nudge to the left in the same lane to clear the traffic cones blocking the center of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to increase clearance to the stopped vehicle with open door on the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to increase clearance to the van encroaching from the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left to increase clearance to the vehicle merging from the right",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the construction barricade blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the construction cones blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the parked vehicle blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped truck blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped vehicle blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge to the left to clear the stopped van blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to the vehicle encroaching from the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to parked vehicles blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    (
        "nudge left due to debris blocking the right side of our lane",
        LateralAction.STEER_LEFT,
        None,
    ),
    ("nudge left", LateralAction.STEER_LEFT, None),
    (
        "change lanes to the right and take the off-ramp",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change lanes to the right and merge into the right-turn lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "change lanes to the right and turn into the driveway",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("change lanes to the right", LateralAction.STEER_RIGHT, None),
    (
        "change to the right-turn-only lane and decelerate to a stop",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("change to the right lane and merge into a gap", LateralAction.STEER_RIGHT, None),
    ("change to the right lane and prepare to stop", LateralAction.STEER_RIGHT, None),
    ("change to the right lane", LateralAction.STEER_RIGHT, None),
    ("change lane to the right", LateralAction.STEER_RIGHT, None),
    ("lane change to the right", LateralAction.STEER_RIGHT, None),
    ("split to the right to take the off-ramp", LateralAction.STEER_RIGHT, None),
    (
        "nudge to the right to create clearance for the oncoming van",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right to increase clearance to the oncoming vehicle",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to the vehicle encroaching from the left side of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge right due to the concrete barrier encroaching from the left",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to clear the construction cones blocking the left side of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "nudge to the right to clear the traffic cones blocking the center of our lane",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("nudge right", LateralAction.STEER_RIGHT, None),
    # --- Stage 2: close paraphrase variants (25 lateral) ---
    (
        "hold the current lane while waiting for a merge gap",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    ("stay in the lane until the adjacent car clears", LateralAction.GO_STRAIGHT, None),
    ("continue centered in the lane", LateralAction.GO_STRAIGHT, None),
    ("track the lane center through the curve", LateralAction.GO_STRAIGHT, None),
    (
        "keep the vehicle centered between the lane markings",
        LateralAction.GO_STRAIGHT,
        None,
    ),
    ("move one lane left to pass slower traffic", LateralAction.STEER_LEFT, None),
    ("merge into the left lane once the gap opens", LateralAction.STEER_LEFT, None),
    ("shift left to avoid the blocked right side", LateralAction.STEER_LEFT, None),
    ("edge left around the stopped vehicle", LateralAction.STEER_LEFT, None),
    ("give the parked car on the right more room", LateralAction.STEER_LEFT, None),
    ("move left within the lane for clearance", LateralAction.STEER_LEFT, None),
    ("take the left lane to overtake", LateralAction.STEER_LEFT, None),
    ("bear left to stay on the mainline", LateralAction.STEER_LEFT, None),
    ("turn left through the green arrow", LateralAction.STEER_LEFT, None),
    ("merge right toward the exit lane", LateralAction.STEER_RIGHT, None),
    ("move one lane right for the off-ramp", LateralAction.STEER_RIGHT, None),
    ("shift right away from the left-side cones", LateralAction.STEER_RIGHT, None),
    ("edge right around the barrels", LateralAction.STEER_RIGHT, None),
    (
        "give the oncoming vehicle more room on the right",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("move right within the lane for clearance", LateralAction.STEER_RIGHT, None),
    ("take the right-turn lane", LateralAction.STEER_RIGHT, None),
    ("bear right onto the ramp", LateralAction.STEER_RIGHT, None),
    ("turn right through the green arrow", LateralAction.STEER_RIGHT, None),
    ("pull over toward the curbside bay", LateralAction.STEER_RIGHT, None),
    ("enter the driveway on the right", LateralAction.STEER_RIGHT, None),
    # --- Stage 3: additional distinct paraphrases (25 lateral) ---
    (
        "nudge slightly right within the lane to give the cyclist more clearance",
        LateralAction.STEER_RIGHT,
        None,
    ),
    (
        "make a small rightward adjustment for clearance",
        LateralAction.STEER_RIGHT,
        None,
    ),
    ("make a small leftward adjustment for clearance", LateralAction.STEER_LEFT, None),
    ("drift gently left within the lane", LateralAction.STEER_LEFT, None),
    ("drift gently right within the lane", LateralAction.STEER_RIGHT, None),
    ("swerve left around the obstruction", LateralAction.STEER_LEFT, None),
    ("swerve right around the obstruction", LateralAction.STEER_RIGHT, None),
    (
        "steer sharply left to avoid the hazard",
        LateralAction.SHARP_STEER_LEFT,
        "strong",
    ),
    (
        "steer sharply right to avoid the hazard",
        LateralAction.SHARP_STEER_RIGHT,
        "strong",
    ),
    ("hard left around the obstacle", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("hard right around the obstacle", LateralAction.SHARP_STEER_RIGHT, "strong"),
    ("make a u-turn", LateralAction.SHARP_STEER_LEFT, "strong"),
    ("reverse left out of the spot", LateralAction.REVERSE_LEFT, None),
    ("reverse right out of the spot", LateralAction.REVERSE_RIGHT, None),
    ("back left into the driveway", LateralAction.REVERSE_LEFT, None),
    ("back right into the driveway", LateralAction.REVERSE_RIGHT, None),
    ("continue straight through the intersection", LateralAction.GO_STRAIGHT, None),
    ("go straight through the green light", LateralAction.GO_STRAIGHT, None),
    ("proceed straight ahead", LateralAction.GO_STRAIGHT, None),
    ("follow the lane through the bend", LateralAction.GO_STRAIGHT, None),
    ("hold lane position", LateralAction.GO_STRAIGHT, None),
    ("stay centered in lane", LateralAction.GO_STRAIGHT, None),
    ("keep at the center of the lane", LateralAction.GO_STRAIGHT, None),
    ("change lanes left", LateralAction.STEER_LEFT, None),
    ("keep lane", LateralAction.GO_STRAIGHT, None),
]
"""


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


# Previous compact seed lexicon (commented out for provenance).
# _LONGITUDINAL_PATTERNS: list[tuple[str, LongitudinalAction, str | None]] = [
#     # --- Stop ---
#     ("come to a stop", LongitudinalAction.STOP, None),
#     ("come to a halt", LongitudinalAction.STOP, None),
#     ("bring to a stop", LongitudinalAction.STOP, None),
#     ("full stop", LongitudinalAction.STOP, None),
#     ("hold position", LongitudinalAction.STOP, None),
#     ("remain stopped", LongitudinalAction.STOP, None),
#     ("stay stopped", LongitudinalAction.STOP, None),
#     ("stopping", LongitudinalAction.STOP, None),
#     ("stop", LongitudinalAction.STOP, None),
#     ("halt", LongitudinalAction.STOP, None),
#     ("yield", LongitudinalAction.GENTLE_DECELERATE, None),
#     # --- Reverse ---
#     ("back up", LongitudinalAction.REVERSE, None),
#     ("reverse", LongitudinalAction.REVERSE, None),
#     # --- Decelerate ---
#     ("emergency brake", LongitudinalAction.STRONG_DECELERATE, "strong"),
#     ("hard brake", LongitudinalAction.STRONG_DECELERATE, "strong"),
#     ("brake hard", LongitudinalAction.STRONG_DECELERATE, "strong"),
#     ("brake heavily", LongitudinalAction.STRONG_DECELERATE, "strong"),
#     ("slam the brakes", LongitudinalAction.STRONG_DECELERATE, "strong"),
#     ("decelerate hard", LongitudinalAction.STRONG_DECELERATE, "strong"),
#     ("decelerating", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("decelerate", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("slowing down", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("slowing", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("slow down", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("slow", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("braking", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("brake", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("ease off", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("ease the throttle", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("reduce speed", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("reduce velocity", LongitudinalAction.GENTLE_DECELERATE, None),
#     ("coast", LongitudinalAction.MAINTAIN_SPEED, None),
#     # --- Accelerate ---
#     ("accelerate hard", LongitudinalAction.STRONG_ACCELERATE, "strong"),
#     ("speed up quickly", LongitudinalAction.STRONG_ACCELERATE, "strong"),
#     ("rapid acceleration", LongitudinalAction.STRONG_ACCELERATE, "strong"),
#     ("floor it", LongitudinalAction.STRONG_ACCELERATE, "strong"),
#     ("accelerating", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("accelerate", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("speed up", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("increase speed", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("pick up speed", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("gain speed", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("resume speed", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("resume from stop", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("creep forward", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("inch forward", LongitudinalAction.GENTLE_ACCELERATE, None),
#     ("pull forward", LongitudinalAction.GENTLE_ACCELERATE, None),
#     # --- Maintain ---
#     ("maintaining speed", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("maintain speed", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("keeping speed", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("keep speed", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("hold speed", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("steady pace", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("constant speed", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("cruise", LongitudinalAction.MAINTAIN_SPEED, None),
#     ("continue at the same speed", LongitudinalAction.MAINTAIN_SPEED, None),
# ]
#
#
# _LATERAL_PATTERNS: list[tuple[str, LateralAction, str | None]] = [
#     # Reverse-with-side phrases first
#     ("reverse left", LateralAction.REVERSE_LEFT, None),
#     ("reverse right", LateralAction.REVERSE_RIGHT, None),
#     ("back left", LateralAction.REVERSE_LEFT, None),
#     ("back right", LateralAction.REVERSE_RIGHT, None),
#     # Sharp steer
#     ("sharp left", LateralAction.SHARP_STEER_LEFT, "strong"),
#     ("sharp right", LateralAction.SHARP_STEER_RIGHT, "strong"),
#     ("hard left", LateralAction.SHARP_STEER_LEFT, "strong"),
#     ("hard right", LateralAction.SHARP_STEER_RIGHT, "strong"),
#     ("u-turn", LateralAction.SHARP_STEER_LEFT, "strong"),
#     ("uturn", LateralAction.SHARP_STEER_LEFT, "strong"),
#     # Steer / turn left/right
#     ("turn left", LateralAction.STEER_LEFT, None),
#     ("turn right", LateralAction.STEER_RIGHT, None),
#     ("steer left", LateralAction.STEER_LEFT, None),
#     ("steer right", LateralAction.STEER_RIGHT, None),
#     ("merge left", LateralAction.STEER_LEFT, None),
#     ("merge right", LateralAction.STEER_RIGHT, None),
#     ("change lane to the left", LateralAction.STEER_LEFT, None),
#     ("change lane to the right", LateralAction.STEER_RIGHT, None),
#     ("change lanes to the left", LateralAction.STEER_LEFT, None),
#     ("change lanes to the right", LateralAction.STEER_RIGHT, None),
#     ("change to the left lane", LateralAction.STEER_LEFT, None),
#     ("change to the right lane", LateralAction.STEER_RIGHT, None),
#     ("changing lanes left", LateralAction.STEER_LEFT, None),
#     ("changing lanes right", LateralAction.STEER_RIGHT, None),
#     ("changing lane left", LateralAction.STEER_LEFT, None),
#     ("changing lane right", LateralAction.STEER_RIGHT, None),
#     ("change lanes left", LateralAction.STEER_LEFT, None),
#     ("change lanes right", LateralAction.STEER_RIGHT, None),
#     ("lane change left", LateralAction.STEER_LEFT, None),
#     ("lane change right", LateralAction.STEER_RIGHT, None),
#     ("nudge left", LateralAction.STEER_LEFT, None),
#     ("nudge right", LateralAction.STEER_RIGHT, None),
#     ("swerve left", LateralAction.STEER_LEFT, None),
#     ("swerve right", LateralAction.STEER_RIGHT, None),
#     ("bear left", LateralAction.STEER_LEFT, None),
#     ("bear right", LateralAction.STEER_RIGHT, None),
#     ("veer left", LateralAction.STEER_LEFT, None),
#     ("veer right", LateralAction.STEER_RIGHT, None),
#     # Straight / lane keep
#     ("keeping lane", LateralAction.GO_STRAIGHT, None),
#     ("keep lane", LateralAction.GO_STRAIGHT, None),
#     ("keep at the center of the lane", LateralAction.GO_STRAIGHT, None),
#     ("keep center of the lane", LateralAction.GO_STRAIGHT, None),
#     ("stay centered in lane", LateralAction.GO_STRAIGHT, None),
#     ("stay in lane", LateralAction.GO_STRAIGHT, None),
#     ("hold the lane", LateralAction.GO_STRAIGHT, None),
#     ("lane keeping", LateralAction.GO_STRAIGHT, None),
#     ("lane keep", LateralAction.GO_STRAIGHT, None),
#     ("continue straight", LateralAction.GO_STRAIGHT, None),
#     ("go straight", LateralAction.GO_STRAIGHT, None),
#     ("follow the road", LateralAction.GO_STRAIGHT, None),
#     ("straight ahead", LateralAction.GO_STRAIGHT, None),
# ]

_PREVIOUS_COMPACT_SEED_LEXICON_COMMENTED_OUT = True


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
