# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Hybrid CoT/trajectory consistency prompt.

This prompt keeps the LLM in charge of language understanding while giving it
structured trajectory-side evidence: ego-frame numbers, optional lane-relative
numbers, and compact deterministic trajectory action hints when available.
"""

import json
from typing import Any


def _ego_stats_text(stats: dict) -> str:
    return (
        f"Duration: {stats.get('duration_s', '?')}s, "
        f"Path length: {stats.get('total_path_length_m', '?')}m, "
        f"Final position: ({stats.get('final_longitudinal_m', '?')}m forward, "
        f"{stats.get('final_lateral_m', '?')}m lateral), "
        f"Max lateral deviation: {stats.get('max_lateral_m', '?')}m / "
        f"{stats.get('min_lateral_m', '?')}m, "
        f"Speed: {stats.get('mean_speed_ms', '?')} avg / "
        f"{stats.get('max_speed_ms', '?')} max m/s"
    )


def _lane_stats_text(stats: dict) -> str:
    return (
        f"Reference: {stats.get('lane_reference', 'lane_center')}, "
        f"Lane progress: {stats.get('lane_path_length_m', '?')}m, "
        f"Offset start/end: {stats.get('initial_offset_m', '?')}m -> "
        f"{stats.get('final_offset_m', '?')}m, "
        f"Delta offset: {stats.get('delta_offset_m', '?')}m, "
        f"Offset range: [{stats.get('min_offset_m', '?')}, "
        f"{stats.get('max_offset_m', '?')}] m"
    )


def _format_action_hints(hints: Any) -> str:
    if not hints:
        return "No deterministic trajectory action hints are available."
    try:
        return json.dumps(hints, indent=2, sort_keys=True)
    except TypeError:
        return str(hints)


def _view_sections(traj_features: dict) -> tuple[str, str]:
    ego = traj_features.get("ego_features")
    lane = traj_features.get("lane_features")
    if ego is None and lane is None:
        frame = traj_features.get("summary_stats", {}).get("reference_frame")
        if frame == "lane_center":
            lane = traj_features
        else:
            ego = traj_features

    lane_ok = (
        isinstance(lane, dict)
        and lane.get("summary_stats", {}).get("reference_frame") == "lane_center"
    )

    sections = []
    notes = []
    if ego is not None:
        sections.append(
            "**View 1 - absolute ego frame (+X forward, +Y left):**\n"
            f"{ego.get('markdown_kv', 'No trajectory data available.')}\n\n"
            f"**View 1 summary:** {_ego_stats_text(ego.get('summary_stats', {}))}"
        )
        notes.append(
            "- Ego-frame x/y shows the planned path relative to the vehicle at t=0. "
            "On curved roads, accumulated ego-frame y can reflect road curvature, "
            "so do not judge lateral intent from final y alone."
        )
    if lane_ok:
        sections.append(
            "**View 2 - route-consistent lane-relative frame:**\n"
            f"{lane.get('markdown_kv', 'No trajectory data available.')}\n\n"
            f"**View 2 summary:** {_lane_stats_text(lane.get('summary_stats', {}))}"
        )
        notes.append(
            "- Lane-relative offset_m and delta_offset_m are the primary lateral "
            "signals for lane keeping, nudges, and lane changes. A sustained "
            "delta near a lane width (~3-4m) indicates a lane change."
        )
    if not sections:
        sections.append("No trajectory data available.")
    return "\n\n".join(sections), "\n".join(notes)


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the hybrid consistency evaluation prompt."""
    trajectory_block, frame_notes = _view_sections(traj_features)
    action_hints = _format_action_hints(traj_features.get("trajectory_action_hints"))

    return f"""You are an expert evaluator for autonomous vehicle reasoning systems.
You are screening whether a driving model's stated Chain-of-Thought (CoT) action is
consistent with the trajectory it actually produced.

Use the LLM to understand the CoT language. Use the numeric trajectory evidence and
the deterministic trajectory action hints as execution evidence. The hints are not a
replacement for the numbers; if they conflict, cite the numbers and explain why.

=== INPUT ===

**Chain-of-Thought Reasoning:**
"{cot_text}"

**Predicted Trajectory:**
{trajectory_block}

**Deterministic Trajectory Action Hints:**
{action_hints}

=== COORDINATE FRAME NOTES ===
{frame_notes}
- Longitudinal intent is judged from speed and acceleration.
- Lateral direction is judged from sustained lateral velocity and committed motion,
  preferably lane-relative offset when View 2 is available.
- Judge only whether the executed trajectory matches the action stated in the CoT.

=== HYBRID CONSISTENCY RUBRIC ===
Return a consistency score from 1 to 5 and a binary verdict. The verdict must be
"inconsistent" when score <= 2, otherwise "consistent".

- Score 5: Clear alignment. The trajectory clearly executes the stated action.
- Score 4: High alignment. The trajectory executes the stated action with only
  minor timing or magnitude differences.
- Score 3: Plausible alignment. The action is partially executed or just starting,
  but commitment is visible by the end of the horizon and the maneuver class matches.
- Score 2: Inconsistent. The trajectory performs a different maneuver class, adds
  a significant unmentioned secondary maneuver, or under-executes the stated action
  so weakly that the action effectively does not happen.
- Score 1: Inconsistent contradiction. The trajectory persistently does the opposite
  of an explicit CoT action, such as accelerating while the CoT says to brake, or
  moving left when the CoT commits to moving right.

Use these inconsistency types when verdict is "inconsistent":
- action_mismatch
- direction_contradiction
- under_execution
- unmentioned_secondary_motion

=== OUTPUT FORMAT ===

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "cot_output_alignment": {{
    "verdict": "<consistent | inconsistent>",
    "score": <1-5>,
    "inconsistency_type": "<action_mismatch | direction_contradiction | under_execution | unmentioned_secondary_motion | none>",
    "confidence": "<high | medium | low>",
    "justification": "<one sentence linking the CoT action to specific trajectory numbers>"
  }}
}}"""
