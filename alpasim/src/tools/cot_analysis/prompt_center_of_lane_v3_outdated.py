# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prompt construction for CoT consistency analysis (dual-frame v3).

Pairs with reference_frame="dual": the predicted trajectory is presented in
two complementary views — the absolute ego/rig frame (default prompt-style x/y)
and the route-consistent lane Frenet frame (prompt_center_of_lane_v2-style
s/offset, built from the lane successor graph). The ego view carries absolute
geometry (turns, heading change, displacement); the lane view carries
lane-relative intent (offsets that road curvature cannot explain away).
Single-dimension rubric (v5 structure):
  CoT-Output Alignment (1-5)
"""


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
        f"{stats.get('max_offset_m', '?')}] m, "
        f"Lanes in reference path: {stats.get('lane_segment_count_used', '?')}"
    )


EGO_FRAME_NOTES = """**View 1 — absolute ego frame:**
- X axis: forward (longitudinal). Positive X = vehicle moves forward.
- Y axis: lateral. Positive Y = leftward from vehicle center at t=0. Negative Y = rightward.
- lat_vel: lateral velocity. Positive = moving left. Negative = moving right.
- accel: longitudinal acceleration. Positive = speeding up. Negative = braking.
Caution: on curved roads, Y accumulates from road curvature alone — large |y| at the
end of the horizon does NOT by itself mean a lateral maneuver."""

LANE_FRAME_NOTES = """**View 2 — route-consistent lane Frenet frame:**
The reference line is a single continuous path built by walking the lane successor
graph from the ego's current lane; the whole trajectory is projected into this one
Frenet frame. The reference cannot switch lanes mid-horizon, so changes in offset_m
reflect real ego motion relative to its route, not curvature or reference artifacts.
- s_lane: longitudinal progress along the reference path (meters).
- offset_m: signed lateral offset from the reference path. Positive = left of it, negative = right.
- delta_offset_m: offset_m relative to t=0. Positive means moved left relative to starting lane position.
- lat_vel: lateral velocity with respect to the reference path (positive left, negative right).
- speed: ground speed of the ego (frame-independent, from the raw trajectory).
- accel: rate of change of ground speed (positive speed-up, negative slow-down).
A sustained delta_offset_m of roughly a lane width (~3-4m) indicates a lane change off
the reference route; a sustained fraction of a lane width indicates an in-lane shift."""

HOW_TO_COMBINE = """**How to combine the two views:**
- Lateral intent (lane keep / nudge / lane change): View 2's offset_m and delta_offset_m
  are the primary signals. Use View 1 only as a cross-check.
- Turns and absolute geometry (turn left/right at intersection, merge geometry): View 1
  shows the executed shape directly (heading change via the x/y path), while in View 2 a
  turn that follows the route shows up as steady s_lane progress with bounded offsets.
- Longitudinal intent (brake / accelerate / hold speed): speed and accel describe the
  same physical quantity in both views and should agree; read either.
- If the views appear to conflict laterally, trust View 2 — View 1's y mixes maneuvers
  with road curvature."""


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the CoT consistency evaluation prompt.

    Args:
        cot_text: The chain-of-thought reasoning text from the driver.
        traj_features: Dict from compute_trajectory_features(), ideally with
            reference_frame="dual". Degrades to a single view when only one
            frame is available.

    Returns:
        Formatted prompt string for LLM judge.
    """
    ego = traj_features.get("ego_features")
    lane = traj_features.get("lane_features")
    if ego is None and lane is None:
        # Not a dual features dict: treat it as a single-frame input.
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
            "**Predicted Trajectory — View 1 (absolute ego frame: +X forward, +Y left):**\n"
            f"{ego.get('markdown_kv', 'No trajectory data available.')}\n\n"
            f"**View 1 summary:** {_ego_stats_text(ego.get('summary_stats', {}))}"
        )
        notes.append(EGO_FRAME_NOTES)
    if lane_ok:
        sections.append(
            "**Predicted Trajectory — View 2 (route-consistent lane Frenet frame):**\n"
            f"{lane.get('markdown_kv', 'No trajectory data available.')}\n\n"
            f"**View 2 summary:** {_lane_stats_text(lane.get('summary_stats', {}))}"
        )
        notes.append(LANE_FRAME_NOTES)
    if ego is not None and lane_ok:
        notes.append(HOW_TO_COMBINE)
    elif ego is not None:
        notes.append(
            "Note: View 2 (lane-relative features) is unavailable for this sample; "
            "where the rubric or examples mention View 2, judge from the ego-frame "
            "View 1 alone and treat smooth lateral drift as possibly caused by road "
            "curvature."
        )

    trajectory_block = "\n\n".join(sections)
    frame_notes = "\n\n".join(notes)

    prompt = f"""You are an expert evaluator for autonomous vehicle reasoning systems.
You are given the Chain-of-Thought (CoT) reasoning produced by a driving model, along with
the raw numerical trajectory it actually produced, presented in two complementary frames.

=== INPUT ===

**Chain-of-Thought Reasoning:**
"{cot_text}"

{trajectory_block}

=== COORDINATE FRAMES ===

{frame_notes}

=== EVALUATION CRITERIA ===
Provide a holistic score (1-5) for CoT-Output Alignment. Ensure to provide justification for score deductions.

**CoT-Output Alignment**
Does the executed trajectory match the ACTION stated in the CoT? Judge only the action.

Rubric:
- **Score 5: Clear alignment.** The trajectory clearly executes the stated action (correct direction and rough kinematic magnitude).
- **Score 4: High alignment.** The trajectory executes the stated action with minor discrepancies in magnitude or timing.
- **Score 3: Plausible alignment.** The stated maneuver is only initiated or partially executed but shows clear, building commitment (e.g., deceleration or lane-relative offset ramping up by the end of the horizon), or alignment is ambiguous due to the short horizon — and the trajectory is the same maneuver class as stated.
- **Score 2: Different maneuver class, or under-execution.** Either (a) the trajectory performs a clearly different category of maneuver than stated, on the axis the CoT asserts (e.g., the CoT states slowing in lane but the trajectory instead makes a committed lane change at unchanged speed; the CoT states a small in-lane adjustment but the trajectory executes a full lane change or turn); or (b) the trajectory nominally points at the stated action but executes it so weakly that the action effectively does not happen — no meaningful response and no building commitment by the end of the horizon (e.g., a stated strong brake with speed essentially unchanged).
- **Score 1: Direct contradiction.** The trajectory persistently does the opposite of an explicit assertion (e.g., the CoT says to brake or hold speed but the trajectory strongly and persistently accelerates, or vice versa; the CoT asserts a committed maneuver in one direction but the trajectory makes a sustained committed maneuver the opposite way). Judge lateral direction by sustained lane-relative offset change (View 2) and committed motion, not by accumulated ego-frame drift, which road curvature can explain.

Example:
1)
  - CoT: "Keep lane to continue driving since no critical agent needs attention."
  - Trajectory: View 2 'delta_offset_m' stays near 0 and 'accel' is near 0, even though View 1 'y' drifts a few meters with the road's curve.
  - Justification: "The lane-relative offset stays centered at steady speed — the ego-frame drift is road curvature, and the trajectory holds the lane exactly as stated."
  - Score: 5
2)
  - CoT: "Slow down and come to a stop for the stop sign at the upcoming intersection."
  - Trajectory: Speed falls steadily (e.g., 11 to 5 m/s) but does not reach 0 within the horizon; View 2 'offset_m' stays within a fraction of a meter of the lane center.
  - Justification: "The trajectory begins the stated stop response with sustained deceleration while tracking the lane; the stop is not completed within the short horizon, so the action still aligns."
  - Score: 4
3)
  - CoT: "Change lanes to the left to pass the slower truck ahead."
  - Trajectory: View 2 'delta_offset_m' stays near 0 for most of the horizon, then 'lat_vel' turns positive and 'delta_offset_m' reaches roughly 1m left by the end.
  - Justification: "The stated lane change is only just initiated within the short horizon, but the lane-relative offset is clearly building leftward by the end — the same maneuver class as stated, partially executed with real commitment."
  - Score: 3
4)
  - CoT: "Nudge slightly right within the lane to give the cyclist more clearance."
  - Trajectory: A committed rightward move with sustained negative 'lat_vel' in both views; View 2 'delta_offset_m' settles near -3.5m (a full lane width).
  - Justification: "The CoT states a small in-lane adjustment, but the lane-relative offset shifts a full lane width — a different maneuver class even though the direction matches."
  - Score: 2
5)
  - CoT: "Brake hard to avoid the queue of stopped traffic ahead."
  - Trajectory: 'accel' hovers near 0 throughout and speed drifts from 13 to 12 m/s, with no building deceleration by the end of the horizon.
  - Justification: "The CoT asserts strong braking, but the trajectory shows no meaningful braking response and no sign of one building — the stated action is under-executed to the point of not happening."
  - Score: 2
6)
  - CoT: "Accelerate to merge onto the highway ahead of the approaching traffic."
  - Trajectory: Speed drops steadily (e.g., 15 to 4 m/s) with persistently negative 'accel' in both views.
  - Justification: "The CoT explicitly asserts acceleration, but the trajectory brakes persistently — the opposite longitudinal response."
  - Score: 1

=== OUTPUT FORMAT ===

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "cot_output_alignment": {{
    "justification": "<one sentence linking the text to specific trajectory numbers>",
    "score": <1-5>
  }}
}}"""
    return prompt
