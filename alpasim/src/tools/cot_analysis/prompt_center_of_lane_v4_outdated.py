# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prompt construction for CoT consistency analysis (lane-center v4).

Pairs with reference_frame="dual": the predicted trajectory is presented in
both the route-consistent lane Frenet frame (s_lane / offset_m, built by
walking the lane successor graph so the reference cannot switch lanes
mid-horizon) AND the absolute ego/rig frame (x / y / lat_vel / accel). The
lane view carries lane-relative intent that road curvature cannot explain
away; the ego view carries absolute geometry (turns, heading change,
displacement). Degrades to a single view when only one frame is available.

Single-dimension rubric:
  CoT-Output Alignment (1-5)
"""


def _lane_stats_text(stats: dict) -> str:
    return (
        f"Reference: {stats.get('lane_reference', 'lane_center')}, "
        f"Duration: {stats.get('duration_s', '?')}s, "
        f"Lane progress: {stats.get('lane_path_length_m', '?')}m, "
        f"Offset start/end: {stats.get('initial_offset_m', '?')}m -> "
        f"{stats.get('final_offset_m', '?')}m, "
        f"Delta offset: {stats.get('delta_offset_m', '?')}m, "
        f"Offset range: [{stats.get('min_offset_m', '?')}, "
        f"{stats.get('max_offset_m', '?')}] m, "
        f"Lanes in reference path: {stats.get('lane_segment_count_used', '?')}, "
        f"Samples: {stats.get('lane_sample_count', '?')} "
        f"(stride={stats.get('lane_sample_stride', '?')})"
    )


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


LANE_FRAME_NOTES = """=== COORDINATE FRAME (LANE-RELATIVE ROUTE FRAME) ===
We use one fixed lane-center route as the reference for the entire predicted
trajectory. Every trajectory point is measured relative to this same route.
Therefore, changes in `offset_m` should usually be interpreted as the ego vehicle
moving left or right relative to its route, not as the reference suddenly switching
to another lane.
- s_lane: longitudinal progress along the reference path (meters).
- offset_m: signed lateral offset from the reference path. Positive = left of it, negative = right.
- delta_offset_m: offset_m relative to t=0. Positive means moved left relative to starting lane position.
- lat_vel: lateral velocity with respect to the reference path (positive left, negative right).
- speed: ground speed of the ego (frame-independent, from the raw trajectory).
- accel: rate of change of ground speed (positive speed-up, negative slow-down).

Important: Interpret lateral actions by their type.

For lane-relative actions such as lane changes, nudges, drifting, or lane keeping,
use offset_m, delta_offset_m, and lat_vel as the primary signals. A sustained
delta_offset_m of roughly a lane width (~3-4m) suggests a lane change relative
to the reference route; a sustained smaller offset change suggests an in-lane lateral
shift.

For turn actions such as turning left or right at an intersection, do not expect
offset_m to grow. If the reference route itself curves through the turn, a correctly
executed turn may keep offset_m near zero. Judge turns using the trajectory's change
in direction, path geometry, heading/curvature if available, and whether the motion
follows a left- or right-turning route. In this case, offset_m is mainly useful
for checking whether the ego stays centered while turning, not for detecting the
turn itself."""

EGO_FRAME_NOTES = """=== COORDINATE FRAME (ABSOLUTE EGO FRAME) ===
- x: forward (longitudinal) position. Positive x = vehicle moves forward.
- y: lateral position. Positive y = leftward from the vehicle center at t=0. Negative y = rightward.
- lat_vel: lateral velocity. Positive = moving left. Negative = moving right.
- accel: longitudinal acceleration. Positive = speeding up. Negative = braking.
Caution: on curved roads, y accumulates from road curvature alone — a large |y| at the
end of the horizon does NOT by itself mean a lateral maneuver. Use the lane-relative
offset_m / delta_offset_m as the primary lateral signal and read the ego frame for
absolute geometry (turn shape, heading change, displacement)."""


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the CoT consistency evaluation prompt.

    Args:
        cot_text: The chain-of-thought reasoning text from the driver.
        traj_features: Dict from compute_trajectory_features(), ideally with
            reference_frame="dual" so both the lane-relative and ego-frame
            views are present. Degrades to a single view otherwise.

    Returns:
        Formatted prompt string for LLM judge.
    """
    # Dual-aware resolution: prefer the nested ego/lane views produced by
    # reference_frame="dual", and fall back to a single-frame features dict.
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
    coord_sections = []
    if lane_ok:
        sections.append(
            "**Predicted Trajectory (route-consistent Frenet frame: one "
            "continuous lane-center reference):**\n"
            f"{lane.get('markdown_kv', 'No trajectory data available.')}\n\n"
            f"**Lane-frame summary:** {_lane_stats_text(lane.get('summary_stats', {}))}"
        )
        coord_sections.append(LANE_FRAME_NOTES)
    if ego is not None:
        sections.append(
            "**Predicted Trajectory (absolute ego frame: +X forward, +Y left):**\n"
            f"{ego.get('markdown_kv', 'No trajectory data available.')}\n\n"
            f"**Ego-frame summary:** {_ego_stats_text(ego.get('summary_stats', {}))}"
        )
        coord_sections.append(EGO_FRAME_NOTES)
    if not sections:
        sections.append("No trajectory data available.")

    trajectory_block = "\n\n".join(sections)
    coordinate_frame = "\n\n".join(coord_sections)

    prompt = f"""You are an expert evaluator for autonomous vehicle reasoning systems.
You are given the Chain-of-Thought (CoT) reasoning produced by a driving model, along with
    the raw numerical trajectory it actually produced.

=== INPUT ===

**Chain-of-Thought Reasoning:**
"{cot_text}"

{trajectory_block}

{coordinate_frame}

=== EVALUATION CRITERIA ===
Provide a holistic score (1-5) for CoT-Output Alignment. Ensure to provide justification for score deductions.

**CoT-Output Alignment**
Does the executed trajectory match the ACTION stated in the CoT? Judge only the action.

For lateral intent, use `offset_m` and `delta_offset_m` (lane-relative) as the primary
signals. For longitudinal intent, use `accel` and speed.

Rubric:
- **Score 5: Clear alignment.** The trajectory clearly executes the stated action (correct direction and rough kinematic magnitude).
- **Score 4: High alignment.** The trajectory executes the stated action with minor discrepancies in magnitude or timing.
- **Score 3: Plausible alignment.** The stated maneuver is only initiated or partially executed but shows clear, building commitment (e.g., deceleration or lane-relative offset ramping up by the end of the horizon), or alignment is ambiguous due to the short horizon or possible road curvature — and the trajectory is the same maneuver class as stated.
- **Score 2: Different maneuver class, or under-execution.** Either (a) the trajectory performs a clearly different category of maneuver than stated, on the axis the CoT asserts (e.g., the CoT states slowing in lane but the trajectory instead makes a committed lane change at unchanged speed; the CoT states a small in-lane adjustment but the trajectory executes a full lane change or turn); or (b) the trajectory nominally points at the stated action but executes it so weakly that the action effectively does not happen — no meaningful response and no building commitment by the end of the horizon (e.g., a stated strong brake with speed essentially unchanged).
- **Score 1: Direct contradiction.** The trajectory persistently does the opposite of an explicit assertion (e.g., the CoT says to brake or hold speed but the trajectory strongly and persistently accelerates, or vice versa; the CoT asserts a committed maneuver in one direction but the trajectory makes a sustained committed maneuver the opposite way). Judge lateral direction by sustained lane-relative offset change and committed motion, not by accumulated world-frame drift, which road curvature can explain.

Example:
1)
  - CoT: "Keep lane to continue driving since no critical agent needs attention."
  - Trajectory: 'delta_offset_m' stays near 0, 'accel' is near 0.
  - Justification: "The trajectory holds the lane at steady speed, exactly the stated action."
  - Score: 5
2)
  - CoT: "Slow down and come to a stop for the stop sign at the upcoming intersection."
  - Trajectory: Speed falls steadily (e.g., 11 to 5 m/s) but does not reach 0 within the horizon; 'offset_m' drifts smoothly by under a meter while tracking the lane.
  - Justification: "The trajectory begins the stated stop response with sustained deceleration; the stop is not completed within the short horizon and the small lane-relative offset is consistent with lane tracking, so the action still aligns."
  - Score: 4
3)
  - CoT: "Change lanes to the left to pass the slower truck ahead."
  - Trajectory: 'delta_offset_m' stays near 0 for most of the horizon, then 'lat_vel' turns positive and 'delta_offset_m' reaches roughly 1m left by the end.
  - Justification: "The stated lane change is only just initiated within the short horizon, but the lane-relative offset is clearly building leftward by the end — the same maneuver class as stated, partially executed with real commitment."
  - Score: 3
4)
  - CoT: "Nudge slightly right within the lane to give the cyclist more clearance."
  - Trajectory: A committed rightward move of a full lane width with sustained negative 'lat_vel', 'delta_offset_m' settling at the next lane center.
  - Justification: "The CoT states a small in-lane adjustment, but the lane-relative offset shifts a full lane width — a different maneuver class even though the direction matches."
  - Score: 2
5)
  - CoT: "Brake hard to avoid the queue of stopped traffic ahead."
  - Trajectory: 'accel' hovers near 0 throughout and speed drifts from 13 to 12 m/s, with no building deceleration by the end of the horizon.
  - Justification: "The CoT asserts strong braking, but the trajectory shows no meaningful braking response and no sign of one building — the stated action is under-executed to the point of not happening."
  - Score: 2
6)
  - CoT: "Accelerate to merge onto the highway ahead of the approaching traffic."
  - Trajectory: Speed drops steadily (e.g., 15 to 4 m/s) with persistently negative 'accel'.
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
