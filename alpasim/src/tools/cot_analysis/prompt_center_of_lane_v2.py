# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prompt construction for CoT consistency analysis (lane-center v2).

Pairs with the route-consistent lane reference (lane_reference="map_graph"):
the trajectory is projected into a single continuous Frenet frame built by
walking the lane successor graph from the ego's current lane, so
s/offset/accel cannot jump from the reference snapping to a different lane
mid-horizon.
Single-dimension rubric:
  CoT-Output Alignment (1-5)
"""


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the CoT consistency evaluation prompt.

    Args:
        cot_text: The chain-of-thought reasoning text from the driver.
        traj_features: Dict from compute_trajectory_features().

    Returns:
        Formatted prompt string for Gemini.
    """
    markdown_kv = traj_features.get("markdown_kv", "No trajectory data available.")
    stats = traj_features.get("summary_stats", {})
    reference_frame = stats.get("reference_frame", "ego_rig")

    if reference_frame == "lane_center":
        stats_text = (
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
        trajectory_title = (
            "Predicted Trajectory (route-consistent Frenet frame: "
            "one continuous lane-center reference):"
        )
        coordinate_frame = """=== COORDINATE FRAME (ROUTE-CONSISTENT LANE FRENET) ===
The reference line is a single continuous path built by walking the lane successor
graph from the ego's current lane; the whole trajectory is projected into this one
Frenet frame. The reference cannot switch lanes mid-horizon, so changes in offset_m
reflect real ego motion relative to its route, not reference-line artifacts.
- s_lane: longitudinal progress along the reference path (meters).
- offset_m: signed lateral offset from the reference path. Positive = left of it, negative = right.
- delta_offset_m: offset_m relative to t=0. Positive means moved left relative to starting lane position.
- lat_vel: lateral velocity with respect to the reference path (positive left, negative right).
- speed: ground speed of the ego (frame-independent, from the raw trajectory).
- accel: rate of change of ground speed (positive speed-up, negative slow-down).

Important: For curved roads, do NOT judge lateral intent using global/world turning direction.
Use offset_m and delta_offset_m as the primary signals for lane-relative intent execution.
A sustained delta_offset_m of roughly a lane width (~3-4m) indicates a lane change off
the reference route; a sustained fraction of a lane width indicates an in-lane shift."""
    else:
        stats_text = (
            f"Duration: {stats.get('duration_s', '?')}s, "
            f"Path length: {stats.get('total_path_length_m', '?')}m, "
            f"Final position: ({stats.get('final_longitudinal_m', '?')}m forward, "
            f"{stats.get('final_lateral_m', '?')}m lateral), "
            f"Max lateral deviation: {stats.get('max_lateral_m', '?')}m / "
            f"{stats.get('min_lateral_m', '?')}m, "
            f"Speed: {stats.get('mean_speed_ms', '?')} avg / "
            f"{stats.get('max_speed_ms', '?')} max m/s"
        )
        trajectory_title = (
            "Predicted Trajectory (ego-frame fallback: +X forward, +Y left):"
        )
        coordinate_frame = """=== COORDINATE FRAME (EGO FALLBACK) ===
- X axis: forward (longitudinal). Positive X = vehicle moves forward.
- Y axis: lateral. Positive Y = leftward from vehicle center. Negative Y = rightward.
- lat_vel: lateral velocity. Positive = moving left. Negative = moving right.
- accel: longitudinal acceleration. Positive = speeding up. Negative = braking."""

    prompt = f"""You are an expert evaluator for autonomous vehicle reasoning systems.
You are given the Chain-of-Thought (CoT) reasoning produced by a driving model, along with
    the raw numerical trajectory it actually produced.

=== INPUT ===

**Chain-of-Thought Reasoning:**
"{cot_text}"

**{trajectory_title}**
{markdown_kv}

**Summary:** {stats_text}

{coordinate_frame}

=== EVALUATION CRITERIA ===
Provide a holistic score (1-5) for CoT-Output Alignment. Ensure to provide justification for score deductions.

**CoT-Output Alignment**
Does the executed trajectory match the ACTION stated in the CoT? Judge only the action.

For lateral intent, use `offset_m` and `delta_offset_m` (lane-relative) as the primary
signals; fall back to `y` only when no lane reference is available. On curved roads a
world-frame turn direction can be misleading, so prioritize lane-relative offset change
over global heading. For longitudinal intent, use `accel` and speed.

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
