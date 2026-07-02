# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Ego-frame v5 prompt with explicit trajectory-heading evidence.

This keeps the scoring behavior of the default :mod:`prompt` while making turns
observable through the trajectory's net and cumulative heading change. Unlike
ego-frame lateral displacement, heading change directly measures how much the
vehicle rotated and is not inflated by distance traveled along a curved path.
"""


def _trajectory_table_with_heading(traj_features: dict) -> str:
    """Render the ego-frame trajectory table with heading change included."""
    rows = traj_features.get("table_rows")
    if not isinstance(rows, list) or not rows:
        return traj_features.get("markdown_kv", "No trajectory data available.")

    lines = []
    for row in rows:
        lines.append(f"### t={row.get('t')}s")
        lines.append(f"- x: {row.get('x')}m")
        lines.append(f"- y: {row.get('y')}m")
        lines.append(
            f"- heading_deg: {row.get('heading_deg')} (cumulative vs t=0, + = left)"
        )
        lines.append(f"- speed: {row.get('speed')} m/s")
        lines.append(f"- lat_vel: {row.get('lat_vel')} m/s")
        lines.append(f"- accel: {row.get('accel')} m/s²")
    return "\n".join(lines)


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the v5 CoT consistency prompt with heading evidence.

    Args:
        cot_text: The chain-of-thought reasoning text from the driver.
        traj_features: Ego-frame features from compute_trajectory_features().

    Returns:
        Formatted prompt string for the LLM judge.
    """
    markdown_kv = _trajectory_table_with_heading(traj_features)
    stats = traj_features.get("summary_stats", {})

    stats_text = (
        f"Duration: {stats.get('duration_s', '?')}s, "
        f"Path length: {stats.get('total_path_length_m', '?')}m, "
        f"Final position: ({stats.get('final_longitudinal_m', '?')}m forward, "
        f"{stats.get('final_lateral_m', '?')}m lateral), "
        f"Max lateral deviation: {stats.get('max_lateral_m', '?')}m / "
        f"{stats.get('min_lateral_m', '?')}m, "
        "Heading change: "
        f"{stats.get('traj_net_heading_change_deg', '?')} deg net / "
        f"{stats.get('traj_total_heading_change_deg', '?')} deg total, "
        f"Speed: {stats.get('mean_speed_ms', '?')} avg / "
        f"{stats.get('max_speed_ms', '?')} max m/s"
    )

    prompt = f"""You are an expert evaluator for autonomous vehicle reasoning systems.
You are given the Chain-of-Thought (CoT) reasoning produced by a driving model, along with
the raw numerical trajectory it actually produced.

=== INPUT ===

**Chain-of-Thought Reasoning:**
"{cot_text}"

**Predicted Trajectory (ego-frame: +X = forward, +Y = left, sampled at ~1Hz):**
{markdown_kv}

**Summary:** {stats_text}

=== COORDINATE FRAME ===
- X axis: forward (longitudinal). Positive X = vehicle moves forward.
- Y axis: lateral. Positive Y = leftward from vehicle center. Negative Y = rightward.
- heading_deg: trajectory heading (direction of travel), cumulative relative to t=0.
  Positive = rotated left/counterclockwise; negative = rotated right/clockwise.
- The net heading change is signed start-to-end rotation. Total heading change is cumulative
  absolute rotation, so it remains large when the vehicle turns out and later turns back.
- lat_vel: lateral velocity. Positive = moving left. Negative = moving right.
- accel: longitudinal acceleration. Positive = speeding up. Negative = braking.
- On curved roads, Y accumulates from road curvature alone. Use heading_deg to judge actual
  vehicle rotation and sustained lateral velocity to distinguish a turn from a brief drift.

=== EVALUATION CRITERIA ===
Provide a holistic score (1-5) for CoT-Output Alignment. Ensure to provide justification for score deductions.

**CoT-Output Alignment**
Does the executed trajectory match the ACTION stated in the CoT? Judge only the action.

Pick the signal that matches the stated action:
- Longitudinal intent (slow/stop/accelerate/hold speed): use `accel` and speed.
- Lateral intent (keep lane/nudge/change lane): use sustained `lat_vel`, Y displacement,
  and whether heading swings out and returns toward 0.
- Turn intent (turn left/right/go straight): use net `heading_deg`. A completed intersection
  turn normally produces tens of degrees of net heading change in the stated direction.

Rubric:
- **Score 5: Clear alignment.** The trajectory clearly executes the stated action (correct direction and rough kinematic magnitude). A stated turn shows a large net heading change in the stated direction.
- **Score 4: High alignment.** The trajectory executes the stated action with minor discrepancies in magnitude or timing.
- **Score 3: Plausible alignment.** The stated maneuver is only initiated or partially executed but shows clear, building commitment (e.g., deceleration, sustained lateral velocity, or heading change ramping up by the end of the horizon), or alignment is ambiguous due to the short horizon or possible road curvature — and the trajectory is the same maneuver class as stated.
- **Score 2: Different maneuver class, or under-execution.** Either (a) the trajectory performs a clearly different category of maneuver than stated, on the axis the CoT asserts (e.g., the CoT states slowing in lane but the trajectory instead makes a committed lane change at unchanged speed; the CoT states a small in-lane adjustment but the trajectory executes a full lane change or turn); or (b) the trajectory nominally points at the stated action but executes it so weakly that the action effectively does not happen — no meaningful response and no building commitment by the end of the horizon (e.g., a stated strong brake with speed essentially unchanged, or a stated turn with only a few degrees of net heading change).
- **Score 1: Direct contradiction.** The trajectory persistently does the opposite of an explicit assertion (e.g., the CoT says to brake or hold speed but the trajectory strongly and persistently accelerates, or vice versa; the CoT asserts a committed maneuver in one direction but the trajectory has sustained heading change or lateral motion in the opposite direction). Judge direction by sustained lateral velocity and heading change, not accumulated Y alone, which road curvature can explain.

Example:
1)
  - CoT: "Keep lane to continue driving since no critical agent needs attention."
  - Trajectory: 'y' stays near 0, net 'heading_deg' stays near 0, and 'accel' is near 0.
  - Justification: "The trajectory holds course at steady speed with no net rotation, exactly the stated action."
  - Score: 5
2)
  - CoT: "Turn left at the intersection to follow the route."
  - Trajectory: 'heading_deg' rises steadily to about +85 deg by the end while speed remains controlled.
  - Justification: "The trajectory rotates about +85 degrees, clearly executing the stated left turn."
  - Score: 5
3)
  - CoT: "Turn right at the upcoming intersection."
  - Trajectory: net 'heading_deg' stays within a few degrees of 0 across the horizon and speed is steady.
  - Justification: "The trajectory heading barely changes, so the stated right turn is not meaningfully executed."
  - Score: 2
4)
  - CoT: "Change lanes to the left to pass the slower truck ahead."
  - Trajectory: 'y' stays near 0 for most of the horizon, then 'lat_vel' turns positive, 'y' reaches roughly 1m, and 'heading_deg' begins a small leftward swing by the end.
  - Justification: "The stated lane change is only just initiated, but positive lateral velocity, displacement, and heading show clear building commitment to the left."
  - Score: 3
5)
  - CoT: "Nudge slightly right within the lane to give the cyclist more clearance."
  - Trajectory: A committed rightward move of a full lane width with sustained negative 'lat_vel', while 'heading_deg' swings right and returns toward 0.
  - Justification: "The CoT states a small in-lane adjustment, but the trajectory executes a full lane change — a different maneuver class even though the direction matches."
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
