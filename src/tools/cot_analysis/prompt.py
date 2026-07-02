# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prompt construction for CoT consistency analysis.

Builds the multimodal evaluation sent to LLM judge.
"""


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the CoT consistency evaluation prompt.

    Args:
        cot_text: The chain-of-thought reasoning text from the driver.
        traj_features: Dict from compute_trajectory_features().

    Returns:
        Formatted prompt string for LLM judge.
    """
    markdown_kv = traj_features.get("markdown_kv", "No trajectory data available.")
    stats = traj_features.get("summary_stats", {})

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
- lat_vel: lateral velocity. Positive = moving left. Negative = moving right.
- accel: longitudinal acceleration. Positive = speeding up. Negative = braking.

=== EVALUATION CRITERIA ===
Provide a holistic score (1-5) for CoT-Output Alignment. Ensure to provide justification for score deductions.

**CoT-Output Alignment**
Does the executed trajectory match the ACTION stated in the CoT? Judge only the action.

Rubric:
- **Score 5: Clear alignment.** The trajectory clearly executes the stated action (correct direction and rough kinematic magnitude).
- **Score 4: High alignment.** The trajectory executes the stated action with minor discrepancies in magnitude or timing.
- **Score 3: Plausible alignment.** The stated maneuver is only initiated or partially executed but shows clear, building commitment (e.g., deceleration or lateral velocity ramping up by the end of the horizon), or alignment is ambiguous due to the short horizon or possible road curvature — and the trajectory is the same maneuver class as stated.
- **Score 2: Different maneuver class, or under-execution.** Either (a) the trajectory performs a clearly different category of maneuver than stated, on the axis the CoT asserts (e.g., the CoT states slowing in lane but the trajectory instead makes a committed lane change at unchanged speed; the CoT states a small in-lane adjustment but the trajectory executes a full lane change or turn); or (b) the trajectory nominally points at the stated action but executes it so weakly that the action effectively does not happen — no meaningful response and no building commitment by the end of the horizon (e.g., a stated strong brake with speed essentially unchanged).
- **Score 1: Direct contradiction.** The trajectory persistently does the opposite of an explicit assertion (e.g., the CoT says to brake or hold speed but the trajectory strongly and persistently accelerates, or vice versa; the CoT asserts a committed maneuver in one direction but the trajectory makes a sustained committed maneuver the opposite way). Judge lateral direction by sustained lateral velocity and committed motion, not by accumulated drift, which road curvature can explain.

Example:
1)
  - CoT: "Keep lane to continue driving since no critical agent needs attention."
  - Trajectory: 'y' stays near 0, 'accel' is near 0.
  - Justification: "The trajectory holds the lane at steady speed, exactly the stated action."
  - Score: 5
2)
  - CoT: "Slow down and come to a stop for the stop sign at the upcoming intersection."
  - Trajectory: Speed falls steadily (e.g., 11 to 5 m/s) but does not reach 0 within the horizon; 'y' drifts smoothly rightward by a few meters.
  - Justification: "The trajectory begins the stated stop response with sustained deceleration; the stop is not completed within the short horizon and the smooth lateral drift is consistent with road curvature, so the action still aligns."
  - Score: 4
3)
  - CoT: "Change lanes to the left to pass the slower truck ahead."
  - Trajectory: 'y' stays near 0 for most of the horizon, then 'lat_vel' turns positive and 'y' reaches roughly 1m by the end.
  - Justification: "The stated lane change is only just initiated within the short horizon, but the lateral velocity is clearly building by the end — the same maneuver class as stated, partially executed with real commitment."
  - Score: 3
4)
  - CoT: "Nudge slightly right within the lane to give the cyclist more clearance."
  - Trajectory: A committed rightward move of a full lane width with sustained negative 'lat_vel', settling in the adjacent lane.
  - Justification: "The CoT states a small in-lane adjustment, but the trajectory executes a full lane change — a different maneuver class even though the direction matches."
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
