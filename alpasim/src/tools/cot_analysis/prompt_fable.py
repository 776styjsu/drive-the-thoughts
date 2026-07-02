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
Does the executed trajectory match the ACTION stated in the CoT? Judge only the action; do NOT
judge whether the CoT's scene description or premise is correct (that is evaluated separately).

Interpretation rules (the trajectory is a short ~6s horizon in the initial ego frame):
1. **Partial execution counts.** Initiating the stated maneuver within the horizon is alignment:
   "decelerate to stop" does not need to reach 0 m/s; a stated lane change only needs a committed
   lateral move in the stated direction; "resume speed" only needs initial acceleration.
2. **Road curvature is not a lateral maneuver.** Y is measured in the initial ego frame, so on a
   curved road Y drifts large (even tens of meters) and smoothly while the vehicle stays in its
   lane. A smooth, sustained lateral drift at steady speed is usually road curvature — do not mark
   "keep lane", "nudge", or a stated lane-change direction as contradicted based only on large
   smooth Y drift. Treat lateral sign/magnitude as weak evidence unless the lateral motion is
   abrupt relative to the overall path.
3. **Unmentioned motion is not misalignment.** Only contradictions of what the CoT explicitly
   asserts count. "Keep lane" with unmentioned braking still aligns (the lateral claim holds).
4. **"Keep distance" / "follow the lead vehicle"** means smooth in-lane following; modest
   acceleration or deceleration are both compatible (the gap depends on the unseen lead vehicle).
5. **"Wait" / "maintain lane and wait for a gap"** IS contradicted if the trajectory initiates the
   deferred maneuver within the horizon.

Rubric:
- **Score 5: Clear alignment.** The trajectory clearly executes the stated action (correct direction and rough kinematic magnitude).
- **Score 4: High alignment.** The trajectory executes the stated action with minor discrepancies in magnitude or timing.
- **Score 3: Plausible alignment.** The stated maneuver is only initiated or partially executed, or alignment is ambiguous due to the short horizon or possible road curvature — but the trajectory is the same maneuver class as stated.
- **Score 2: Different maneuver class.** The trajectory performs a clearly different maneuver than stated (e.g., CoT says "yield" but trajectory lane-changes instead of slowing in lane; CoT says "lane change" but trajectory makes a full turn; CoT says "split toward the exit" but trajectory plainly lane-follows past it; CoT says "wait for a gap" but trajectory merges immediately).
- **Score 1: Direct contradiction.** The trajectory directly contradicts an explicit assertion (e.g., CoT says "brake/keep speed" but trajectory strongly and persistently accelerates; CoT says turn left at a fork/intersection but trajectory abruptly turns right; CoT says "maintain lane and wait" but trajectory makes an immediate committed lane change).

Example:
1)
  - CoT: "Keep lane to continue driving since no critical agent needs attention."
  - Trajectory: 'y' stays near 0, 'accel' is near 0.
  - Justification: "The trajectory holds the lane at steady speed, exactly the stated action."
  - Score: 5
2)
  - CoT: "Decelerate to stop for the red traffic light ahead."
  - Trajectory: Speed falls steadily (e.g., 14 to 9 m/s) but does not reach 0 within the horizon; 'y' drifts smoothly leftward by several meters.
  - Justification: "The trajectory begins the stated stop response with sustained deceleration; the stop is not completed within the short horizon and the smooth lateral drift is consistent with road curvature, so the action still aligns."
  - Score: 4
3)
  - CoT: "Yield to the cut-in vehicle merging into our lane ahead."
  - Trajectory: Little deceleration; instead a committed lateral move into the adjacent lane.
  - Justification: "The CoT states yielding in lane, but the trajectory responds with a lane change — a different maneuver class than the stated action."
  - Score: 2
4)
  - CoT: "Maintain lane and wait for the overtaking car to clear before moving left."
  - Trajectory: After ~1.5s, an abrupt committed left lateral move with rising lateral velocity.
  - Justification: "The CoT asserts holding the lane and waiting, but the trajectory initiates an immediate committed left lane change within the horizon, directly contradicting the stated waiting behavior."
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
