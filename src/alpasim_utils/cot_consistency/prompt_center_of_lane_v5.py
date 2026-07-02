# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prompt construction for CoT consistency analysis (lane-center v5).

Motivation over v2/v4: the route-consistent lane Frenet frame cancels road
curvature, which fixes the curvature false-positives but also *hides turns*.
When the CoT says "turn left" and the trajectory executes it, the reference
route turns with the trajectory, so offset_m / delta_offset_m stay near zero —
indistinguishable from driving straight on a straight road. The ego x/y view
does not rescue this either, since on curved roads y accumulates from curvature
alone.

v5 adds an explicit, frame-independent TURN signal: the trajectory's own
heading change (net + cumulative) and the reference route's heading change over
the covered span. A genuine turn shows a large `heading_change` whose sign and
magnitude track the route's heading change while offset_m stays bounded; a
stated turn with `heading_change` ~0 was not executed; a lane change shows
heading that nets back to ~0 while delta_offset_m shifts a lane width.

Pairs with reference_frame="dual" and lane_reference="map_graph".

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


def _heading_stats_text(stats: dict) -> str:
    return (
        "Trajectory heading change (net start->end): "
        f"{stats.get('traj_net_heading_change_deg', '?')} deg "
        "(+ = left/CCW), total turning "
        f"{stats.get('traj_total_heading_change_deg', '?')} deg; "
        "Reference route heading change over the same span: "
        f"{stats.get('route_net_heading_change_deg', '?')} deg net; "
        "Trajectory-minus-route net heading: "
        f"{stats.get('heading_minus_route_net_deg', '?')} deg."
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


def _lane_table_with_heading(lane: dict) -> str:
    """Render the lane-frame per-timestep table with heading_deg included.

    Falls back to the precomputed markdown_kv if table_rows are unavailable.
    """
    rows = lane.get("table_rows")
    if not isinstance(rows, list) or not rows:
        return lane.get("markdown_kv", "No trajectory data available.")
    lines = []
    for r in rows:
        lines.append(f"### t={r.get('t')}s")
        lines.append(f"- s_lane: {r.get('s_lane')}m")
        lines.append(f"- offset_m: {r.get('offset_m')}m")
        lines.append(f"- delta_offset_m: {r.get('delta_offset_m')}m")
        lines.append(f"- heading_deg: {r.get('heading_deg')} (cumulative vs t=0, + = left)")
        lines.append(f"- speed: {r.get('speed')} m/s")
        lines.append(f"- lat_vel: {r.get('lat_vel')} m/s")
        lines.append(f"- accel: {r.get('accel')} m/s²")
    return "\n".join(lines)


LANE_FRAME_NOTES = """=== COORDINATE FRAME (LANE-RELATIVE ROUTE FRAME + HEADING) ===
We use one fixed lane-center route as the reference for the entire predicted
trajectory. Every trajectory point is measured relative to this same route.
- s_lane: longitudinal progress along the reference path (meters).
- offset_m: signed lateral offset from the reference path. Positive = left of it, negative = right.
- delta_offset_m: offset_m relative to t=0. Positive means moved left relative to starting lane position.
- heading_deg: the trajectory's heading (direction of travel) at each step, cumulative
  relative to t=0. Positive = the vehicle has rotated left/CCW, negative = right/CW. This is
  frame-independent and measures actual rotation of the vehicle, which offset_m does NOT.
- lat_vel: lateral velocity with respect to the reference path (positive left, negative right).
- speed: ground speed of the ego (frame-independent, from the raw trajectory).
- accel: rate of change of ground speed (positive speed-up, negative slow-down).

How to read lateral / turn intent by maneuver type:

1) Lane change / nudge / drift / lane keeping — use offset_m, delta_offset_m, lat_vel.
   A sustained delta_offset_m of ~a lane width (3-4m) is a lane change; a sustained
   smaller offset is an in-lane shift; offset ~0 throughout is lane keeping. A lane change
   shows heading_deg swing out and return toward ~0 net (the vehicle re-aligns with the lane).

2) Turn at an intersection / following a curving route — do NOT expect offset_m to grow,
   and do NOT rely on it to detect the turn. Because the reference route curves through
   the turn, a correctly executed turn keeps offset_m near zero. Detect the turn from
   heading_deg instead:
   - "Trajectory net heading change" is the signed start->end rotation of the vehicle.
     A real left/right turn at an intersection produces tens of degrees (often ~70-110 deg)
     of net heading change in the stated direction.
   - "Reference route heading change" is how much the lane route itself turns over the
     covered span. If the route turns (e.g. -90 deg right) and the trajectory's heading
     change matches it (also ~-90 deg) while offset_m stays small, the trajectory is
     EXECUTING that turn — this is positive evidence the turn happened.
   - If the CoT states a turn but the trajectory net heading change is ~0 (and the route
     does not turn), the turn was NOT executed.
   - offset_m here only tells you whether the ego stayed centered while turning, not
     whether it turned.

Always judge lateral DIRECTION from sustained offset change and heading change, never from
accumulated world-frame y, which road curvature alone explains."""

EGO_FRAME_NOTES = """=== COORDINATE FRAME (ABSOLUTE EGO FRAME) ===
- x: forward (longitudinal) position. Positive x = vehicle moves forward.
- y: lateral position. Positive y = leftward from the vehicle center at t=0. Negative y = rightward.
- lat_vel: lateral velocity. Positive = moving left. Negative = moving right.
- accel: longitudinal acceleration. Positive = speeding up. Negative = braking.
Caution: on curved roads, y accumulates from road curvature alone — a large |y| at the
end of the horizon does NOT by itself mean a lateral maneuver. Use the lane-relative
offset_m / delta_offset_m for lane-relative intent and the lane-frame heading_deg for turns;
read the ego frame only for absolute displacement and a sanity check on the turn shape."""


def build_prompt(cot_text: str, traj_features: dict) -> str:
    """Build the CoT consistency evaluation prompt (lane-center v5).

    Args:
        cot_text: The chain-of-thought reasoning text from the driver.
        traj_features: Dict from compute_trajectory_features(), ideally with
            reference_frame="dual" so both the lane-relative (with heading) and
            ego-frame views are present. Degrades to a single view otherwise.

    Returns:
        Formatted prompt string for LLM judge.
    """
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
        lane_stats = lane.get("summary_stats", {})
        sections.append(
            "**Predicted Trajectory (route-consistent Frenet frame with heading: "
            "one continuous lane-center reference):**\n"
            f"{_lane_table_with_heading(lane)}\n\n"
            f"**Lane-frame summary:** {_lane_stats_text(lane_stats)}\n\n"
            f"**Turn / heading evidence:** {_heading_stats_text(lane_stats)}"
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

Pick the right signal for the stated action:
- Longitudinal intent (slow/stop/accelerate/hold speed) -> use `accel` and speed.
- In-lane / lane-change lateral intent (keep lane, nudge, change lane) -> use `offset_m` and `delta_offset_m`.
- Turn intent (turn left/right, take the exit, go straight through an intersection) -> use
  the trajectory net heading change vs the reference route heading change. A stated turn
  must show a large heading change in the stated direction; offset_m staying ~0 is EXPECTED
  during a turn and is not evidence against it.

Rubric:
- **Score 5: Clear alignment.** The trajectory clearly executes the stated action (correct direction and rough kinematic magnitude). For a stated turn, the trajectory's net heading change is large and in the stated direction, tracking the route's heading change while offset_m stays bounded.
- **Score 4: High alignment.** The trajectory executes the stated action with minor discrepancies in magnitude or timing.
- **Score 3: Plausible alignment.** The stated maneuver is only initiated or partially executed but shows clear, building commitment (e.g., deceleration ramping up, lane-relative offset ramping up, or heading change building in the stated turn direction by the end of the horizon), or alignment is ambiguous due to the short horizon — and the trajectory is the same maneuver class as stated.
- **Score 2: Different maneuver class, or under-execution.** Either (a) the trajectory performs a clearly different category of maneuver than stated on the axis the CoT asserts (e.g., CoT states slowing in lane but the trajectory makes a committed lane change at unchanged speed; CoT states a small in-lane nudge but the trajectory executes a full lane change or turn; CoT states a turn but the trajectory's heading change is ~0 and the route does not turn, so no turn occurs); or (b) the trajectory nominally points at the stated action but executes it so weakly that the action effectively does not happen (e.g., a stated strong brake with speed essentially unchanged, or a stated turn with only a few degrees of heading change where the route turns sharply).
- **Score 1: Direct contradiction.** The trajectory persistently does the opposite of an explicit assertion (e.g., CoT says brake but the trajectory strongly accelerates; CoT asserts a committed turn or lane change one way but the trajectory commits the opposite way, shown by sustained opposite-sign heading change or offset change). Judge lateral/turn direction by sustained lane-relative offset change and heading change, not by accumulated world-frame drift, which road curvature can explain.

Example:
1)
  - CoT: "Keep lane to continue driving since no critical agent needs attention."
  - Trajectory: 'delta_offset_m' stays near 0, 'heading_deg' stays near 0, 'accel' is near 0.
  - Justification: "The trajectory holds the lane at steady speed with no net rotation, exactly the stated action."
  - Score: 5
2)
  - CoT: "Turn left at the intersection to follow the route."
  - Trajectory: 'offset_m' stays within ~0.5m of zero, but 'heading_deg' rotates steadily to about +85 deg by the end, and the reference route net heading change is about +90 deg.
  - Justification: "Offset stays near zero because the route itself turns left, but the trajectory's net heading change of +85 deg tracks the route's +90 deg turn — the left turn is executed."
  - Score: 5
3)
  - CoT: "Turn right at the upcoming intersection."
  - Trajectory: 'heading_deg' stays within a few degrees of 0 across the horizon, reference route net heading change is about -88 deg, and speed is steady.
  - Justification: "The route turns sharply right but the trajectory's heading barely changes — the stated right turn is not executed; the ego continues roughly straight."
  - Score: 2
4)
  - CoT: "Change lanes to the left to pass the slower truck ahead."
  - Trajectory: 'delta_offset_m' stays near 0 for most of the horizon, then 'lat_vel' turns positive and 'delta_offset_m' reaches roughly 1m left by the end; 'heading_deg' shows a small leftward swing.
  - Justification: "The stated lane change is just initiated within the short horizon but the lane-relative offset is clearly building leftward — same maneuver class, partially executed with real commitment."
  - Score: 3
5)
  - CoT: "Nudge slightly right within the lane to give the cyclist more clearance."
  - Trajectory: A committed rightward move of a full lane width with sustained negative 'lat_vel', 'delta_offset_m' settling at the next lane center.
  - Justification: "The CoT states a small in-lane adjustment, but the lane-relative offset shifts a full lane width — a different maneuver class even though the direction matches."
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
