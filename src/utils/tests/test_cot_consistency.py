# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from alpasim_utils.consistency import match_cot_to_trajectory
from alpasim_utils.cot_meta_actions import parse_cot


def _meta(longitudinal: list[str], lateral: list[str]) -> dict:
    return {
        "per_segment": [
            {
                "longitudinal": long_action,
                "lateral": lat_action,
            }
            for long_action, lat_action in zip(longitudinal, lateral)
        ]
    }


def test_compact_patterns_generalize_across_reasons() -> None:
    keep_lane = parse_cot("Keep lane to stay clear of the construction cones ahead.")
    turn_left = parse_cot(
        "Turn left through the intersection since the left-turn light is green."
    )

    assert [action.value for action in keep_lane.lateral_sequence] == ["go_straight"]
    assert keep_lane.lateral_evidence[0].phrase == "keep lane"
    assert [action.value for action in turn_left.lateral_sequence] == ["steer_left"]
    assert turn_left.lateral_evidence[0].phrase == "turn left"


def test_compact_patterns_prefer_specific_magnitude_phrase() -> None:
    parsed = parse_cot("Brake hard and then turn right.")

    assert [action.value for action in parsed.longitudinal_sequence] == [
        "strong_decelerate"
    ]
    assert parsed.longitudinal_evidence[0].phrase == "brake hard"


def test_compact_patterns_do_not_turn_a_stop_purpose_into_an_action_sequence() -> None:
    keep_distance = parse_cot("Stop to keep distance to the lead vehicle.")
    yield_to_pedestrian = parse_cot("Stop to yield to the pedestrian.")

    assert [action.value for action in keep_distance.longitudinal_sequence] == ["stop"]
    assert [action.value for action in yield_to_pedestrian.longitudinal_sequence] == [
        "stop"
    ]


def test_binary_reward_uses_trajectory_sequence() -> None:
    report = match_cot_to_trajectory(
        "Decelerate, stop, and then accelerate straight through the intersection.",
        _meta(
            [
                "maintain_speed",
                "gentle_decelerate",
                "stop",
                "gentle_accelerate",
            ],
            ["go_straight", "go_straight", "go_straight", "go_straight"],
        ),
    )

    assert report.score == 1.0
    assert report.label == "consistent"
    assert report.longitudinal.cot_sequence == [
        "gentle_decelerate",
        "stop",
        "gentle_accelerate",
    ]
    assert report.longitudinal.trajectory_sequence == [
        "maintain_speed",
        "gentle_decelerate",
        "stop",
        "gentle_accelerate",
    ]


def test_missing_lateral_intent_matches_neutral_lateral_trajectory() -> None:
    report = match_cot_to_trajectory(
        "Stop to keep distance to the lead vehicle since it is stopped ahead.",
        _meta(
            ["maintain_speed", "gentle_decelerate", "stop"],
            ["go_straight", "go_straight", "go_straight"],
        ),
    )

    assert report.score == 1.0
    assert report.lateral.verdict == "no_intent"


def test_silent_axis_imposes_no_requirement_even_when_trajectory_non_neutral() -> None:
    # The CoT states only the lateral turn; the deceleration the turn requires
    # lands on the silent longitudinal axis. Under no-requirement semantics that
    # silent axis is unconstrained, so the clip stays consistent rather than
    # being penalized for an unmentioned (but entailed) deceleration.
    report = match_cot_to_trajectory(
        "Turn right through the green arrow.",
        _meta(["gentle_decelerate"], ["steer_right"]),
    )

    assert report.score == 1.0
    assert report.label == "consistent"
    assert report.longitudinal.verdict == "no_intent"
    assert report.lateral.verdict == "consistent"


def test_binary_reward_zero_for_unparsed_reasoning() -> None:
    report = match_cot_to_trajectory(
        "The scene is complex and requires careful attention.",
        _meta(["maintain_speed"], ["go_straight"]),
    )

    assert report.score == 0.0
    assert report.label == "invalid_parse"


def test_stop_is_its_own_family_and_contradicts_acceleration() -> None:
    # stop is no longer folded into the decel family, but accelerating still
    # opposes a stated stop, so this stays a contradiction (not just a mismatch).
    report = match_cot_to_trajectory(
        "Come to a full stop at the red light.",
        _meta(["gentle_accelerate"], ["go_straight"]),
    )

    assert report.score == 0.0
    assert report.label == "contradictory"
    assert report.longitudinal.verdict == "contradict"


def test_stated_decelerate_not_satisfied_by_a_full_stop() -> None:
    # With stop in its own family, a stated gentle deceleration is no longer
    # covered/allowed by a trajectory that comes to a full stop: the stop is
    # surplus motion outside the stated decel family.
    report = match_cot_to_trajectory(
        "Slow down for the red traffic light.",
        _meta(["gentle_decelerate", "stop"], ["go_straight", "go_straight"]),
    )

    assert report.score == 0.0
    assert report.longitudinal.verdict == "mismatch"


def test_binary_reward_zero_for_contradictory_lateral_behavior() -> None:
    report = match_cot_to_trajectory(
        "Keep lane to continue driving because the lane ahead is clear.",
        _meta(["maintain_speed", "maintain_speed"], ["steer_left", "go_straight"]),
    )

    assert report.score == 0.0
    assert report.label == "inconsistent"
    assert report.lateral.verdict == "mismatch"


# --- exact match mode -------------------------------------------------------


def test_exact_mode_consistent_when_trajectory_has_exact_label() -> None:
    # CoT gentle deceleration is satisfied because the trajectory contains the
    # exact gentle_decelerate label (neutral maintain_speed extras are tolerated).
    report = match_cot_to_trajectory(
        "Slow down gently for the car ahead.",
        _meta(["maintain_speed", "gentle_decelerate"], ["go_straight", "go_straight"]),
        match_mode="exact",
    )

    assert report.score == 1.0
    assert report.label == "consistent"
    assert report.longitudinal.verdict == "consistent"
    # Families are not used in exact mode, so they are not reported.
    assert report.longitudinal.cot_family is None
    assert report.longitudinal.trajectory_family is None


def test_exact_mode_inconsistent_for_same_family_different_magnitude() -> None:
    # gentle vs strong are the same family (consistent under "family" mode) but
    # different exact labels, so exact mode marks this inconsistent.
    family_report = match_cot_to_trajectory(
        "Slow down gently for the car ahead.",
        _meta(["strong_decelerate"], ["go_straight"]),
        match_mode="family",
    )
    assert family_report.score == 1.0

    exact_report = match_cot_to_trajectory(
        "Slow down gently for the car ahead.",
        _meta(["strong_decelerate"], ["go_straight"]),
        match_mode="exact",
    )
    assert exact_report.score == 0.0
    assert exact_report.label == "inconsistent"
    assert exact_report.longitudinal.verdict == "mismatch"


def test_exact_mode_has_no_contradictory_label() -> None:
    # Opposite directions contradict under family mode but are a plain
    # inconsistency (never "contradictory") under exact mode.
    report = match_cot_to_trajectory(
        "Come to a full stop at the red light.",
        _meta(["gentle_accelerate"], ["go_straight"]),
        match_mode="exact",
    )

    assert report.score == 0.0
    assert report.label == "inconsistent"
    assert report.longitudinal.verdict == "mismatch"


def test_exact_mode_keeps_invalid_parse_rule() -> None:
    report = match_cot_to_trajectory(
        "The scene is complex and requires careful attention.",
        _meta(["maintain_speed"], ["go_straight"]),
        match_mode="exact",
    )

    assert report.score == 0.0
    assert report.label == "invalid_parse"


def test_exact_mode_keeps_single_silent_channel_no_requirement() -> None:
    # The CoT states only the lateral turn; the silent longitudinal axis imposes
    # no requirement, and the exact lateral label is present in the trajectory.
    report = match_cot_to_trajectory(
        "Turn right through the green arrow.",
        _meta(["gentle_decelerate"], ["steer_right"]),
        match_mode="exact",
    )

    assert report.score == 1.0
    assert report.label == "consistent"
    assert report.longitudinal.verdict == "no_intent"
    assert report.lateral.verdict == "consistent"
