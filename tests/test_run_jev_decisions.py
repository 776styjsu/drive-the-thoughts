# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json

import pytest

# The OpenRouter runners need httpx, which comes with the llm extra (via openai).
httpx = pytest.importorskip("httpx")

from alpasim_utils.cot_consistency import build_center_of_lane_prompt
from cot_analysis.pipeline import result_is_complete
from cot_analysis.prompt import build_prompt
from tools import run_jev_decisions
from tools.run_jev_decisions import (
    _call,
    _decisions_url,
    evaluation_from_answer,
    jev_request,
)

COT = "['Slow down for the stopped car ahead']"
EGO = {
    "markdown_kv": "### t=0s\n- x: 0.0m\n- accel: -1.2 m/s²",
    "summary_stats": {"duration_s": 5.0, "mean_speed_ms": 8.1},
}
LANE = {
    "summary_stats": {"reference_frame": "lane_center", "lane_reference": "lane_center"},
    "table_rows": [{"t": 0.0, "s_lane": 0.0, "offset_m": 0.1, "speed": 8.0}],
}


def _criteria(request: dict) -> list[dict]:
    return request["questions"]["cot_output_alignment"]["criteria"]


def test_default_prompt_becomes_state_plus_rubric_levels() -> None:
    request = jev_request(build_prompt(COT, EGO), COT, "m")

    assert request["model"] == "m"
    assert request["state"]["chain_of_thought"] == COT
    assert request["state"]["trajectory"].startswith("**Predicted Trajectory")
    assert "- accel: -1.2 m/s²" in request["state"]["trajectory"]
    question = request["questions"]["cot_output_alignment"]
    assert question["type"] == "score"
    assert set(question["instructions"]) == {"question", "context", "coordinate_frames"}
    assert question["instructions"]["coordinate_frames"].startswith("COORDINATE FRAME\n")
    levels = _criteria(request)
    # Lowest score first, with the "Score N:" prefixes dropped.
    assert [level["what"].split(".")[0] for level in levels] == [
        "Direct contradiction",
        "Different maneuver class, or under-execution",
        "Plausible alignment",
        "High alignment",
        "Clear alignment",
    ]
    assert not any("Score" in level["what"] for level in levels)
    assert [len(level["examples"]) for level in levels] == [1, 2, 1, 1, 1]
    assert levels[0]["examples"][0].startswith('CoT: "Accelerate to merge')
    assert "OUTPUT FORMAT" not in json.dumps(request)


def test_lane_prompt_keeps_signal_guidance_and_both_frames() -> None:
    features = {"lane_features": LANE, "ego_features": EGO}
    request = jev_request(build_center_of_lane_prompt(COT, features), COT, "m")

    instructions = request["questions"]["cot_output_alignment"]["instructions"]
    assert instructions["signal_selection"].startswith("Pick the right signal")
    assert instructions["coordinate_frames"].count("COORDINATE FRAME (") == 2
    assert "Lane-frame summary" in request["state"]["trajectory"]
    # This prompt has no worked example for score 4.
    assert [len(level["examples"]) for level in _criteria(request)] == [1, 2, 1, 0, 2]


def test_argmax_level_maps_to_score_and_ties_go_low() -> None:
    answer = {
        "type": "score",
        "score": 2.5,
        "confidence": 0.4,
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.0, "3": 0.4, "4": 0.1},
    }

    evaluation = evaluation_from_answer(answer)

    data = evaluation["cot_output_alignment"]
    assert data["score"] == 2
    assert data["argmax_tie"] is True
    assert data["expected_score"] == 3.5
    assert data["score_probabilities"]["2"] == 0.4
    assert result_is_complete({"evaluation": evaluation})


def test_decisions_url_comes_from_the_openrouter_base_url() -> None:
    assert (
        _decisions_url("https://openrouter.ai/api/v1")
        == "https://openrouter.ai/api/alpha/decisions"
    )


def test_call_retries_rate_limits(monkeypatch) -> None:
    monkeypatch.setattr(run_jev_decisions.time, "sleep", lambda _: None)
    replies = iter(
        [
            httpx.Response(429, text="slow down"),
            httpx.Response(
                200,
                json={
                    "model": "typesafe/jev-x",
                    "id": "gen-1",
                    "provider": "TypeSafe",
                    "answers": {},
                    "usage": {"input_tokens": 10},
                },
            ),
        ]
    )
    transport = httpx.MockTransport(lambda request: next(replies))

    with httpx.Client(transport=transport) as client:
        body, timing = _call(client, "https://x/decisions", {"model": "m"})

    assert body["id"] == "gen-1"
    assert timing["status"] == "ok"
    assert timing["retry_count"] == 1
    assert [a["http_status"] for a in timing["attempts"]] == [429, 200]
    assert timing["model"] == "typesafe/jev-x"
