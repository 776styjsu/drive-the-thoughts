# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
from argparse import Namespace

import pytest

# The OpenRouter runners need httpx, which comes with the llm extra (via openai).
httpx = pytest.importorskip("httpx")

from alpasim_utils.cot_consistency.llm_judge import resolve_provider
from cot_analysis.pipeline import JudgeInput, result_is_complete
from tools.run_openrouter_batch import (
    _result_for,
    _split_provider_params,
    _submit,
    _write_variant,
)


def _provider(monkeypatch) -> dict:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    return resolve_provider("claude_opus_openrouter", load_dotenv=False)


def test_routing_moves_to_batch_level_and_streaming_is_dropped(monkeypatch) -> None:
    routing, body = _split_provider_params(_provider(monkeypatch), seed=42)

    assert routing == {"only": ["anthropic"], "allow_fallbacks": False}
    assert body == {
        "model": "anthropic/claude-opus-5.5",
        "seed": 42,
        "max_tokens": 32768,
        "reasoning": {"effort": "high"},
        "response_format": {"type": "json_object"},
    }


def _spec() -> dict:
    judge_input = JudgeInput("cot", {"features": 1}, "prompt text", None, "fp")
    return {
        "variant": "llm",
        "entries": [{"clip_id": "clip-a"}],
        "judge_inputs": {"clip-a": judge_input},
    }


def test_submit_puts_routing_keys_before_requests(monkeypatch) -> None:
    routing, body = _split_provider_params(_provider(monkeypatch), seed=42)
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["payload"] = json.loads(request.content)
        return httpx.Response(202, json={"id": "batch-1", "status": "validating"})

    client = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))
    batch, custom_ids = _submit(client, routing, body, [_spec()])

    payload = sent["payload"]
    assert list(payload) == ["endpoint", "model", "provider", "completion_window", "requests"]
    assert payload["requests"][0]["custom_id"] == "llm-0000"
    assert payload["requests"][0]["body"]["messages"][1] == {"role": "user", "content": "prompt text"}
    assert batch["id"] == "batch-1"
    assert custom_ids == {"llm-0000": {"variant": "llm", "clip_id": "clip-a", "fingerprint": "fp"}}


class _Args:
    variant = "llm"
    prompt = "default"
    trajectory_frame = "ego_rig"
    lane_reference = "auto"


def _result_spec() -> dict:
    spec = _spec()
    spec["cot_args"] = _Args()
    spec["provider"] = {"model": "anthropic/claude-opus-5.5"}
    return spec


def test_successful_item_becomes_complete_result() -> None:
    content = json.dumps({"cot_output_alignment": {"justification": "ok", "score": 2}})
    item = {
        "custom_id": "llm-0000",
        "response": {
            "status_code": 200,
            "body": {
                "id": "gen-1",
                "model": "anthropic/claude-opus-5.5",
                "provider": "Anthropic",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        },
        "error": None,
    }

    result = _result_for(_result_spec(), {"clip_id": "clip-a"}, item, "batch-1")

    assert result_is_complete(result)
    assert result["evaluation"]["cot_output_alignment"]["score"] == 2
    assert result["llm_timing"]["served_by"] == "Anthropic"
    assert result["llm_timing"]["attempts"][0]["usage"]["completion_tokens"] == 5


def test_errored_or_missing_item_is_incomplete() -> None:
    item = {"custom_id": "llm-0000", "response": None, "error": {"type": "server_error", "message": "x"}}

    errored = _result_for(_result_spec(), {"clip_id": "clip-a"}, item, "batch-1")
    missing = _result_for(_result_spec(), {"clip_id": "clip-a"}, None, "batch-1")

    assert not result_is_complete(errored)
    assert errored["llm_timing"]["attempts"][0]["status"] == "error"
    assert not result_is_complete(missing)


def _item(custom_id: str, score: int) -> dict:
    content = json.dumps({"cot_output_alignment": {"justification": "j", "score": score}})
    return {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": {"choices": [{"message": {"content": content}}]}},
        "error": None,
    }


def test_results_match_entries_by_saved_clip_id_not_position(tmp_path) -> None:
    spec = _result_spec()
    spec["entries"] = [{"clip_id": "clip-a"}, {"clip_id": "clip-b"}]
    spec["judge_inputs"] = {
        clip: JudgeInput("cot", {}, f"prompt {clip}", None, f"fp-{clip}") for clip in ("clip-a", "clip-b")
    }
    spec.update(run_config={}, original_entry_count=2, skipped=[])
    custom_ids = {
        "llm-0000": {"variant": "llm", "clip_id": "clip-a", "fingerprint": "fp-clip-a"},
        "llm-0001": {"variant": "llm", "clip_id": "clip-b", "fingerprint": "fp-clip-b"},
        # Another variant's request with the same clip must not leak in.
        "f_llm_map_graph-0000": {"variant": "f_llm_map_graph", "clip_id": "clip-a", "fingerprint": "x"},
    }
    # Results arrive out of order.
    items = {
        "f_llm_map_graph-0000": _item("f_llm_map_graph-0000", 3),
        "llm-0001": _item("llm-0001", 1),
        "llm-0000": _item("llm-0000", 5),
    }
    args = Namespace(output_dir=tmp_path, label="opus", run_index=1)

    incomplete = _write_variant(args, spec, {"id": "batch-1"}, items, custom_ids, {})

    results = json.loads((tmp_path / "opus.llm.run_001.json").read_text())["results"]
    scores = {r["clip_id"]: r["evaluation"]["cot_output_alignment"]["score"] for r in results}
    prompts = {r["clip_id"]: r["judge_prompt"] for r in results}
    assert incomplete == 0
    assert scores == {"clip-a": 5, "clip-b": 1}
    assert prompts == {"clip-a": "prompt clip-a", "clip-b": "prompt clip-b"}
