# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import copy
import json
import sys

import numpy as np
import pytest
from PIL import Image

from cot_analysis import __main__ as cli
from cot_analysis import pipeline


@pytest.fixture
def entry():
    return {
        "clip_id": "clip",
        "chain_of_thought": "Keep lane and hold speed.",
        "trajectory_xy": np.column_stack((np.arange(11), np.zeros(11))),
    }


@pytest.fixture
def run_config():
    args = cli.build_parser().parse_args(["--benchmark_json", "unused.json"])
    args.variant, args.prompt, args.trajectory_frame, args.lane_reference = (
        cli._resolve_variant_settings(args)
    )
    return cli._judge_run_config(
        args,
        {
            "model": "test-model",
            "base_url": "http://judge.test/v1",
            "temperature": 0,
            "extra_params": {"max_tokens": 1024},
            "api_key": "must-not-be-persisted",
        },
        use_images=True,
    )


def saved_result(entry, evaluation=None):
    result = pipeline.process_entry(None, "test-model", entry)
    result["evaluation"] = evaluation or {"cot_output_alignment": {"score": 4}}
    return result


def resume(path, entries, run_config):
    return cli.load_resumable_results(
        path,
        entries,
        run_config=run_config,
        input_fingerprints={
            item["clip_id"]: pipeline.prepare_judge_input(item).fingerprint
            for item in entries
        },
    )


@pytest.mark.parametrize(
    "evaluation",
    [
        None,
        [],
        {},
        {"dry_run": True},
        {"parse_error": True},
        {"error": "timeout"},
        {"cot_output_alignment": {}},
        {"cot_output_alignment": {"verdict": "unknown"}},
        *[
            {"cot_output_alignment": {"score": score}}
            for score in [None, True, "bad", "NaN", "Infinity", 0, 6]
        ],
    ],
)
def test_resume_retries_unusable_evaluations(tmp_path, entry, run_config, evaluation):
    result = saved_result(entry)
    result["evaluation"] = evaluation
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"run_config": run_config, "results": [result]}))

    assert resume(path, [entry], run_config) == ([], set())
    assert pipeline.aggregate_results([result])["total_evaluated"] == 0


@pytest.mark.parametrize(
    "dimension", [{"score": 1}, {"score": "5"}, {"verdict": "inconsistent"}]
)
def test_resume_keeps_complete_matching_judgments(
    tmp_path, entry, run_config, dimension
):
    result = saved_result(entry, {"cot_output_alignment": dimension})
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"run_config": run_config, "results": [result]}))

    assert resume(path, [entry], run_config) == ([result], {"clip"})
    assert pipeline.aggregate_results([result])["total_evaluated"] == 1
    assert "must-not-be-persisted" not in json.dumps(run_config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model", "different-model"),
        ("provider", "openai"),
        ("base_url", "http://other.test/v1"),
        ("seed", 43),
        ("temperature", 1),
        ("extra_params", {"max_tokens": 2048}),
        ("variant", "center_of_lane"),
        ("prompt", "center_of_lane"),
        ("trajectory_frame", "dual"),
        ("lane_reference", "route"),
        ("use_images", False),
    ],
)
def test_resume_invalidates_changed_request_settings(
    tmp_path, entry, run_config, key, value
):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"run_config": run_config, "results": [saved_result(entry)]})
    )
    changed = {**run_config, key: value}

    assert resume(path, [entry], changed) == ([], set())


@pytest.mark.parametrize("change", ["cot", "trajectory", "image", "timestamp"])
def test_resume_invalidates_only_the_changed_clip(tmp_path, entry, run_config, change):
    second = {**entry, "clip_id": "other"}
    results = [saved_result(entry), saved_result(second)]
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"run_config": run_config, "results": results}))
    changed = copy.deepcopy(entry)
    if change == "cot":
        changed["chain_of_thought"] = "Turn left."
    elif change == "trajectory":
        changed["trajectory_xy"][:, 0] *= 2
    elif change == "image":
        changed["image"] = Image.new("RGB", (2, 2), "red")
    else:
        changed["timestamp_us"] = 1_000_000

    assert resume(path, [changed, second], run_config) == ([results[1]], {"other"})


def test_prompt_edits_change_input_fingerprint(monkeypatch, entry):
    original = pipeline.prepare_judge_input(entry).fingerprint
    monkeypatch.setattr(
        pipeline,
        "resolve_prompt_builder",
        lambda name: lambda cot, features: "new prompt",
    )
    assert pipeline.prepare_judge_input(entry).fingerprint != original


def test_system_prompt_edits_change_input_fingerprint(monkeypatch, entry):
    original = pipeline.prepare_judge_input(entry).fingerprint
    monkeypatch.setattr(pipeline, "JUDGE_SYSTEM_PROMPT", "updated system prompt")
    assert pipeline.prepare_judge_input(entry).fingerprint != original


@pytest.mark.parametrize("missing", ["run_config", "judge_input_fingerprint"])
def test_legacy_results_without_provenance_are_retried(
    tmp_path, entry, run_config, missing
):
    result = saved_result(entry)
    payload = {"run_config": run_config, "results": [result]}
    (payload if missing == "run_config" else result).pop(missing)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))
    assert resume(path, [entry], run_config) == ([], set())


def test_cli_resume_retries_dry_run_and_reuses_only_matching_results(
    monkeypatch, tmp_path, entry
):
    output = tmp_path / "results.json"
    state = {"api_key": None}
    calls = []

    def provider(name, **overrides):
        return {
            "label": "mock",
            "model": overrides.get("model") or "test-model",
            "base_url": "http://judge.test/v1",
            "temperature": 0,
            "extra_params": {},
            "supports_images": True,
            **state,
        }

    def call_llm(client, model, prompt, **kwargs):
        calls.append((model, prompt, kwargs["seed"]))
        return '{"cot_output_alignment":{"score":4}}', {"status": "ok"}

    monkeypatch.setattr(cli, "resolve_provider", provider)
    monkeypatch.setattr(cli, "build_client", lambda *args: object())
    monkeypatch.setattr(
        cli,
        "extract_entries_from_benchmark",
        lambda *args, **kwargs: [copy.deepcopy(entry)],
    )
    monkeypatch.setattr(pipeline, "call_llm", call_llm)

    def run(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cot_analysis",
                "--benchmark_json",
                "unused.json",
                "--output",
                str(output),
                "--resume",
                "--delay",
                "0",
                *extra,
            ],
        )
        cli.main()
        return json.loads(output.read_text())

    assert run()["results"][0]["evaluation"] == {"dry_run": True}
    assert not calls
    state["api_key"] = "private-key"
    assert run()["summary"]["total_evaluated"] == 1
    assert len(calls) == 1
    run()
    assert len(calls) == 1
    entry["chain_of_thought"] = "Slow down."
    run()
    assert len(calls) == 2
    entry["trajectory_xy"][:, 0] *= 2
    run()
    assert len(calls) == 3
    run("--seed", "43")
    assert len(calls) == 4 and calls[-1][2] == 43
    report = run("--seed", "43", "--model", "other-model")
    assert len(calls) == 5 and calls[-1][0] == "other-model"
    assert report["run_config"]["model"] == "other-model"
    assert "private-key" not in output.read_text()
