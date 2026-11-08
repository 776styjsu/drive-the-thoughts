"""Causality, cohort selection, and paired comparisons for temporal voting."""

import json

import pytest

from tools.evaluate_temporal_consistency import (
    evaluate,
    paired_intervals,
    weighted_decay_score,
)
from tools.prepare_temporal_consistency import prepare


def manifest_and_predictions(sequences):
    windows, predictions = [], []
    for scene, consistent, flags in sequences:
        ids = [f"{scene}_t{i:04d}" for i in range(5)]
        windows.append(
            {
                "clip_id": ids[-1],
                "scene_id": scene,
                "is_consistent": consistent,
                "frame_ids": ids,
                "timestamps_us": [i * 100_000 for i in range(5)],
            }
        )
        predictions.extend(
            {
                "clip_id": frame,
                "report": {"label": "inconsistent" if flag else "consistent"},
            }
            for frame, flag in zip(ids, flags)
        )
    manifest = {
        "kind": "temporal_consistency_windows",
        "schema_version": 1,
        "config": {"step_us": 100_000},
        "n_targets": len(windows),
        "windows": windows,
        "excluded": [],
    }
    return manifest, predictions


def test_votes_distinguish_isolated_alarm_persistence_and_majority():
    manifest, predictions = manifest_and_predictions(
        [
            ("isolated", True, [0, 0, 0, 0, 1]),
            ("onset", False, [0, 0, 0, 1, 1]),
            ("intermittent", False, [1, 0, 1, 0, 1]),
        ]
    )
    # A future prediction must not influence any target.
    predictions.append(
        {"clip_id": "isolated_t0005", "report": {"label": "inconsistent"}}
    )
    result = evaluate(manifest, predictions, "rule_based", bootstrap=100)
    rows = result["targets"]
    assert list(rows[0]["alarms"].values()) == [True, False, False, False, False, False]
    assert list(rows[1]["alarms"].values()) == [True, True, False, True, False, True]
    assert list(rows[2]["alarms"].values()) == [True, False, False, True, True, True]
    assert result["policies"]["two_of_three"]["accuracy"] == 1
    assert [row["alarms"]["weighted_decay"] for row in rows] == [False, True, True]
    assert result["policies"]["single_frame"]["confusion"]["false_positive"] == 1
    delta = result["policies"]["two_of_three"]["difference_vs_single_frame"]["accuracy"]
    assert delta["estimate"] == pytest.approx(1 / 3)


def test_weighted_decay_uses_causal_half_life_and_threshold():
    # With a two-frame half-life, weights are [1/4, 1/(2 sqrt(2)), 1/2,
    # 1/sqrt(2), 1] from oldest to current. Normalize before thresholding.
    assert weighted_decay_score([1, 0, 0, 0, 0], 2) < 0.5
    assert weighted_decay_score([1, 1, 0, 0, 1], 2) > 0.5
    manifest, predictions = manifest_and_predictions([("a", False, [1, 1, 0, 0, 1])])
    default = evaluate(manifest, predictions, "rule_based", bootstrap=20)
    assert default["targets"][0]["alarms"]["weighted_decay"]
    assert not default["targets"][0]["alarms"]["two_of_three"]
    assert default["config"]["decay_half_life_frames"] == 2
    strict = evaluate(
        manifest, predictions, "rule_based", decay_threshold=0.6, bootstrap=20
    )
    assert not strict["targets"][0]["alarms"]["weighted_decay"]
    for half_life in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="half-life"):
            evaluate(
                manifest, predictions, "rule_based", decay_half_life_frames=half_life
            )
    for threshold in (0, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="Decay threshold"):
            evaluate(manifest, predictions, "rule_based", decay_threshold=threshold)


@pytest.mark.parametrize(
    "failure", ["missing", "invalid_parse", "error", "dry_run", "nan", "timestamp"]
)
def test_bad_history_excludes_target_from_every_policy(failure):
    manifest, predictions = manifest_and_predictions(
        [
            ("a", True, [0] * 5),
            ("b", False, [1] * 5),
        ]
    )
    if failure == "missing":
        predictions.pop(0)
    elif failure == "invalid_parse":
        predictions[0]["report"]["label"] = "invalid_parse"
    elif failure == "error":
        predictions[0]["error"] = "failed request"
    elif failure == "dry_run":
        predictions[0]["evaluation"] = {"dry_run": True}
    elif failure == "nan":
        predictions[0]["report"]["score"] = float("nan")
    else:
        predictions[0]["timestamp_us"] = 999
    result = evaluate(manifest, predictions, "rule_based", bootstrap=20)
    assert result["n_evaluated"] == 1
    assert result["prediction_exclusions"][0]["clip_id"] == "a_t0004"
    assert all(p["n_targets"] == 1 for p in result["policies"].values())
    assert all(p["balanced_accuracy"] is None for p in result["policies"].values())
    json.dumps(result, allow_nan=False)


def test_llm_threshold_and_duplicate_repeats():
    manifest, predictions = manifest_and_predictions([("a", False, [1] * 5)])
    predictions = [
        {"clip_id": p["clip_id"], "evaluation": {"cot_output_alignment": {"score": 2}}}
        for p in predictions
    ]
    result = evaluate(manifest, predictions, "cot_output_alignment", bootstrap=20)
    assert result["policies"]["single_frame"]["recall"] == 1
    result = evaluate(
        manifest, predictions, "cot_output_alignment", score_threshold=1, bootstrap=20
    )
    assert result["policies"]["single_frame"]["recall"] == 0
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate(manifest, predictions + predictions[:1], "cot_output_alignment")


def test_window_cannot_include_future_or_another_scene():
    manifest, predictions = manifest_and_predictions([("a", True, [0] * 5)])
    manifest["windows"][0]["timestamps_us"][0] = 999_999
    with pytest.raises(ValueError, match="Invalid causal window"):
        evaluate(manifest, predictions, "rule_based")


def test_empty_cohort_has_no_spurious_performance():
    manifest, _ = manifest_and_predictions([("a", True, [0] * 5)])
    result = evaluate(manifest, [], "rule_based", bootstrap=20)
    assert result["n_evaluated"] == 0
    assert result["policies"]["single_frame"]["accuracy"] is None
    assert result["policies"]["single_frame"]["precision"] is None


def test_bootstrap_samples_scenes_together_and_is_reproducible():
    manifest, predictions = manifest_and_predictions(
        [
            ("a", True, [0, 0, 0, 0, 1]),
            ("b", False, [1] * 5),
        ]
    )
    rows = evaluate(manifest, predictions, "rule_based", bootstrap=20)["targets"]
    # Put both targets in one scene: its paired accuracy improvement is fixed.
    rows[1]["scene_id"] = "a"
    first = paired_intervals(rows, 100, 42)
    assert first == paired_intervals(rows, 100, 42)
    assert first["two_of_three"]["accuracy"]["ci95"] == [0.5, 0.5]
    assert first["single_frame"]["accuracy"]["ci95"] == [0, 0]


@pytest.fixture
def source_data(tmp_path):
    paths = []
    for i in range(7):
        path = tmp_path / "source" / f"scene_t{i:04d}" / "metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "clip_id": path.parent.name,
                    "time_now_us": i * 100_000,
                    "time_query_us": (i + 1) * 100_000,
                    "chain_of_thought": f"decision {i}",
                    "trajectory_poses": [{"x": j, "y": 0} for j in range(3)],
                }
            )
        )
        paths.append(path)
    target_path = tmp_path / "released" / "metadata.json"
    target_path.parent.mkdir()
    target_path.write_bytes(paths[5].read_bytes())
    entry = {
        "clip_id": "scene_t0005",
        "cot_reliable": True,
        "cot_action_consistency": False,
    }
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps([{**entry, "source_scene_file": str(target_path)}]))
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{**entry, "source_scene_file": str(paths[5])}]))
    return benchmark, source, paths


def test_preparation_exports_only_past_inputs_without_propagating_labels(source_data):
    benchmark, source, _ = source_data
    manifest, frames = prepare(benchmark, source, [])
    assert len(manifest["windows"]) == 1
    assert [f["clip_id"] for f in frames] == [f"scene_t{i:04d}" for i in range(1, 6)]
    assert all(
        "cot_reliable" not in f and "cot_action_consistency" not in f for f in frames
    )
    assert manifest["windows"][0]["is_consistent"] is False


def test_preparation_includes_unreliable_targets_only_on_request(source_data):
    benchmark, source, _ = source_data
    for path in (benchmark, source):
        entries = json.loads(path.read_text())
        entries[0]["cot_reliable"] = False
        path.write_text(json.dumps(entries))

    assert prepare(benchmark, source, [])[0]["windows"] == []
    manifest, frames = prepare(benchmark, source, [], include_unreliable=True)
    assert [w["clip_id"] for w in manifest["windows"]] == ["scene_t0005"]
    assert manifest["config"]["reliable_only"] is False
    assert len(frames) == 5


@pytest.mark.parametrize(
    "problem",
    ["missing", "empty", "string_empty", "timestamp", "target_mismatch", "geometry"],
)
def test_preparation_rejects_incomplete_or_misaligned_windows(source_data, problem):
    benchmark, source, paths = source_data
    if problem == "missing":
        paths[2].unlink()
    elif problem != "geometry":
        path = paths[5] if problem == "target_mismatch" else paths[2]
        meta = json.loads(path.read_text())
        if problem == "timestamp":
            meta["time_now_us"] += 1
        elif problem == "string_empty":
            meta["chain_of_thought"] = "['']"
        else:
            meta["chain_of_thought"] = None
        path.write_text(json.dumps(meta))
    manifest, frames = prepare(
        benchmark, source, [], require_lane_geometry=problem == "geometry"
    )
    assert frames == []
    assert len(manifest["excluded"]) == 1
    assert manifest["excluded"][0]["reason"]
