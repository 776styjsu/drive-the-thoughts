import copy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from selforacle_baseline.benchmark_safety import (
    classification_metrics, evaluate, prepare_manifest, validate_windows,
)
from selforacle_baseline.data import read_json, sha256, write_json


def test_unsafe_is_positive_and_continuous_metrics_handle_ties():
    actual = [True, True, False, False]
    scores = [0.9, 0.6, 0.8, 0.1]
    result = classification_metrics(actual, scores, 0.7)
    assert result["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert result["unsafe_recall"] == result["unsafe_precision"] == 0.5
    assert result["safe_false_alarm_rate"] == 0.5
    assert result["auroc_continuous_anomaly_score"] == 0.75
    assert result["average_precision_unsafe"] == pytest.approx(5 / 6)
    tied = classification_metrics(actual, [1, 1, 1, 1], 1)
    assert tied["unsafe_recall"] == 0  # Strict greater-than matches the monitor.
    assert tied["unsafe_precision"] is None
    assert tied["auroc_continuous_anomaly_score"] == tied["average_precision_unsafe"] == 0.5
    safe_only = classification_metrics([False, False], [1, 2], 1.5)
    assert safe_only["unsafe_recall"] is None
    assert safe_only["auroc_continuous_anomaly_score"] is None


@pytest.fixture
def benchmark_files(tmp_path):
    entries, sources = [], []
    for index in range(2):
        sid = f"clipgt-scene{index}"
        for frame in range(5):
            cid = f"{sid}_t{frame:04}"
            directory = tmp_path / "source" / sid / cid
            directory.mkdir(parents=True)
            # History needs only the camera and timestamp, not CoT/driver returns.
            metadata = {"clip_id": cid, "time_now_us": frame * 100_000}
            if frame == 4:
                metadata.update(chain_of_thought="test", trajectory_poses=[{"x": 1, "y": 0}])
            write_json(directory / "metadata.json", metadata)
            Image.fromarray(np.full((8, 16, 3), 20 + frame + index, dtype=np.uint8)).save(
                directory / "camera_front_wide_120fov.jpg")
        target = tmp_path / "released" / cid
        shutil.copytree(directory, target)
        entries.append({"clip_id": cid, "planned_trajectory_safe": index == 0,
                        "cot_reliability": {"reliable": index == 0},
                        "source_scene_file": str(target / "metadata.json")})
        sources.append({"clip_id": cid, "source_scene_file": str(directory / "metadata.json")})
    benchmark, source = tmp_path / "benchmark.json", tmp_path / "source.json"
    write_json(benchmark, entries)
    write_json(source, sources)
    return benchmark, source, tmp_path


def test_causal_window_without_historical_cot_and_with_original_target_identity(benchmark_files):
    benchmark, source, root = benchmark_files
    manifest = prepare_manifest(benchmark, source, [], root)
    validate_windows(manifest, set())
    assert len(manifest["windows"]) == 2 and len(manifest["frames"]) == 10
    assert [window["is_unsafe"] for window in manifest["windows"]] == [False, True]
    assert not manifest["excluded"]
    with pytest.raises(ValueError, match="scene leakage"):
        validate_windows(manifest, {"clipgt-scene1"})
    broken = copy.deepcopy(manifest)
    broken["windows"][0]["timestamps_us"][-1] += 100_000
    with pytest.raises(ValueError, match="causal"):
        validate_windows(broken, set())
    broken = copy.deepcopy(manifest)
    broken["frames"][0]["scene_id"] = "another-scene"
    with pytest.raises(ValueError, match="identity"):
        validate_windows(broken, set())


def test_mismatched_target_and_missing_history_are_excluded_not_filled(benchmark_files):
    benchmark, source, root = benchmark_files
    (root / "released" / "clipgt-scene0_t0004" / "camera_front_wide_120fov.jpg").write_bytes(b"different")
    (root / "source" / "clipgt-scene1" / "clipgt-scene1_t0002" / "metadata.json").unlink()
    manifest = prepare_manifest(benchmark, source, [], root)
    assert not manifest["windows"] and not manifest["frames"]
    assert len(manifest["excluded"]) == 2
    assert "differs" in manifest["excluded"][0]["reason"]
    assert manifest["excluded"][0]["is_unsafe"] is False
    assert manifest["excluded"][1]["is_unsafe"] is True


@pytest.fixture
def full_evaluation(benchmark_files):
    import torch
    from selforacle_baseline.full_training import SCHEMA as FULL_SCHEMA
    from selforacle_baseline.models import Reconstructor

    torch.set_num_threads(1)
    benchmark, source, root = benchmark_files
    rows = []
    for split in ("train", "validation", "calibration", "test"):
        sid = f"clipgt-full-{split}"
        rows.append({"scene_id": sid, "sequence_id": sid, "clip_id": f"{sid}_t0000",
                     "now_us": 0, "nominal": None, "split": split,
                     "source_split": "train" if split == "validation" else split,
                     "image_path": str(root / sid / "camera_front_wide_120fov.jpg"),
                     "render_width": 1920, "render_height": 1080})
    training = {"schema_version": FULL_SCHEMA,
                "nominal_policy": "audited_GT_driving_pending_nominal_review",
                "research_training_ready": False, "camera": "camera_front_wide_120fov.jpg",
                "reserved_test_scenes": [], "frames": rows}
    path = root / "full_manifest.json"
    write_json(path, training)
    windows = root / "windows.json"
    write_json(windows, prepare_manifest(benchmark, source, [], root))
    config = {"model": "sae", "height": 8, "width": 16, "camera": training["camera"],
              "preprocessing": "rgb_box_resize_0_1_hwc", "score": "l2",
              "vae_inference": "posterior_mean", "seed": 1}
    model = Reconstructor(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = root / "full_model.pt"
    torch.save({"provenance": {"schema_version": FULL_SCHEMA, "manifest_sha256": sha256(path),
                               "research_training_ready": False, "nominal_review_pending": True,
                               "config": config, "epochs": 2, "epochs_completed": 7, "seed": 1},
                "state_dict": model.state_dict()}, checkpoint)
    calibration = root / "full_calibration.json"
    write_json(calibration, {"schema_version": FULL_SCHEMA, "nominal_review_pending": True,
                             "manifest_sha256": sha256(path), "checkpoint_sha256": sha256(checkpoint),
                             "config": config, "window": 5, "max_gap_us": 150_000,
                             "calibrations": [{"distribution": {"epsilon": 0.05, "threshold": 7.0}},
                                              {"distribution": {"epsilon": 0.01, "threshold": 9.0}}]})
    return SimpleNamespace(checkpoint=checkpoint, training_manifest=path, calibration=calibration,
                           manifest=windows, allow_pilot_checkpoint=False, allow_pending_nominal=True,
                           device="cpu", batch_size=4, workers=0, output=root / "evaluation")


def test_full_checkpoint_requires_explicit_pending_review_opt_in(full_evaluation):
    full_evaluation.allow_pending_nominal = False
    with pytest.raises(ValueError, match="allow-pending-nominal"):
        evaluate(full_evaluation)
    assert not full_evaluation.output.exists()


def test_full_evaluation_preserves_review_status_and_frozen_thresholds(full_evaluation):
    before = {path: sha256(path) for path in
              (full_evaluation.checkpoint, full_evaluation.calibration, full_evaluation.training_manifest)}
    evaluate(full_evaluation)
    result = read_json(full_evaluation.output / "evaluation.json")
    assert result["result_status"] == "full_scale_provisional_nominal_review_pending"
    assert result["nominal_review_pending"] is True and result["research_training_ready"] is False
    assert result["training_epochs"] == 2 and result["epochs_completed"] == 7
    assert result["n_scored"] == 2 and not result["excluded"]
    primary = [row for row in result["summaries"]
               if row["cohort"] == "all_150" and row["policy"] == "five_frame_mean"]
    assert [row["threshold"] for row in primary] == [7.0, 9.0]
    assert primary[0]["confusion"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}
    assert primary[1]["confusion"] == {"tp": 0, "fp": 0, "tn": 1, "fn": 1}
    assert all(sha256(path) == digest for path, digest in before.items())


@pytest.mark.parametrize("split", ["train", "validation", "calibration", "test"])
def test_full_evaluation_blocks_benchmark_overlap_with_every_cohort_split(full_evaluation, split):
    import torch

    training = read_json(full_evaluation.training_manifest)
    row = next(row for row in training["frames"] if row["split"] == split)
    row.update(scene_id="clipgt-scene0", sequence_id="clipgt-scene0", clip_id="clipgt-scene0_t0000")
    write_json(full_evaluation.training_manifest, training)
    digest = sha256(full_evaluation.training_manifest)
    checkpoint = torch.load(full_evaluation.checkpoint, weights_only=True)
    checkpoint["provenance"]["manifest_sha256"] = digest
    torch.save(checkpoint, full_evaluation.checkpoint)
    calibration = read_json(full_evaluation.calibration)
    calibration.update(manifest_sha256=digest, checkpoint_sha256=sha256(full_evaluation.checkpoint))
    write_json(full_evaluation.calibration, calibration)
    with pytest.raises(ValueError, match="scene leakage"):
        evaluate(full_evaluation)
    assert not full_evaluation.output.exists()
