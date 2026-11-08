import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from selforacle_baseline.cli import main
from selforacle_baseline.data import (
    discover, load_split, prepare, read_json, validate_manifest, write_json,
)
from selforacle_baseline.pipeline import filtered_scores, fit_threshold


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(7)
    for scene in range(3):
        for frame in range(7):
            clip_id = f"clipgt-scene{scene}_t{frame:04}"
            directory = tmp_path / "frames" / f"scene{scene}" / clip_id
            directory.mkdir(parents=True)
            write_json(directory / "metadata.json", {
                "clip_id": clip_id, "time_now_us": frame * 100_000,
            })
            pixels = rng.integers(0, 256, size=(8, 16, 3), dtype=np.uint8)
            # Distinct brightnesses ensure a non-degenerate calibration distribution.
            pixels = (pixels.astype(float) * ((frame + 1) / 8)).astype(np.uint8)
            Image.fromarray(pixels).save(directory / "camera_front_wide_120fov.jpg")
    inventory = discover([tmp_path / "frames"], "camera_front_wide_120fov.jpg")
    for sequence, split in zip(inventory["sequences"], ("train", "calibration", "test")):
        sequence.update(split=split, nominal=split != "test")
    inventory_path = tmp_path / "inventory.json"
    write_json(inventory_path, inventory)
    path = tmp_path / "manifest.json"
    write_json(path, prepare(inventory_path))
    return path


def test_explicit_curation_and_protected_scenes(dataset):
    root = dataset.parent
    inventory = discover([root / "frames"], "camera_front_wide_120fov.jpg")
    assert all(s["split"] == "unassigned" and s["nominal"] is None for s in inventory["sequences"])
    pending = root / "pending.json"
    write_json(pending, inventory)
    with pytest.raises(ValueError, match="Assign"):
        prepare(pending)
    manifest = read_json(dataset)
    manifest["reserved_test_scenes"] = ["clipgt-scene0"]
    with pytest.raises(ValueError, match="Reserved"):
        validate_manifest(manifest)


def test_overlap_missing_nominal_duplicates_and_mutated_image(dataset):
    manifest = read_json(dataset)
    assert validate_manifest(manifest)["frames"] == {"train": 7, "calibration": 7, "test": 7}
    broken = copy.deepcopy(manifest)
    train_index = next(i for i, r in enumerate(manifest["frames"]) if r["split"] == "train")
    broken["frames"][train_index]["split"] = "calibration"
    with pytest.raises(ValueError, match="Scene leakage"):
        validate_manifest(broken)
    broken = copy.deepcopy(manifest)
    broken["frames"][train_index]["nominal"] = None
    with pytest.raises(ValueError, match="nominal"):
        validate_manifest(broken)
    broken = copy.deepcopy(manifest)
    broken["frames"].append(broken["frames"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(broken)
    first = next(r for r in manifest["frames"] if r["split"] == "train")
    Path(first["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image changed"):
        load_split(dataset, "train")


def test_temporal_filter_is_causal_and_resets():
    rows = [{"sequence_id": "a", "now_us": t} for t in (0, 100, 200, 900, 1000)]
    rows += [{"sequence_id": "b", "now_us": 0}]
    result = filtered_scores(rows, [1, 3, 5, 7, 9, 100], 2, 150)
    assert [r["score"] for r in result] == [None, 2, 4, None, 8, None]
    prefix = filtered_scores(rows[:3], [1, 3, 5], 2, 150)
    assert prefix == result[:3]
    with pytest.raises(ValueError, match="increase"):
        filtered_scores([rows[0], rows[0]], [1, 2], 2, 150)


def test_gamma_threshold():
    rng = np.random.default_rng(4)
    samples = rng.gamma(3, 2, 2000).tolist()
    loose = fit_threshold(samples, 0.05)
    strict = fit_threshold(samples, 0.01)
    assert strict["threshold"] > loose["threshold"] > 0
    assert loose["loc"] == 0
    for invalid in ([1, 1], [0, 1], [float("nan"), 2], [1]):
        with pytest.raises(ValueError):
            fit_threshold(invalid, 0.05)


@pytest.mark.parametrize("model", ["sae", "vae"])
def test_train_calibrate_predict(dataset, model):
    import torch
    torch.set_num_threads(1)
    output = dataset.parent / model
    common = ["--manifest", str(dataset), "--device", "cpu", "--batch-size", "4"]
    main(["train", *common, "--model", model, "--epochs", "1", "--height", "8",
          "--width", "16", "--output", str(output)])
    checkpoint = output / "model.pt"
    calibration = output / "calibration.json"
    prediction = output / "predictions.json"
    main(["calibrate", *common, "--checkpoint", str(checkpoint), "--window", "2",
          "--output", str(calibration)])
    main(["predict", *common, "--checkpoint", str(checkpoint), "--calibration", str(calibration),
          "--output", str(prediction)])
    result = read_json(prediction)
    assert len(result["results"]) == 7  # Includes the final partial batch.
    assert result["results"][0]["flag"] is None
    assert sum(r["scored"] for r in result["results"]) == 6
    assert all(r["scene_id"] == "clipgt-scene2" for r in result["results"])
    main(["predict", *common, "--checkpoint", str(checkpoint), "--calibration", str(calibration),
          "--output", str(output / "repeat.json")])
    assert read_json(output / "repeat.json")["results"] == result["results"]
    # Relabeling a previously trained scene in a new manifest must still be rejected.
    changed = read_json(dataset)
    changed["frames"] = [dict(r, split="test") for r in changed["frames"] if r["split"] == "train"]
    changed_path = dataset.parent / "leak.json"
    write_json(changed_path, changed)
    with pytest.raises(SystemExit):
        main(["predict", "--manifest", str(changed_path), "--checkpoint", str(checkpoint),
              "--calibration", str(calibration), "--output", str(output / "leak.json")])
    bad_calibration = read_json(calibration)
    bad_calibration["checkpoint_sha256"] = "wrong"
    bad_path = output / "bad_calibration.json"
    write_json(bad_path, bad_calibration)
    with pytest.raises(SystemExit):
        main(["predict", *common, "--checkpoint", str(checkpoint), "--calibration", str(bad_path),
              "--output", str(output / "bad_predictions.json")])

