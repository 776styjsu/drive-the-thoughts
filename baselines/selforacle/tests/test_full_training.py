import copy

import numpy as np
import pytest

from selforacle_baseline.data import validate_manifest, write_json, sha256, read_json
from selforacle_baseline.full_training import (
    SCHEMA, EarlyStopping, augment, rows_digest, validate_full_manifest,
)


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for index, split in enumerate(("train", "validation", "calibration", "test")):
        sid = f"clipgt-scene{index}"
        for frame in range(3):
            rows.append({"scene_id": sid, "sequence_id": sid, "clip_id": f"{sid}_t{frame:04}",
                         "now_us": frame * 100_000, "nominal": None, "split": split,
                         "source_split": "train" if split == "validation" else split,
                         "image_path": str(tmp_path / sid / str(frame) / "camera.jpg"),
                         "image_sha256": str(index) * 64, "render_width": 1920, "render_height": 1080})
    return {"schema_version": SCHEMA, "nominal_policy": "audited_GT_driving_pending_nominal_review",
            "research_training_ready": False, "camera": "camera.jpg",
            "reserved_test_scenes": [], "frames": rows}


def test_full_provisional_training_does_not_accept_or_change_nominal_labels(manifest):
    before = copy.deepcopy(manifest)
    summary = validate_full_manifest(manifest, allow_pending=True)
    assert summary["scenes"] == {"train": 1, "validation": 1, "calibration": 1, "test": 1}
    assert summary["pending_nominal_frames"] == 12
    assert manifest == before
    with pytest.raises(ValueError, match="explicit opt-in"):
        validate_full_manifest(manifest, allow_pending=False)
    with pytest.raises(ValueError, match="Unsupported manifest schema"):
        validate_manifest(manifest)
    manifest["frames"][0]["nominal"] = False
    with pytest.raises(ValueError, match="Rejected"):
        validate_full_manifest(manifest, allow_pending=True)


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m.update(reserved_test_scenes=["clipgt-scene0"]), "reserved benchmark"),
    (lambda m: m["frames"][6].update(source_split="train"), "holdout reassigned"),
    (lambda m: m["frames"][0].update(split="validation"), "leakage"),
    (lambda m: m["frames"][1].update(now_us=150_000), "consecutive"),
    (lambda m: m.update(research_training_ready=True), "research readiness"),
])
def test_full_split_holdout_and_review_guards(manifest, mutation, match):
    mutation(manifest)
    with pytest.raises(ValueError, match=match):
        validate_full_manifest(manifest, allow_pending=True)


def test_cache_row_digest_binds_order_frame_hash_and_timestamp(manifest):
    rows = manifest["frames"]
    digest = rows_digest(rows)
    assert digest != rows_digest(list(reversed(rows)))
    changed = copy.deepcopy(rows)
    changed[0]["image_sha256"] = "changed"
    assert digest != rows_digest(changed)
    changed = copy.deepcopy(rows)
    changed[0]["now_us"] += 1
    assert digest != rows_digest(changed)


def test_cache_cli_builds_verified_arrays_and_binds_manifest(manifest, tmp_path, monkeypatch):
    import hashlib
    import sys
    from pathlib import Path
    from PIL import Image
    from selforacle_baseline.full_training import main

    for row in manifest["frames"]:
        image = Path(row["image_path"])
        image.parent.mkdir(parents=True)
        pixels = np.full((80, 160, 3), (row["now_us"] // 100_000 + 1) * 25, dtype=np.uint8)
        Image.fromarray(pixels).save(image)
        row["image_sha256"] = sha256(image)
        with Image.open(image) as opened:
            reduced = np.asarray(opened.convert("RGB").resize((160, 80), Image.Resampling.BOX))
        row["preprocessed_rgb_sha256"] = hashlib.sha256(reduced.tobytes()).hexdigest()
    path, output = tmp_path / "manifest.json", tmp_path / "cache"
    write_json(path, manifest)
    monkeypatch.setattr(sys, "argv", ["full_training", "cache", "--manifest", str(path),
                                     "--output", str(output), "--workers", "2"])
    main()
    index = read_json(output / "index.json")
    assert index["manifest_sha256"] == sha256(path)
    assert set(index["splits"]) == {"train", "validation", "calibration", "test"}
    for split, info in index["splits"].items():
        assert info["sha256"] == sha256(Path(info["path"]))
        assert np.load(info["path"], allow_pickle=False).shape == (3, 80 * 160 * 3)


def test_early_stopping_saves_actual_best_even_without_meaningful_improvement():
    stopping = EarlyStopping(patience=2, min_relative=0.01)
    assert stopping.update(1.0) == (True, False)
    assert stopping.update(0.995) == (True, False)
    assert stopping.update(0.993) == (True, True)
    assert stopping.best == 0.993
    stopping = EarlyStopping(patience=2, min_relative=0.01)
    assert stopping.update(1.0) == (True, False)
    assert stopping.update(0.995) == (True, False)
    assert stopping.update(0.985) == (True, False)  # Cumulative meaningful improvement resets patience.
    with pytest.raises(ValueError, match="Nonfinite"):
        stopping.update(float("nan"))


def test_augmentation_is_reproducible_bounded_and_leaves_unselected_inputs_unchanged():
    import torch
    torch.set_num_threads(1)
    images = torch.rand((16, 80 * 160 * 3), generator=torch.Generator().manual_seed(7))
    sizes = torch.tensor([[1920, 1080]] * len(images))
    expected_selected = torch.rand(16, generator=torch.Generator().manual_seed(3)) < 0.6
    a = augment(images, sizes, torch.Generator().manual_seed(3), probability=0.6)
    b = augment(images, sizes, torch.Generator().manual_seed(3), probability=0.6)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a[~expected_selected], images[~expected_selected], rtol=0, atol=0)
    assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert (a[expected_selected] != images[expected_selected]).any()
    assert augment(images, sizes, torch.Generator(), probability=0) is images
    black = augment(torch.zeros_like(images), sizes, torch.Generator().manual_seed(1), probability=1)
    white = augment(torch.ones_like(images), sizes, torch.Generator().manual_seed(1), probability=1)
    assert torch.isfinite(black).all() and not black.any()
    assert torch.isfinite(white).all() and white.min() >= 0 and white.max() <= 1
