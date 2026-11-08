import copy

import numpy as np
import pytest
from PIL import Image

from selforacle_baseline.data import sha256, validate_manifest
from selforacle_baseline.smoke import (
    PURPOSE, SCHEMA, _decode, _select, validate_smoke_manifest,
)


@pytest.fixture
def smoke_manifest(tmp_path):
    rows = []
    for index, split in enumerate(("train", "calibration", "test")):
        sid = f"clipgt-scene{index}"
        for frame in range(6):
            rows.append({"scene_id": sid, "sequence_id": f"sequence{index}",
                         "clip_id": f"{sid}_t{frame:04}", "now_us": frame * 100_000,
                         "image_path": str(tmp_path / sid / str(frame) / "camera.jpg"),
                         "split": split, "source_split": "train", "nominal": None})
    return {"schema_version": SCHEMA, "intended_use": PURPOSE, "training_ready": False,
            "camera": "camera.jpg", "reserved_test_scenes": [], "frames": rows}


def test_pending_pilot_is_isolated_from_research_manifest(smoke_manifest):
    assert validate_smoke_manifest(smoke_manifest)["scenes"] == {
        "train": 1, "calibration": 1, "test": 1}
    with pytest.raises(ValueError, match="Unsupported manifest schema"):
        validate_manifest(smoke_manifest)
    research = copy.deepcopy(smoke_manifest)
    research["schema_version"] = 1
    with pytest.raises(ValueError, match="requires explicit nominal"):
        validate_manifest(research)


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m["frames"][0].update(source_split="calibration"), "original training pool"),
    (lambda m: m.update(reserved_test_scenes=["clipgt-scene0"]), "Reserved"),
    (lambda m: m["frames"][0].update(split="test"), "Scene leakage"),
    (lambda m: m["frames"][0].update(nominal=False), "rejected nominal"),
    (lambda m: m["frames"][1].update(now_us=200_000), "Duplicate"),
    (lambda m: m["frames"][1].update(now_us=150_000), "10 Hz"),
    (lambda m: m.update(training_ready=True), "research training readiness"),
])
def test_smoke_excludes_holdouts_leakage_and_bad_inputs(smoke_manifest, mutation, match):
    mutation(smoke_manifest)
    with pytest.raises(ValueError, match=match):
        validate_smoke_manifest(smoke_manifest)


def test_cached_preprocessing_matches_loader_and_detects_corruption(tmp_path):
    import hashlib
    from selforacle_baseline.pipeline import Images

    image = tmp_path / "camera.jpg"
    pixels = np.random.default_rng(4).integers(0, 256, size=(160, 320, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image)
    with Image.open(image) as source:
        reduced = np.asarray(source.convert("RGB").resize((160, 80), Image.Resampling.BOX))
    row = {"image_path": str(image), "image_sha256": sha256(image),
           "preprocessed_rgb_sha256": hashlib.sha256(reduced.tobytes()).hexdigest()}
    actual = _decode(row).astype(np.float32) / 255.0
    expected = Images([row], {"width": 160, "height": 80})[0]
    np.testing.assert_array_equal(actual, expected)
    changed = dict(row, preprocessed_rgb_sha256="changed")
    with pytest.raises(ValueError, match="Preprocessed pixels"):
        _decode(changed)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image changed"):
        _decode(row)


def test_selection_is_reproducible_and_covers_lighting_speed():
    pool = [{"scene_id": f"scene{index}", "time_of_day": "day" if index % 2 else "night",
             "ego_speed": ("low", "medium", "high")[index % 3]} for index in range(90)]
    selected = _select(pool, 30, 0, "train")
    assert selected == _select(list(reversed(pool)), 30, 0, "train")
    assert len({row["scene_id"] for row in selected}) == 30
    assert len({(row["time_of_day"], row["ego_speed"]) for row in selected}) == 6
