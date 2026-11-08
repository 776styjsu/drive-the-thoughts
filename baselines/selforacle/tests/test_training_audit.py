"""Reject corrupted streams/leakage while keeping visual labels explicitly pending."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from selforacle_baseline.data import validate_manifest

spec = importlib.util.spec_from_file_location(
    "training_audit", Path(__file__).parents[1] / "tools/audit_training_data.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def stream(tmp_path):
    sid = "clipgt-example"
    row = {"scene_id": sid, "uuid": "reconstruction", "path": "sample_set/example.usdz",
           "hf_revision": "a" * 40, "proposed_split": "train", "time_of_day": "day"}
    directory = tmp_path / "campaign" / "scenes" / sid
    for index in range(3):
        frame = directory / "extracted_frames" / f"{sid}_t{index:04d}"
        frame.mkdir(parents=True)
        image = frame / audit.CAMERA
        Image.new("RGB", (64, 32), (20 + index * 40, 60, 80)).save(image)
        (frame / "metadata.json").write_text(json.dumps({"clip_id": frame.name,
            "time_now_us": 100_000 + index * 100_000, "source": "alpasim_full_gt_replay",
            "image_sha256": audit.file_hash(image)}))
    report = {"scene_id": sid, "artifact_uuid": row["uuid"], "hf_revision": row["hf_revision"],
              "proposed_split": "train", "cadence_us": 100_000, "driver_calls": 0,
              "source_usdz_sha256": "b" * 64, "source_asl_sha256": "c" * 64,
              "frames": 3, "dimensions_wh": [64, 32], "first_timestamp_us": 100_000,
              "last_timestamp_us": 300_000}
    (directory / "render_complete.json").write_text(json.dumps(report))
    (directory / "render_request.json").write_text(json.dumps({"source": row}))
    (directory / "rig_trajectories.json").write_text(json.dumps({"T_world_base": [[1]],
        "rig_trajectories": [{"sequence_id": sid, "T_rig_world_timestamps_us": [0, 400_000],
                              "T_rig_worlds": [[[1]], [[2]]]}]}))
    (directory / "metadata.yaml").write_text("scene_id: clipgt-example\n")
    (directory / "datasource_summary.json").write_text("{}")
    output = tmp_path / "audit"
    (output / "contact_sheets").mkdir(parents=True)
    return row, tmp_path / "campaign", output, directory


def inspect(stream):
    row, campaign, output, _ = stream
    return audit.audit_scene(row, campaign, output, minimum_frames=3,
                             width_bounds=(64, 64), expected_height=32)


def test_valid_stream_preserves_pending_labels_and_training_gate(stream):
    scene, frames, measurements = inspect(stream)
    assert scene["automatic_status"] == "passed"
    assert len(frames) == len(measurements) == 3
    assert scene["nominal"] is None and all(f["nominal"] is None for f in frames)
    assert (stream[2] / scene["contact_sheet"]).is_file()
    with pytest.raises(ValueError, match="nominal=true"):
        validate_manifest({"schema_version": 1, "frames": frames})


@pytest.mark.parametrize("kind", ["hash", "decode", "cadence", "source", "count", "dimension"])
def test_invalid_input_is_reported(stream, kind):
    row, campaign, output, directory = stream
    frame = directory / "extracted_frames" / "clipgt-example_t0001"
    if kind in ("hash", "decode"):
        image = frame / audit.CAMERA
        image.write_bytes(b"not a JPEG")
        if kind == "decode":
            meta = audit.read_json(frame / "metadata.json")
            meta["image_sha256"] = audit.file_hash(image)
            (frame / "metadata.json").write_text(json.dumps(meta))
    elif kind == "cadence":
        meta = audit.read_json(frame / "metadata.json")
        meta["time_now_us"] = 100_000
        (frame / "metadata.json").write_text(json.dumps(meta))
    elif kind == "source":
        report = audit.read_json(directory / "render_complete.json")
        report["hf_revision"] = "d" * 40
        (directory / "render_complete.json").write_text(json.dumps(report))
    elif kind == "count":
        report = audit.read_json(directory / "render_complete.json")
        report["frames"] = 4
        (directory / "render_complete.json").write_text(json.dumps(report))
    else:
        report = audit.read_json(directory / "render_complete.json")
        report["dimensions_wh"] = [63, 32]
        (directory / "render_complete.json").write_text(json.dumps(report))
    scene, _, _ = inspect(stream)
    assert scene["automatic_status"] == "failed"
    assert scene["errors"]
    if kind == "hash":
        assert any("hash differs" in e for e in scene["errors"])
    if kind == "decode":
        assert any("cannot identify image" in e for e in scene["errors"])


def test_reserved_scene_is_rejected_before_output(stream):
    row, campaign, _, _ = stream
    with (campaign / "cohort.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    reserved = campaign / "reserved.csv"
    reserved.write_text("scene_id\nclipgt-example\n")
    with pytest.raises(ValueError, match="Reserved evaluation scenes"):
        audit.load_inputs(campaign, reserved)


def test_duplicates_distinguish_static_frames_from_cross_split_leakage():
    rows = [{"scene_id": "a", "clip_id": "a_t0000", "split": "train", "hash": "same"},
            {"scene_id": "a", "clip_id": "a_t0001", "split": "train", "hash": "same"}]
    groups = audit.duplicate_groups(rows, "hash", frame_records=True)
    assert len(groups) == 1 and groups[0]["cross_split"] is False
    rows.append({"scene_id": "b", "clip_id": "b_t0000", "split": "test", "hash": "same"})
    groups = audit.duplicate_groups(rows, "hash", frame_records=True)
    assert groups[0]["cross_split"] is True


def test_recording_fingerprint_ignores_reconstruction_identity(stream):
    scene1, _, _ = inspect(stream)
    row, campaign, output, directory = stream
    row["scene_id"] = "clipgt-renamed"
    renamed = directory.with_name(row["scene_id"])
    directory.rename(renamed)
    rig_path = renamed / "rig_trajectories.json"
    rig = audit.read_json(rig_path)
    rig["rig_trajectories"][0]["sequence_id"] = row["scene_id"]
    rig_path.write_text(json.dumps(rig))
    # Other identity checks now fail, but the recording fingerprint must agree.
    scene2, _, _ = inspect(stream)
    assert scene1["gt_recording_fingerprint"] == scene2["gt_recording_fingerprint"]
