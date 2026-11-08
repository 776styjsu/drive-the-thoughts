"""Regression checks for scene identity, coverage counts, and annotation handoff."""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_benchmark_viewer import annotator_record, build_records
from build_frame_selector import build, cot_id, load_proposals, validate_categories

A = "clipgt-00000000-0000-0000-0000-000000000001"
B = "clipgt-00000000-0000-0000-0000-000000000002"
C = "clipgt-00000000-0000-0000-0000-000000000003"
CATEGORIES = {
    "longitudinal": "speed_adaptation_road_events",
    "lateral": "lane_keeping_centering",
    "causal_factors": ["road_events"],
}


def label(categories=CATEGORIES):
    return {
        "high_level_decision": {a: categories[a] for a in ["longitudinal", "lateral"]},
        "critical_component_categories": categories["causal_factors"],
    }


def frame(root, sid, index, cot="Slow for the bend", plan=True):
    directory = (
        root / f"scene_{sid[-1]}_{sid}" / "extracted_frames" / f"{sid}_t{index:04}"
    )
    directory.mkdir(parents=True)
    metadata = directory / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "clip_id": directory.name,
                "chain_of_thought": cot,
                "time_now_us": 1000000 + index * 100000,
                "trajectory_poses": [{"x": 0}] if plan else [],
            }
        )
    )
    (directory / "camera_front_wide_120fov.jpg").write_bytes(b"image")
    return metadata


@pytest.fixture
def args(tmp_path):
    root = tmp_path / "rollouts"
    root.mkdir()
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "clip_id": A + "_t0001",
                    "chain_of_thought": "Slow for the bend",
                    "labels": {"cot_decision_label": label()},
                }
            ]
        )
    )
    return argparse.Namespace(
        rollouts=root,
        output_dir=tmp_path / "selector",
        benchmark=benchmark,
        exclude_scenes=[],
        cot_labels=None,
        target_n=20,
        clip_step_s=0.1,
        video_offset_s=0.0,
        title="Selector",
    )


def test_baseline_exclusion_and_distinct_frame_counts(args):
    frame(args.rollouts, A, 1)
    frame(args.rollouts, B, 1)
    frame(args.rollouts, B, 2)
    data = build(args)
    assert data["baseline_counts"]["longitudinal:speed_adaptation_road_events"] == 1
    assert len(data["scenes"][0]["frames"]) == 0
    scene = data["scenes"][1]
    assert len(scene["frames"]) == 2
    assert scene["frames"][0]["categories"] == CATEGORIES
    assert scene["frames"][0]["video_time_s"] == 0.1
    assert "reliable" not in json.dumps(data)
    requests = json.loads((args.output_dir / "cot_label_requests.json").read_text())[
        "requests"
    ]
    assert len(requests) == 1 and len(requests[0]["frame_ids"]) == 2
    assert (args.output_dir / "media" / scene["name"]).is_symlink()


def test_unsafe_scene_excluded_even_at_other_timestamp(args, tmp_path):
    frame(args.rollouts, B, 1)
    exclusion = tmp_path / "unsafe.txt"
    exclusion.write_text(B + "_t0900, unsafe")
    args.exclude_scenes = [exclusion]
    assert not build(args)["scenes"][0]["frames"]


def test_unextracted_scenes_stay_visible_and_unclassified_is_not_none(args):
    (args.rollouts / f"scene_2_{B}").mkdir()
    frame(args.rollouts, C, 1, cot="An unfamiliar statement")
    data = build(args)
    assert data["scenes"][0]["status"] == "Needs frame extraction"
    assert data["scenes"][1]["frames"][0]["categories"] is None


def test_bad_candidates_do_not_silently_become_selectable(args):
    frame(args.rollouts, B, 1, cot="")
    frame(args.rollouts, B, 2, plan=False)
    data = build(args)
    assert data["scenes"][0]["skipped_frames"] == 2
    assert not data["scenes"][0]["frames"]


def test_duplicate_scene_is_rejected(args):
    frame(args.rollouts, B, 1)
    (args.rollouts / f"another_{B}").symlink_to(
        args.rollouts / f"scene_2_{B}", target_is_directory=True
    )
    with pytest.raises(ValueError, match="Duplicate source scene"):
        build(args)


def test_duplicate_baseline_scene_is_rejected(args):
    data = json.loads(args.benchmark.read_text())
    data.append({**data[0], "clip_id": A + "_t0002"})
    args.benchmark.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="multiple frames"):
        build(args)


def test_proposals_are_validated_and_override_text_matches(args, tmp_path):
    frame(args.rollouts, B, 1)
    proposal = tmp_path / "labels.json"
    categories = {**CATEGORIES, "longitudinal": "none"}
    proposal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "labels": [
                    {
                        "cot_id": cot_id("Slow for the bend"),
                        "source": "model draft",
                        "categories": categories,
                    }
                ],
            }
        )
    )
    args.cot_labels = proposal
    assert build(args)["scenes"][0]["frames"][0]["categories"] == categories
    data = json.loads(proposal.read_text())
    data["labels"].append(data["labels"][0])
    proposal.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        load_proposals(proposal)
    with pytest.raises(ValueError, match="Unknown"):
        validate_categories({**CATEGORIES, "lateral": "nonsense"})


def test_payload_does_not_allow_script_injection(args):
    frame(args.rollouts, B, 1, cot="</script><script>alert(1)</script>")
    build(args)
    page = (args.output_dir / "index.html").read_text()
    assert "</script><script>alert(1)" not in page
    assert "\\u003c/script>" in page


def test_rebuild_keeps_identity_and_rejects_conflicting_media(args):
    frame(args.rollouts, B, 1)
    first = build(args)
    assert first["dataset_id"] == build(args)["dataset_id"]
    link = args.output_dir / "media" / f"scene_2_{B}"
    link.unlink()
    link.mkdir()
    with pytest.raises(ValueError, match="conflicts"):
        build(args)


def test_selected_example_handoff_preserves_categories_and_provenance(args, tmp_path):
    metadata = frame(args.rollouts, B, 1)
    entry = {
        "clip_id": B + "_t0001",
        "chain_of_thought": "Slow for the bend",
        "source_scene_file": str(metadata),
        "labels": {"cot_decision_label": label()},
        "selection": {
            "reason": "Increase rare category n",
            "method": "manual_category_guided",
        },
    }
    examples = tmp_path / "selected.json"
    examples.write_text(json.dumps({"entries": [entry]}))
    records = build_records([("New", examples)], {}, tmp_path, tmp_path, False)
    record = annotator_record(records[0])
    assert record["decision"]["longitudinal"] == "speed_adaptation_road_events"
    assert record["selection_source"]["selection"] == entry["selection"]
    assert record["selection_source"]["source_scene_file"] == str(metadata)
    assert "released" not in record
