# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import base64
import json
import math
import re
from pathlib import Path

from tools.build_benchmark_viewer import (
    ANNOTATOR_TEMPLATE_PATH,
    ANSWER_FACETS,
    AssetBundle,
    DEFAULT_EXAMPLES,
    FACET_GROUPS,
    TEMPLATE_PATH,
    annotator_facet_groups,
    annotator_record,
    as_list,
    attribute_section,
    attributes_for,
    build_records,
    camera_plan,
    clip_index,
    consistency_justification,
    clip_timing,
    default_output,
    load_attributes,
    evenness,
    heading_band,
    parse_example_arg,
    project_ftheta,
    relative_asset,
    reliability_labels,
    released_answers,
    render,
    speed_band,
    split_attribute_cell,
    trajectory_points,
    video_duration_s,
)


CLIP_ID = "clipgt-test_t0001"


def write_scene(repo_root: Path, clip_id: str = CLIP_ID) -> Path:
    """Minimal on-disk scene folder mirroring data/media/scenes/<clip_id>."""
    scene_dir = repo_root / "data" / "media" / "scenes" / clip_id
    scene_dir.mkdir(parents=True)
    (scene_dir / "camera_front_wide_120fov.jpg").write_bytes(b"jpg")
    (scene_dir / "trajectory_plot.png").write_bytes(b"png")
    (scene_dir / "metadata.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "chain_of_thought": "['fallback cot']",
                "trajectory_xy_rig_frame": [
                    {"rx": 0.0, "ry": 0.0},
                    {"rx": 1.0, "ry": -0.5},
                ],
            }
        )
    )
    (scene_dir / "additional_info.json").write_text(
        json.dumps(
            {
                "summary": {
                    "time_horizon_s": 6.4,
                    "max_speed_kph": 45.0,
                    "heading_change_deg": -50.0,
                    "waypoint_count": 2,
                },
                "meta_actions": {
                    "dominant": {
                        "longitudinal": "gentle_decelerate",
                        "lateral": "steer_right",
                    },
                    "longitudinal_distribution": {"gentle_decelerate": 2},
                    "lateral_distribution": {"steer_right": 2},
                    "transition_count": 1,
                },
                "waypoints": [
                    {"position_m": {"x": 0.0, "y": 0.0}},
                    {"position_m": {"x": 1.0, "y": -0.5}},
                ],
                "segments": [{"velocity_mps": {"speed_kph": 36.0}}],
            }
        )
    )
    (repo_root / "data" / "media" / "videos").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "media" / "videos" / f"{clip_id}.mp4").write_bytes(b"mp4")
    return scene_dir


def example_entry(clip_id: str = CLIP_ID) -> dict:
    return {
        "clip_id": clip_id,
        "chain_of_thought": "['Slow for the red light']",
        "cot_action_consistency": False,
        "cot_action_consistency_justification": ["Trajectory does not stop."],
        "cot_reliability": {
            "grounded": True,
            "safe": False,
            "logically_coherent": True,
            "justification": "Plans an unlawful lane entry.",
        },
        "planned_trajectory_safe": True,
        "planned_trajectory_safety_justification": None,
        "note": "",
        "checkpoint_path": "nvidia/Alpamayo-1.5-10B",
        "source_scene_file": f"data/media/scenes/{clip_id}/metadata.json",
        "rollout_mp4": f"data/media/videos/{clip_id}.mp4",
    }


def benchmark_entry(clip_id: str = CLIP_ID) -> dict:
    return {
        "clip_id": clip_id,
        "labels": {
            "nurec_scene": {
                "scene_uuid": "test-uuid",
                "behavior": ["stop"],
                "layout": ["intersection"],
                "road_types": ["urban"],
                "traffic_density": ["medium"],
                "lighting": ["daytime"],
                "weather": ["clear/cloudy"],
                "surface_conditions": ["dry"],
                "vrus": True,
            },
            "cot_decision_label": {
                "high_level_decision": {
                    "longitudinal": "stop_for_static_constraints",
                    "lateral": "lane_keeping_centering",
                },
                "primary_causal_factor": {"category": "traffic_lights"},
                "critical_component_categories": ["traffic_lights"],
                "critical_component_tags": ["red_light", "intersection"],
                "atomic_meta_action_hint": {
                    "longitudinal": "strong_decelerate",
                    "lateral": "go_straight",
                },
            },
        },
    }


def build_one(tmp_path: Path, **overrides) -> dict:
    write_scene(tmp_path)
    entry = example_entry()
    entry.update(overrides)
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([entry]))
    records = build_records(
        [("Worked", examples)],
        {CLIP_ID: benchmark_entry()},
        tmp_path,
        tmp_path / "data" / "benchmark",
        include_points=True,
    )
    assert len(records) == 1
    return records[0]


def test_record_collects_assets_relative_to_the_output_page(tmp_path: Path) -> None:
    record = build_one(tmp_path)

    assert record["assets"]["frame"] == (
        f"../media/scenes/{CLIP_ID}/camera_front_wide_120fov.jpg"
    )
    assert record["assets"]["trajectory"] == (
        f"../media/scenes/{CLIP_ID}/trajectory_plot.png"
    )
    assert record["assets"]["video"] == f"../media/videos/{CLIP_ID}.mp4"
    assert record["assets"]["scene_dir"] == f"data/media/scenes/{CLIP_ID}"


def test_missing_assets_become_none_rather_than_broken_links(tmp_path: Path) -> None:
    write_scene(tmp_path)
    (tmp_path / "data" / "media" / "videos" / f"{CLIP_ID}.mp4").unlink()
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([example_entry()]))

    record = build_records(
        [("Worked", examples)], {}, tmp_path, tmp_path, include_points=False
    )[0]

    assert record["assets"]["video"] is None
    assert record["assets"]["frame"] is not None
    assert record["points"] == []


def test_record_joins_benchmark_labels_into_filterable_facets(tmp_path: Path) -> None:
    record = build_one(tmp_path)
    facets = record["facets"]

    assert facets["behavior"] == ["stop"]
    assert facets["vrus"] == ["VRUs present"]
    assert facets["causal_factor"] == ["traffic_lights"]
    assert facets["component_tags"] == ["red_light", "intersection"]
    assert facets["meta_longitudinal"] == ["gentle_decelerate"]
    assert facets["speed_band"] == ["40-60 kph"]
    assert facets["turn_band"] == ["turn right"]
    assert facets["consistency"] == ["no"]
    assert facets["reliability"] == ["not safe"]
    assert facets["trajectory_safe"] == ["yes"]
    assert record["in_benchmark"] is True


def test_unanswered_entries_are_labelled_rather_than_dropped(tmp_path: Path) -> None:
    record = build_one(
        tmp_path,
        cot_action_consistency=None,
        cot_action_consistency_justification=[],
        cot_reliability={
            "grounded": None,
            "safe": None,
            "logically_coherent": None,
            "justification": None,
        },
        planned_trajectory_safe=None,
    )

    assert record["facets"]["consistency"] == ["unanswered"]
    assert record["facets"]["reliability"] == ["unanswered"]
    assert record["facets"]["trajectory_safe"] == ["unanswered"]
    assert record["answers"]["consistency_justification"] == []


def test_chain_of_thought_is_flattened_and_falls_back_to_the_scene(
    tmp_path: Path,
) -> None:
    assert build_one(tmp_path)["cot"] == "Slow for the red light"

    record = build_one(tmp_path / "b", chain_of_thought=None)
    assert record["cot"] == "fallback cot"


def test_facets_are_empty_when_no_benchmark_labels_exist(tmp_path: Path) -> None:
    write_scene(tmp_path)
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([example_entry()]))

    record = build_records(
        [("Worked", examples)], {}, tmp_path, tmp_path, include_points=True
    )[0]

    assert record["in_benchmark"] is False
    assert record["facets"]["behavior"] == []
    assert record["facets"]["causal_factor"] == []
    # Trajectory-derived facets come from the scene folder, not the benchmark.
    assert record["facets"]["meta_lateral"] == ["steer_right"]


def test_set_label_and_order_are_preserved(tmp_path: Path) -> None:
    write_scene(tmp_path)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps([example_entry(), example_entry()]))
    second.write_text(json.dumps([example_entry()]))

    records = build_records(
        [("Worked", first), ("Qualification", second)],
        {},
        tmp_path,
        tmp_path,
        include_points=False,
    )

    assert [(r["set"], r["order"]) for r in records] == [
        ("Worked", 0),
        ("Worked", 1),
        ("Qualification", 0),
    ]


def test_trajectory_points_pair_waypoints_with_segment_speeds() -> None:
    points = trajectory_points(
        {
            "waypoints": [
                {"position_m": {"x": 0.0, "y": 0.0}},
                {"position_m": {"x": 2.0, "y": -1.0}},
            ],
            "segments": [{"velocity_mps": {"speed_kph": 30.0}}],
        },
        {},
    )

    assert points == [[0.0, 0.0, 30.0], [2.0, -1.0, 30.0]]


def test_trajectory_points_fall_back_to_rig_frame_poses() -> None:
    points = trajectory_points(
        {}, {"trajectory_xy_rig_frame": [{"rx": 0.0, "ry": 0.0}, {"rx": 1.5, "ry": 0.25}]}
    )

    assert points == [[0.0, 0.0, None], [1.5, 0.25, None]]


def forward_camera(origin_x: float = 1.5, origin_z: float = 1.0) -> dict:
    """An f-theta camera looking along rig +x, as proto JSON with zeros omitted.

    The quaternion turns camera axes (x right, y down, z forward) into rig axes
    (x forward, y left, z up); 1000 px per radian off the optical axis.
    """
    return {
        "logical_id": "camera_front_wide_120fov",
        "intrinsics": {
            "resolution_w": 1920,
            "resolution_h": 1080,
            "ftheta_param": {
                "principal_point_x": 960.0,
                "principal_point_y": 540.0,
                "angle_to_pixeldist_poly": [0.0, 1000.0],
                "max_angle": 1.3,
                "linear_cde": {"linear_c": 1.0},
            },
        },
        "rig_to_camera": {
            "vec": {"x": origin_x, "z": origin_z},
            "quat": {"w": 0.5, "x": -0.5, "y": 0.5, "z": -0.5},
        },
    }


def test_project_ftheta_maps_rig_points_through_the_camera_pose() -> None:
    pixels = project_ftheta(
        forward_camera(),
        [
            [11.5, 0.0, 1.0],  # dead ahead at lens height: the principal point
            [11.5, 0.0, 0.0],  # same distance, on the ground: below it
            [11.5, 1.0, 1.0],  # one metre left: left of it
            [0.0, 0.0, 0.0],  # behind the lens
        ],
    )

    assert pixels[0] == [960.0, 540.0]
    assert pixels[1] == [960.0, round(540.0 + 1000.0 * math.atan(0.1), 1)]
    assert pixels[2][0] < 960.0 and pixels[2][1] == 540.0
    assert pixels[3] is None


def test_camera_plan_puts_the_waypoints_on_the_ground_plane(tmp_path: Path) -> None:
    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps({"cameras": {"camera_front_wide_120fov": forward_camera()}})
    )

    plan = camera_plan(calibration, [[0.0, 0.0, 10.0], [11.5, 0.0, 10.0], [21.5, 0.0, 10.0]])

    assert plan["width"] == 1920 and plan["height"] == 1080
    assert plan["pixels"][0] is None  # the rig origin sits behind the lens
    assert plan["pixels"][1] == [960.0, round(540.0 + 1000.0 * math.atan(0.1), 1)]
    # Further along the ground is closer to the horizon.
    assert plan["pixels"][2][1] < plan["pixels"][1][1]
    # Nothing to draw without a calibration, or with under two visible points.
    assert camera_plan(None, [[11.5, 0.0, None], [21.5, 0.0, None]]) is None
    assert camera_plan(calibration, [[0.0, 0.0, None], [1.0, 0.0, None]]) is None


def test_records_find_the_calibration_above_the_frame_folder(tmp_path: Path) -> None:
    scene_dir = write_scene(tmp_path)
    (scene_dir.parent / "camera_calibration.json").write_text(
        json.dumps({"cameras": {"camera_front_wide_120fov": forward_camera(origin_x=-5.0)}})
    )
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([example_entry()]))

    record = build_records([("Worked", examples)], {}, tmp_path, tmp_path, True)[0]
    without_points = build_records([("Worked", examples)], {}, tmp_path, tmp_path, False)[0]

    assert len(record["camera_plan"]["pixels"]) == len(record["points"])
    assert all(record["camera_plan"]["pixels"])
    assert without_points["camera_plan"] is None


def test_annotator_page_shows_the_camera_overlay_and_the_hd_map() -> None:
    template = ANNOTATOR_TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "function cameraPlanSvg(record)" in template
    assert 'data-role="camplan-toggle"' in template
    assert "HD map at the decision point" in template


def test_reliability_labels_support_both_schemas() -> None:
    flags, facet = reliability_labels(
        {"grounded": True, "safe": True, "logically_coherent": True}
    )
    assert facet == ["reliable"]
    assert flags["grounded"] is True

    _, facet = reliability_labels({"grounded": False, "safe": False})
    assert facet == ["not grounded", "not safe"]

    assert reliability_labels({"reliable": True})[1] == ["reliable"]
    assert reliability_labels(None)[1] == ["unanswered"]
    assert reliability_labels({})[1] == ["unanswered"]


def test_bands_bin_speed_and_heading() -> None:
    assert speed_band(0.0) == "0-20 kph"
    assert speed_band(39.9) == "20-40 kph"
    assert speed_band(60.0) == "60+ kph"
    assert speed_band(None) is None

    assert heading_band(2.0) == "straight"
    assert heading_band(-20.0) == "curve right"
    assert heading_band(75.0) == "turn left"
    assert heading_band(None) is None


def test_as_list_normalises_strings_lists_and_blanks() -> None:
    assert as_list(None) == []
    assert as_list("") == []
    assert as_list("  text  ") == ["text"]
    assert as_list(["a", "", ["b"]]) == ["a", "b"]


def test_evenness_is_one_for_uniform_and_lower_for_skewed() -> None:
    assert evenness([]) is None
    assert evenness([5]) == 1.0
    assert evenness([3, 3, 3]) == 1.0
    assert evenness([9, 1]) < 0.5


def test_parse_example_arg_reads_optional_label(tmp_path: Path) -> None:
    label, path = parse_example_arg("Worked=data/examples.json", tmp_path)
    assert label == "Worked"
    assert path == tmp_path / "data" / "examples.json"

    label, path = parse_example_arg("data/worked_examples.json", tmp_path)
    assert label == "worked examples"
    assert path == tmp_path / "data" / "worked_examples.json"


def test_render_fills_the_template_with_parseable_json(tmp_path: Path) -> None:
    record = build_one(tmp_path)
    page = render(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        "Example sets",
        [record],
        [{"label": "Worked", "path": "examples.json", "count": 1}],
    )

    assert "__VIEWER_DATA__" not in page
    assert "__PAGE_TITLE__" not in page
    assert "<title>Example sets</title>" in page

    embedded = re.search(
        r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S
    )
    payload = json.loads(embedded.group(1))
    assert payload["records"][0]["clip_id"] == CLIP_ID
    assert payload["facet_groups"] == FACET_GROUPS
    assert [source["count"] for source in payload["sources"]] == [1]


def test_render_escapes_markup_that_would_close_the_data_block(tmp_path: Path) -> None:
    record = build_one(tmp_path, note="</script><script>alert(1)</script>")
    page = render(
        TEMPLATE_PATH.read_text(encoding="utf-8"), "t", [record], []
    )

    embedded = re.search(
        r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S
    )
    payload = json.loads(embedded.group(1))
    assert payload["records"][0]["answers"]["note"] == [
        "</script><script>alert(1)</script>"
    ]


def write_attributes_csv(tmp_path: Path, key_column: str = "clip_id") -> Path:
    path = tmp_path / "attributes.csv"
    path.write_text(
        f"{key_column},source,signals,note_id,blank\n"
        f"{CLIP_ID if key_column == 'clip_id' else CLIP_ID.rsplit('_t', 1)[0]},"
        "cascade_eventful,night|VRU,row-1,\n"
        "other-clip,legacy_taxonomy,stop;intersection,row-2,\n"
        "third-clip,legacy_taxonomy,stop,row-3,\n"
        "fourth-clip,cascade_eventful,night,row-4,\n"
    )
    return path


def test_attributes_csv_becomes_facets_keyed_by_clip_id(tmp_path: Path) -> None:
    key_column, table = load_attributes(write_attributes_csv(tmp_path))

    assert key_column == "clip_id"
    assert table[CLIP_ID]["source"] == ["cascade_eventful"]
    assert table[CLIP_ID]["signals"] == ["night", "VRU"]
    # A column that is unique per row carries no grouping information, and an
    # empty column has nothing to group by.
    assert "note_id" not in table[CLIP_ID]
    assert "blank" not in table[CLIP_ID]


def test_attributes_fall_back_to_the_scene_level_clip_id(tmp_path: Path) -> None:
    _, table = load_attributes(write_attributes_csv(tmp_path, key_column="scene_id"))

    assert attributes_for(table, CLIP_ID)["source"] == ["cascade_eventful"]
    assert attributes_for(table, "unknown-clip") == {}


def test_attribute_cells_split_on_both_separators() -> None:
    assert split_attribute_cell("night|VRU; stop ") == ["night", "VRU", "stop"]
    assert split_attribute_cell("  ") == []


def test_records_carry_attribute_facets_under_a_namespaced_key(tmp_path: Path) -> None:
    write_scene(tmp_path)
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([example_entry()]))
    _, table = load_attributes(write_attributes_csv(tmp_path))

    record = build_records(
        [("OOD", examples)],
        {},
        tmp_path,
        tmp_path,
        include_points=False,
        attribute_tables=[table],
    )[0]

    assert record["facets"]["attr:signals"] == ["night", "VRU"]
    assert record["facets"]["attr:source"] == ["cascade_eventful"]
    # Built-in facets are untouched by a CSV column of the same name.
    assert record["facets"]["consistency"] == ["no"]


def test_attribute_section_labels_its_columns() -> None:
    section = attribute_section("Suite provenance", ["source", "selection_signals"])

    assert section["section"] == "Suite provenance"
    assert section["groups"] == [
        {"key": "attr:source", "label": "Source"},
        {"key": "attr:selection_signals", "label": "Selection signals"},
    ]


def test_render_accepts_run_specific_facet_groups(tmp_path: Path) -> None:
    record = build_one(tmp_path)
    groups = [attribute_section("Suite provenance", ["source"])]
    page = render(
        TEMPLATE_PATH.read_text(encoding="utf-8"), "t", [record], [], groups
    )

    embedded = re.search(
        r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S
    )
    assert json.loads(embedded.group(1))["facet_groups"] == groups


def test_default_output_never_clobbers_the_default_viewer(tmp_path: Path) -> None:
    assert default_output(list(DEFAULT_EXAMPLES), tmp_path) == (
        tmp_path / "data" / "benchmark" / "viewer.html"
    )

    other = [("OOD", tmp_path / "data" / "benchmark" / "ood_traces.json")]
    assert default_output(other, tmp_path) == (
        tmp_path / "data" / "benchmark" / "ood_traces.viewer.html"
    )


def test_clip_index_reads_the_rollout_step_suffix() -> None:
    assert clip_index("clipgt-x_t0100") == 100
    assert clip_index("clipgt-x_t0007") == 7
    assert clip_index("clipgt-x") is None
    assert clip_index("") is None


def test_clip_timing_places_the_clip_on_the_rollout_video() -> None:
    timing = clip_timing("clipgt-x_t0100", {"time_horizon_s": 6.4}, None)

    assert timing["index"] == 100
    assert timing["start_s"] == 10.0
    assert timing["horizon_s"] == 6.4
    assert timing["end_s"] == 16.4
    assert timing["video_duration_s"] is None


def test_clip_timing_survives_a_missing_index_or_horizon() -> None:
    timing = clip_timing("clipgt-x", {}, None)
    assert timing["start_s"] is None
    assert timing["end_s"] is None

    timing = clip_timing("clipgt-x_t0050", {}, None, step_s=0.2)
    assert timing["start_s"] == 10.0
    assert timing["end_s"] is None


def test_video_duration_reads_the_mp4_header(tmp_path: Path) -> None:
    # A bare moov/mvhd box is all the duration probe looks at.
    mvhd = b"mvhd" + bytes(4) + bytes(8) + (1000).to_bytes(4, "big") + (20100).to_bytes(4, "big")
    mvhd = (len(mvhd) + 4).to_bytes(4, "big") + mvhd
    moov = (len(mvhd) + 8).to_bytes(4, "big") + b"moov" + mvhd
    path = tmp_path / "rollout.mp4"
    path.write_bytes(b"\x00\x00\x00\x08ftyp" + moov)

    assert video_duration_s(path) == 20.1
    assert video_duration_s(tmp_path / "absent.mp4") is None

    path.write_bytes(b"not an mp4")
    assert video_duration_s(path) is None


def test_records_carry_the_timing_of_their_decision_point(tmp_path: Path) -> None:
    record = build_one(tmp_path)

    assert record["timing"]["index"] == 1
    assert record["timing"]["start_s"] == 0.1
    assert record["timing"]["horizon_s"] == 6.4
    assert record["timing"]["end_s"] == 6.5


def test_annotator_record_hides_the_answers_and_drops_the_note(tmp_path: Path) -> None:
    record = annotator_record(build_one(tmp_path, note="private aside"))

    assert "answers" not in record
    for key in ANSWER_FACETS:
        assert key not in record["facets"]
    # Scene facets are evidence, not answers, so they stay.
    assert record["facets"]["behavior"] == ["stop"]

    revealed = json.loads(base64.b64decode(record["reference"]).decode("utf-8"))
    assert revealed["consistency"] is False
    assert revealed["consistency_justification"] == ["Trajectory does not stop."]
    assert revealed["reliability_flags"]["safe"] is False
    assert "note" not in revealed


def test_consistency_justification_reads_either_key() -> None:
    long_key = "cot_action_consistency_justification"

    assert consistency_justification({long_key: ["from the set"]}) == (
        long_key,
        ["from the set"],
    )
    # The released benchmark keeps the same reason under the short key.
    assert consistency_justification({"justification": ["from the benchmark"]}) == (
        "justification",
        ["from the benchmark"],
    )
    assert consistency_justification({}) == (long_key, [])


def test_review_record_keeps_the_answers_in_the_clear(tmp_path: Path) -> None:
    record = annotator_record(build_one(tmp_path, note="private aside"), prefill=True)

    assert "answers" not in record
    assert "reference" not in record
    # Nothing is hidden in a review pass, so the answer facets stay filterable.
    assert record["facets"]["consistency"] == ["no"]

    released = record["released"]
    assert released["consistency"] is False
    assert released["consistency_justification"] == ["Trajectory does not stop."]
    assert (released["grounded"], released["safe"], released["coherent"]) == (
        True,
        False,
        True,
    )
    assert released["reliability_schema"] == "flags"
    assert released["consistency_justification_key"] == (
        "cot_action_consistency_justification"
    )
    assert "note" not in released


def test_released_answers_derive_the_flags_from_the_benchmark_categories() -> None:
    released = released_answers(
        {
            "consistency": True,
            "consistency_justification": ["Accelerates as stated."],
            "reliability_flags": {"reliable": False},
            "reliability_justification": ["Sees a truck that is not there."],
            "trajectory_safe": True,
            "safety_justification": [],
            "source": {
                "consistency_justification_key": "justification",
                "cot_reliability": {
                    "reliable": False,
                    "primary_unreliability_category": "logical_coherence_error",
                    "unreliability_categories": [
                        "logical_coherence_error",
                        "scene_entity_hallucination",
                    ],
                },
            },
        }
    )

    # One category per flag: the two named ones failed, the third did not.
    assert released["coherent"] is False
    assert released["grounded"] is False
    assert released["safe"] is True
    assert released["reliability_schema"] == "reliable"
    assert released["consistency_justification_key"] == "justification"
    # Order and primary are carried so an untouched clip exports unchanged.
    assert released["primary_unreliability_category"] == "logical_coherence_error"
    assert released["unreliability_categories"] == [
        "logical_coherence_error",
        "scene_entity_hallucination",
    ]


def test_released_answers_leave_an_unanswered_entry_unanswered() -> None:
    released = released_answers({"reliability_flags": {}, "source": {}})

    assert released["consistency"] is None
    assert (released["grounded"], released["safe"], released["coherent"]) == (
        None,
        None,
        None,
    )
    assert released["reliability_schema"] == "flags"


def test_annotator_facet_groups_drop_the_answers_panel() -> None:
    groups = annotator_facet_groups(FACET_GROUPS)

    keys = [group["key"] for section in groups for group in section["groups"]]
    assert not set(keys) & set(ANSWER_FACETS)
    assert "behavior" in keys
    assert [section["section"] for section in groups] != [
        section["section"] for section in FACET_GROUPS
    ]


def test_annotator_page_embeds_the_records_without_plain_answers(tmp_path: Path) -> None:
    record = annotator_record(build_one(tmp_path))
    page = render(
        ANNOTATOR_TEMPLATE_PATH.read_text(encoding="utf-8"),
        "Annotation pass",
        [record],
        [{"label": "Worked", "path": "examples.json", "count": 1}],
        annotator_facet_groups(FACET_GROUPS),
        {"storage_key": "examples.json"},
    )

    assert "__VIEWER_DATA__" not in page
    assert "<title>Annotation pass</title>" in page
    assert "Trajectory does not stop." not in page

    embedded = re.search(
        r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S
    )
    payload = json.loads(embedded.group(1))
    assert payload["storage_key"] == "examples.json"
    assert payload["records"][0]["timing"]["start_s"] == 0.1


def test_review_page_carries_the_answers_and_the_waypoint_speeds(
    tmp_path: Path,
) -> None:
    record = annotator_record(build_one(tmp_path), prefill=True)
    page = render(
        ANNOTATOR_TEMPLATE_PATH.read_text(encoding="utf-8"),
        "Review pass",
        [record],
        [{"label": "Benchmark", "path": "benchmark.json", "count": 1}],
        FACET_GROUPS,
        {"storage_key": "benchmark.json:review", "prefill": True},
    )

    embedded = re.search(
        r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S
    )
    payload = json.loads(embedded.group(1))
    assert payload["prefill"] is True
    assert payload["storage_key"] == "benchmark.json:review"
    assert payload["records"][0]["released"]["consistency"] is False
    # The colour on the plan is the segment speed, so it has to travel with it.
    assert payload["records"][0]["points"] == [[0.0, 0.0, 36.0], [1.0, -0.5, 36.0]]

    # Both resets ship, and neither throws work away without asking first.
    assert 'id="reset-all"' in page
    assert "Reset this clip to the released answer" in page
    assert "Reset this clip back to the answer it was released" in page
    assert "discards every edit in this browser" in page


def test_bundle_copies_media_next_to_the_page(tmp_path: Path) -> None:
    write_scene(tmp_path)
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([example_entry()]))
    output_dir = tmp_path / "handover"
    output_dir.mkdir()
    bundle = AssetBundle(output_dir / "media", output_dir)

    record = build_records(
        [("Worked", examples)], {}, tmp_path, output_dir, False, bundle=bundle
    )[0]

    assert record["assets"]["frame"] == f"media/{CLIP_ID}/camera_front_wide_120fov.jpg"
    assert record["assets"]["video"] == f"media/{CLIP_ID}/{CLIP_ID}.mp4"
    assert (output_dir / record["assets"]["frame"]).read_bytes() == b"jpg"
    assert (output_dir / record["assets"]["video"]).read_bytes() == b"mp4"
    # A file referenced twice is copied once.
    assert bundle.add(tmp_path / "data" / "media" / "videos" / f"{CLIP_ID}.mp4", CLIP_ID) == (
        record["assets"]["video"]
    )
    assert len(bundle.copied) == 3


def test_default_output_names_the_annotator_build_apart(tmp_path: Path) -> None:
    other = [("OOD", tmp_path / "data" / "benchmark" / "ood_traces.json")]

    assert default_output(other, tmp_path, ".annotator.html") == (
        tmp_path / "data" / "benchmark" / "ood_traces.annotator.html"
    )
    assert default_output(list(DEFAULT_EXAMPLES), tmp_path, ".annotator.html") == (
        tmp_path / "data" / "benchmark" / "annotator.html"
    )
    assert default_output(other, tmp_path, ".review.html") == (
        tmp_path / "data" / "benchmark" / "ood_traces.review.html"
    )
