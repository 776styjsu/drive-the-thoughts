#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Build a single-file HTML viewer for benchmark example sets.

Joins each example JSON (e.g. the worked and qualification example sets) with
the released benchmark labels and the per-clip scene folder, then writes one
self-contained HTML page that shows, for every clip:

- the original front-wide camera frame and the trajectory plot,
- the chain of thought and any recorded answers/justifications,
- the evidence behind the entry: NuRec scene attributes, the CoT decision
  label (Alpamayo-R1 tables), the executed trajectory kinematics, and the
  per-segment meta-action distribution,
- the rollout video, when present on disk.

The page opens with a faceted diversity panel (counts, distinct values and
evenness per attribute, split by example set) whose chips also filter the
cards, so a set can be checked for coverage and skew at a glance. Facets with
no values are dropped, so a set carrying fewer labels simply shows a shorter
panel.

Any file ``benchmark_analysis.extract_entries`` can read works as an example
set; the worked/qualification pair is only the default. Sets outside
``benchmark.json`` have no NuRec or decision labels, so pass ``--attributes``
with a CSV keyed by ``clip_id`` or ``scene_id`` to give them facets of their
own.

``--annotator`` builds the version to hand to someone else: the same evidence
per clip, but the recorded answers are hidden behind a per-clip reveal, the
note field is dropped, each clip states exactly where on the rollout video its
decision point falls, and the annotator records their own verdicts and
justifications, exported as JSON in the example-set schema.

``--prefill`` builds the review variant of that page: the recorded answers are
in the form from the start, editable in place, and the export says which clips
were changed and what they were changed from. Nothing is hidden there — the
answer facets stay in the filter panel — because the page is for checking
answers rather than producing them independently.

Images and videos are referenced by relative path rather than inlined, so the
output must stay inside the repository next to the media it points at —
unless ``--bundle`` is given, which copies the media into the output folder so
the folder can be zipped or served on its own.

Examples:
    # The default pair, into data/benchmark/viewer.html
    uv run python tools/build_benchmark_viewer.py

    # Any other set; the output path defaults to <examples>.viewer.html
    uv run python tools/build_benchmark_viewer.py \
        --examples data/benchmark/ood_hazard_2604_unsafe_traces.json \
        --attributes alpasim/data/scenes/sim_scenes_ood_hazard_2604_provenance.csv \
        --title "OOD hazard 2604 unsafe traces"

    # A self-contained folder to send to an annotator
    uv run python tools/build_benchmark_viewer.py --annotator \
        --examples data/benchmark/ood_hazard_2604_unsafe_traces.json \
        --bundle data/benchmark/annotation/ood_hazard_2604 \
        --title "OOD hazard 2604 — annotation pass"

    # The released 150, answers in place and editable, as a review bundle
    uv run python tools/build_benchmark_viewer.py --prefill \
        --examples "Benchmark=data/benchmark/benchmark.json" \
        --bundle data/benchmark/annotation/benchmark_150_review \
        --title "Benchmark 150 — review pass"
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmark_analysis import clean_cot, extract_entries, index_by_clip_id, load_json
from benchmark_analysis.schema import cot_decision_label, nurec_scene

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path(__file__).resolve().parent / "benchmark_viewer_template.html"
ANNOTATOR_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "benchmark_annotator_template.html"
)

DEFAULT_EXAMPLES = (
    ("Worked", REPO_ROOT / "data" / "benchmark" / "worked_examples_with_answers.json"),
    (
        "Qualification",
        REPO_ROOT / "data" / "benchmark" / "qualification_examples_unanswered.json",
    ),
)

FRAME_CAMERA = "camera_front_wide_120fov"
FRAME_NAME = f"{FRAME_CAMERA}.jpg"
TRAJECTORY_PLOT_NAME = "trajectory_plot.png"
#: Per-rollout calibration written by ``alpasim_utils.extract_frame`` into its
#: output directory, i.e. the frame folder's parent (or further up).
CAMERA_CALIBRATION_NAME = "camera_calibration.json"

#: Seconds between two consecutive ``_t<index>`` clips of the same rollout, and
#: so also the slice of each plan that is driven before the model re-plans.
#: Clip ``_t0100`` is therefore the rollout state 10.0 s after the video starts;
#: measured against the rollout metric timestamps this lands within one frame.
DEFAULT_CLIP_STEP_S = 0.1

SCENE_FACETS = (
    ("behavior", "Behavior"),
    ("layout", "Layout"),
    ("road_types", "Road type"),
    ("traffic_density", "Traffic density"),
    ("lighting", "Lighting"),
    ("weather", "Weather"),
    ("surface_conditions", "Surface"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single-file HTML viewer showing the frame, trajectory and "
            "evidence for every clip in one or more benchmark example sets."
        )
    )
    parser.add_argument(
        "--examples",
        action="append",
        metavar="[LABEL=]PATH",
        help=(
            "Example set to include, optionally prefixed with a display label. "
            "Repeatable. Default: the worked and qualification example sets."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=REPO_ROOT / "data" / "benchmark" / "benchmark.json",
        help=(
            "Released benchmark JSON used to look up NuRec scene attributes and "
            "CoT decision labels per clip. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--attributes",
        action="append",
        metavar="[SECTION=]PATH.csv",
        help=(
            "CSV of per-clip attributes to add as facets, keyed by a clip_id or "
            "scene_id column. Every other column becomes a facet group; cells may "
            "hold several values separated by '|' or ';'. Columns that are empty "
            "or unique per row (ids, paths, timestamps), and numeric measurement "
            "columns, are skipped. Repeatable."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "HTML file to write. Defaults to data/benchmark/viewer.html for the "
            "default example sets, else <first --examples path>.viewer.html."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to resolve relative scene/video paths. Default: %(default)s",
    )
    parser.add_argument(
        "--title",
        default="Benchmark example sets",
        help="Page title. Default: %(default)s",
    )
    parser.add_argument(
        "--no-trajectory-points",
        action="store_true",
        help="Skip embedding trajectory waypoints (drops the speed-coloured plot).",
    )
    parser.add_argument(
        "--annotator",
        action="store_true",
        help=(
            "Build the annotation version instead: recorded answers are hidden "
            "behind a per-clip reveal, the note field is dropped, and each clip "
            "gets verdict buttons and justification boxes whose results export "
            "as JSON in the example-set schema."
        ),
    )
    parser.add_argument(
        "--prefill",
        action="store_true",
        help=(
            "Review build: seed every clip with the answer already recorded "
            "for it, editable in place, and mark in the export which clips "
            "were changed and from what. Implies --annotator."
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help=(
            "Write a self-contained folder instead of a loose page: index.html "
            "plus a copy of every frame, plot and video under media/. Zip it or "
            "serve it with 'python -m http.server' to hand it to someone else."
        ),
    )
    parser.add_argument(
        "--clip-step-s",
        type=float,
        default=DEFAULT_CLIP_STEP_S,
        help=(
            "Seconds between consecutive '_t<index>' clips of one rollout, used "
            "to place a clip on the rollout video timeline. Default: %(default)s"
        ),
    )
    return parser.parse_args()


def parse_example_arg(raw: str, repo_root: Path) -> tuple[str, Path]:
    """Split a ``[LABEL=]PATH`` argument, defaulting the label to the stem."""
    label, sep, path_text = raw.partition("=")
    if not sep:
        path = Path(raw)
        label = path.stem.replace("_", " ")
    else:
        path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return label.strip(), path


def resolve(repo_root: Path, value: Any) -> Path | None:
    """Resolve a repo-relative path recorded in an entry."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def relative_asset(path: Path | None, output_dir: Path) -> str | None:
    """Path usable as an ``src`` from the generated page, or None if missing."""
    if path is None or not path.exists():
        return None
    return os.path.relpath(path, output_dir).replace(os.sep, "/")


class AssetBundle:
    """Copies referenced media next to the page, so the folder ships standalone.

    The default build points at media elsewhere in the repository, which is
    fine locally but useless once the page is mailed or hosted on its own. A
    bundle copies every frame, plot and video it is asked for into
    ``<directory>/<clip_id>/`` and hands back the page-relative path, so the
    output directory can be zipped or served as-is.
    """

    def __init__(self, directory: Path, output_dir: Path) -> None:
        self.directory = directory
        self.output_dir = output_dir
        self.copied: dict[Path, str] = {}
        self.bytes = 0

    def add(self, path: Path, clip_id: str) -> str:
        source = path.resolve()
        if source in self.copied:
            return self.copied[source]
        target_dir = self.directory / (clip_id or "shared")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        size = source.stat().st_size
        if not target.exists() or target.stat().st_size != size:
            shutil.copy2(source, target)
        reference = os.path.relpath(target, self.output_dir).replace(os.sep, "/")
        self.copied[source] = reference
        self.bytes += size
        return reference


def asset_reference(
    path: Path | None,
    output_dir: Path,
    bundle: AssetBundle | None = None,
    clip_id: str = "",
) -> str | None:
    """Page-relative ``src`` for a media file, copied into ``bundle`` if given."""
    if path is None or not path.exists():
        return None
    if bundle is not None:
        return bundle.add(path, clip_id)
    return relative_asset(path, output_dir)


def as_list(value: Any) -> list[str]:
    """Normalise a justification/note field into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(as_list(item))
        return out
    return [str(value)]


def string_list(value: Any) -> list[str]:
    """NuRec attributes are lists of strings but tolerate a bare string."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


#: Columns probed, in order, for the clip key inside an attributes CSV.
ATTRIBUTE_KEY_COLUMNS = ("clip_id", "scene_id", "clip", "uuid")

#: Separators allowed inside one attributes cell holding several values.
ATTRIBUTE_VALUE_SEPARATORS = ("|", ";")


#: The ``_t<index>`` suffix naming which rollout step a clip was cut at.
CLIP_INDEX_RE = re.compile(r"_t(\d+)$")


def base_clip_id(clip_id: str) -> str:
    """``clipgt-x_t0100`` -> ``clipgt-x``, so scene-level tables still match."""
    return CLIP_INDEX_RE.sub("", clip_id)


def clip_index(clip_id: str) -> int | None:
    """Rollout step a clip was cut at, from its ``_t<index>`` suffix."""
    match = CLIP_INDEX_RE.search(clip_id or "")
    return int(match.group(1)) if match else None


def iter_boxes(handle, end: int):
    """Yield ``(kind, body_start, body_length)`` for the MP4 boxes up to ``end``."""
    while handle.tell() < end:
        header = handle.read(8)
        if len(header) < 8:
            return
        size = int.from_bytes(header[:4], "big")
        kind = header[4:8]
        if size == 1:  # 64-bit size, stored right after the type
            size = int.from_bytes(handle.read(8), "big")
            start = handle.tell()
            body = size - 16
        elif size == 0:  # box runs to the end of its parent
            start = handle.tell()
            body = end - start
        else:
            start = handle.tell()
            body = size - 8
        if body < 0:
            return
        yield kind, start, body
        handle.seek(start + body)


def video_duration_s(path: Path | None) -> float | None:
    """Duration of an MP4 from its ``mvhd`` header, without an ffprobe binary."""
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(0)
            for kind, start, body in iter_boxes(handle, end):
                if kind != b"moov":
                    continue
                handle.seek(start)
                for inner, inner_start, _ in iter_boxes(handle, start + body):
                    if inner != b"mvhd":
                        continue
                    handle.seek(inner_start)
                    version = handle.read(4)[0]  # version byte plus three flags
                    handle.read(16 if version == 1 else 8)  # creation, modification
                    timescale = int.from_bytes(handle.read(4), "big")
                    duration = int.from_bytes(handle.read(8 if version == 1 else 4), "big")
                    return round(duration / timescale, 3) if timescale else None
    except OSError:
        return None
    return None


def clip_timing(
    clip_id: str,
    summary: dict,
    video_path: Path | None,
    step_s: float = DEFAULT_CLIP_STEP_S,
) -> dict:
    """Where in the rollout video a clip sits, and how long its plan claims.

    The clip index counts rollout steps, so ``_t0100`` is the state ``100 *
    step_s`` seconds into the video, exact to about one video frame — enough to
    seek an annotator to the right moment.

    ``horizon_s`` and ``end_s`` describe the *plan*, not the video. The rollout
    is closed-loop and re-plans at every step, so only the first ``step_s`` of a
    ``time_horizon_s`` plan is ever driven before a fresh chain of thought
    replaces it; consecutive clips of one rollout routinely state unrelated
    intents. Video past ``start_s`` is therefore what the rollout did next, not
    this plan being carried out, and anything presenting ``end_s`` must say so.
    """
    index = clip_index(clip_id)
    horizon = summary.get("time_horizon_s")
    horizon = float(horizon) if isinstance(horizon, (int, float)) else None
    start = round(index * step_s, 3) if index is not None else None
    end = round(start + horizon, 3) if start is not None and horizon else None
    return {
        "index": index,
        "step_s": step_s,
        "start_s": start,
        "horizon_s": horizon,
        "end_s": end,
        "video_duration_s": video_duration_s(video_path),
    }


def split_attribute_cell(cell: str) -> list[str]:
    values = [cell]
    for separator in ATTRIBUTE_VALUE_SEPARATORS:
        values = [part for value in values for part in value.split(separator)]
    return [value.strip() for value in values if value.strip()]


def is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def load_attributes(path: Path) -> tuple[str, dict[str, dict[str, list[str]]]]:
    """Read an attributes CSV into ``{clip key: {column: values}}``.

    The key column is the first of :data:`ATTRIBUTE_KEY_COLUMNS` present, else
    the first column. Columns that are empty everywhere, whose values are never
    shared by two clips (ids, paths, timestamps), or that hold many distinct
    numbers (measurements) carry no grouping information and are dropped.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "", {}

    columns = [name for name in (rows[0].keys() or []) if name]
    key_column = next(
        (name for name in ATTRIBUTE_KEY_COLUMNS if name in columns), columns[0]
    )
    facet_columns = []
    for column in columns:
        if column == key_column:
            continue
        # Count split values, not raw cells: a multi-value column is a useful
        # facet even when every cell holds a different combination.
        seen: Counter[str] = Counter()
        for row in rows:
            seen.update(set(split_attribute_cell(str(row.get(column) or ""))))
        if not seen:
            continue
        if len(rows) > 3 and max(seen.values()) == 1:
            continue  # identifier-like: no value is shared by two clips
        if len(seen) > 6 and all(is_number(value) for value in seen):
            continue  # a measurement, not a category
        facet_columns.append(column)

    table: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        key = str(row.get(key_column) or "").strip()
        if not key:
            continue
        table[key] = {
            column: split_attribute_cell(str(row.get(column) or ""))
            for column in facet_columns
        }
    return key_column, table


def attributes_for(table: dict[str, dict[str, list[str]]], clip_id: str) -> dict:
    """Attributes for a clip, falling back to its scene-level (base) id."""
    return table.get(clip_id) or table.get(base_clip_id(clip_id)) or {}


def scene_paths(repo_root: Path, entry: dict) -> Path | None:
    """Directory holding the extracted frame files for an entry."""
    scene_file = resolve(repo_root, entry.get("source_scene_file"))
    if scene_file is not None and scene_file.parent.is_dir():
        return scene_file.parent
    clip_id = str(entry.get("clip_id") or "").strip()
    if clip_id:
        guess = repo_root / "data" / "media" / "scenes" / clip_id
        if guess.is_dir():
            return guess
    return None


def read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def speed_band(max_speed_kph: float | None) -> str | None:
    if max_speed_kph is None:
        return None
    if max_speed_kph < 20:
        return "0-20 kph"
    if max_speed_kph < 40:
        return "20-40 kph"
    if max_speed_kph < 60:
        return "40-60 kph"
    return "60+ kph"


def heading_band(heading_change_deg: float | None) -> str | None:
    """Coarse turn class, so straight/curve/turn coverage is visible."""
    if heading_change_deg is None:
        return None
    magnitude = abs(heading_change_deg)
    side = "left" if heading_change_deg > 0 else "right"
    if magnitude < 10:
        return "straight"
    if magnitude < 45:
        return f"curve {side}"
    return f"turn {side}"


def answer_label(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unanswered"


#: Where an entry keeps the reason behind its consistency verdict: the example
#: sets use the long key, the released benchmark the short one.
CONSISTENCY_JUSTIFICATION_KEYS = (
    "cot_action_consistency_justification",
    "justification",
)


def consistency_justification(entry: dict) -> tuple[str, list[str]]:
    """The consistency reason, and the key the entry kept it under."""
    for key in CONSISTENCY_JUSTIFICATION_KEYS:
        values = as_list(entry.get(key))
        if values:
            return key, values
    return CONSISTENCY_JUSTIFICATION_KEYS[0], []


def reliability_labels(reliability: Any) -> tuple[dict[str, Any], list[str]]:
    """Reliability sub-flags plus the facet values naming any failed flag."""
    if not isinstance(reliability, dict):
        return {}, ["unanswered"]
    flags = {
        key: reliability.get(key)
        for key in ("grounded", "safe", "logically_coherent", "reliable")
        if key in reliability
    }
    failed = [key for key, value in flags.items() if value is False]
    if failed:
        return flags, [f"not {key}" for key in failed]
    if any(value is True for value in flags.values()):
        return flags, ["reliable"]
    return flags, ["unanswered"]


def trajectory_points(additional_info: dict, metadata: dict) -> list[list[float]]:
    """Waypoints as ``[forward_m, left_m, speed_kph]`` in the ego rig frame."""
    waypoints = additional_info.get("waypoints")
    segments = additional_info.get("segments")
    points: list[list[float]] = []
    if isinstance(waypoints, list) and waypoints:
        speeds: list[float | None] = []
        if isinstance(segments, list):
            speeds = [
                (segment.get("velocity_mps") or {}).get("speed_kph")
                if isinstance(segment, dict)
                else None
                for segment in segments
            ]
        for index, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, dict):
                continue
            position = waypoint.get("position_m") or {}
            x, y = position.get("x"), position.get("y")
            if x is None or y is None:
                continue
            speed = None
            if speeds:
                speed = speeds[min(index, len(speeds) - 1)]
            points.append([round(float(x), 3), round(float(y), 3), speed])
        if points:
            return points
    rig = metadata.get("trajectory_xy_rig_frame")
    if isinstance(rig, list):
        for pose in rig:
            if isinstance(pose, dict) and pose.get("rx") is not None:
                points.append(
                    [round(float(pose["rx"]), 3), round(float(pose.get("ry") or 0.0), 3), None]
                )
    return points


def find_camera_calibration(scene_dir: Path | None, repo_root: Path) -> Path | None:
    """The nearest ``camera_calibration.json`` at or above a frame folder."""
    if scene_dir is None:
        return None
    for folder in (scene_dir, *scene_dir.parents):
        candidate = folder / CAMERA_CALIBRATION_NAME
        if candidate.is_file():
            return candidate
        if folder == repo_root:
            break
    return None


def quaternion_matrix(quat: dict) -> list[list[float]]:
    """Rotation matrix of a ``{w, x, y, z}`` quaternion (proto JSON omits zeros)."""
    w, x, y, z = (float(quat.get(key, 0.0)) for key in ("w", "x", "y", "z"))
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def project_ftheta(camera: dict, points_rig: list[list[float]]) -> list[list[float] | None]:
    """Pixel position of each rig-frame ``[x, y, z]`` point, None when unseen.

    Mirrors ``eval.data.CameraProjector`` so the overlay lands where the eval
    video draws it: ``rig_to_camera`` is the camera pose in the rig frame, so
    a point is moved into the camera frame by its inverse, then mapped through
    the f-theta angle -> pixel-distance polynomial. Points behind the camera
    or past ``max_angle`` are None.
    """
    intrinsics = camera.get("intrinsics") or {}
    ftheta = intrinsics.get("ftheta_param")
    if not ftheta or not ftheta.get("angle_to_pixeldist_poly"):
        return [None for _ in points_rig]
    poly = [float(c) for c in ftheta["angle_to_pixeldist_poly"]]
    cx = float(ftheta.get("principal_point_x", 0.0))
    cy = float(ftheta.get("principal_point_y", 0.0))
    max_angle = float(ftheta.get("max_angle", 0.0)) or None
    linear = ftheta.get("linear_cde")
    c, d, e = (
        (float(linear.get(k, 0.0)) for k in ("linear_c", "linear_d", "linear_e"))
        if isinstance(linear, dict)
        else (1.0, 0.0, 0.0)
    )

    pose = camera.get("rig_to_camera") or {}
    rotation = quaternion_matrix(pose.get("quat") or {})
    vec = pose.get("vec") or {}
    origin = [float(vec.get(key, 0.0)) for key in ("x", "y", "z")]

    pixels: list[list[float] | None] = []
    for point in points_rig:
        rel = [point[i] - origin[i] for i in range(3)]
        # R^T @ rel: the camera pose's rotation, undone.
        x, y, z = (sum(rotation[row][col] * rel[row] for row in range(3)) for col in range(3))
        if z <= 1e-3:
            pixels.append(None)
            continue
        radial = math.hypot(x, y)
        theta = math.atan2(radial, z)
        if max_angle is not None and theta > max_angle + 1e-6:
            pixels.append(None)
            continue
        distance = sum(coef * theta**power for power, coef in enumerate(poly))
        scale = distance / radial if radial > 1e-9 else 0.0
        ox, oy = x * scale, y * scale
        pixels.append([round(c * ox + d * oy + cx, 1), round(e * ox + oy + cy, 1)])
    return pixels


def camera_plan(
    calibration_path: Path | None,
    points: list[list[float]],
    camera_id: str = FRAME_CAMERA,
) -> dict | None:
    """The plan's waypoints projected into one camera image, for an overlay.

    Waypoints are ``[forward_m, left_m, ...]`` in the ego rig frame at the
    decision point, and the plan is flat (its poses keep the ego's height), so
    every waypoint sits on the rig ground plane, z = 0. Pixels index the
    calibrated resolution; the page scales them with the image.
    """
    if calibration_path is None or len(points) < 2:
        return None
    camera = (read_json(calibration_path).get("cameras") or {}).get(camera_id)
    if not isinstance(camera, dict):
        return None
    intrinsics = camera.get("intrinsics") or {}
    width = int(intrinsics.get("resolution_w") or 0)
    height = int(intrinsics.get("resolution_h") or 0)
    pixels = project_ftheta(camera, [[p[0], p[1], 0.0] for p in points])
    if not width or not height or sum(1 for p in pixels if p) < 2:
        return None
    return {"camera": camera_id, "width": width, "height": height, "pixels": pixels}


def build_record(
    *,
    set_label: str,
    order: int,
    entry: dict,
    benchmark_entry: dict,
    repo_root: Path,
    output_dir: Path,
    include_points: bool,
    attributes: dict[str, list[str]] | None = None,
    bundle: AssetBundle | None = None,
    step_s: float = DEFAULT_CLIP_STEP_S,
) -> dict:
    """One viewer card: assets, chain of thought, answers and evidence."""
    clip_id = str(entry.get("clip_id") or "").strip()
    scene_dir = scene_paths(repo_root, entry)
    metadata = read_json(scene_dir / "metadata.json" if scene_dir else None)
    additional_info = read_json(
        scene_dir / "additional_info.json" if scene_dir else None
    )

    scene_attributes = (
        nurec_scene(benchmark_entry) or nurec_scene(entry) or nurec_scene(metadata)
    )
    # The benchmark labels win, but the scene folder often carries extra keys
    # (primary_causal_factor, causal_factor) that the released labels omit.
    decision = {
        **cot_decision_label(metadata),
        **cot_decision_label(entry),
        **cot_decision_label(benchmark_entry),
    }
    high_level = {
        **(cot_decision_label(metadata).get("high_level_decision") or {}),
        **(decision.get("high_level_decision") or {}),
    }
    causal = (
        (decision.get("primary_causal_factor") or {}).get("category")
        or high_level.get("causal_factor")
        or next(iter(string_list(decision.get("critical_component_categories"))), None)
    )
    hint = decision.get("atomic_meta_action_hint") or {}

    summary = additional_info.get("summary") or {}
    meta_actions = additional_info.get("meta_actions") or {}
    dominant = meta_actions.get("dominant") or {}

    cot = clean_cot(entry.get("chain_of_thought")) or clean_cot(
        metadata.get("chain_of_thought")
    )
    reliability_flags, reliability_facet = reliability_labels(entry.get("cot_reliability"))
    justification_key, justification = consistency_justification(entry)

    video_path = resolve(repo_root, entry.get("rollout_mp4"))
    if (video_path is None or not video_path.exists()) and clip_id:
        video_path = repo_root / "data" / "media" / "videos" / f"{clip_id}.mp4"
    video = asset_reference(video_path, output_dir, bundle, clip_id)

    facets: dict[str, list[str]] = {}
    for key, _ in SCENE_FACETS:
        facets[key] = string_list(scene_attributes.get(key))
    vrus = scene_attributes.get("vrus")
    facets["vrus"] = (
        ["VRUs present"] if vrus is True else ["no VRUs"] if vrus is False else []
    )
    facets["longitudinal_decision"] = (
        [str(high_level["longitudinal"])] if high_level.get("longitudinal") else []
    )
    facets["lateral_decision"] = (
        [str(high_level["lateral"])] if high_level.get("lateral") else []
    )
    facets["causal_factor"] = [str(causal)] if causal else []
    facets["component_tags"] = string_list(decision.get("critical_component_tags"))
    facets["meta_longitudinal"] = (
        [str(dominant["longitudinal"])] if dominant.get("longitudinal") else []
    )
    facets["meta_lateral"] = [str(dominant["lateral"])] if dominant.get("lateral") else []
    band = speed_band(summary.get("max_speed_kph"))
    facets["speed_band"] = [band] if band else []
    turn = heading_band(summary.get("heading_change_deg"))
    facets["turn_band"] = [turn] if turn else []
    for column, values in (attributes or {}).items():
        facets[attribute_facet_key(column)] = list(values)
    facets["consistency"] = [answer_label(entry.get("cot_action_consistency"))]
    facets["reliability"] = reliability_facet
    facets["trajectory_safe"] = [answer_label(entry.get("planned_trajectory_safe"))]
    points = trajectory_points(additional_info, metadata) if include_points else []

    return {
        "set": set_label,
        "order": order,
        "clip_id": clip_id,
        "selection_source": {
            key: entry[key]
            for key in (
                "source_scene_file", "rollout_mp4", "chain_of_thought",
                "scene_id", "labels", "selection",
            )
            if "selection" in entry and key in entry
        },
        "scene_uuid": scene_attributes.get("scene_uuid"),
        "scene_folder": entry.get("scene_folder"),
        "checkpoint": entry.get("checkpoint_path"),
        "in_benchmark": bool(benchmark_entry),
        "assets": {
            "frame": asset_reference(
                scene_dir / FRAME_NAME if scene_dir else None,
                output_dir,
                bundle,
                clip_id,
            ),
            "trajectory": asset_reference(
                scene_dir / TRAJECTORY_PLOT_NAME if scene_dir else None,
                output_dir,
                bundle,
                clip_id,
            ),
            "video": video,
            "scene_dir": (
                os.path.relpath(scene_dir, repo_root).replace(os.sep, "/")
                if scene_dir
                else None
            ),
        },
        "cot": cot,
        "timing": clip_timing(clip_id, summary, video_path, step_s),
        "answers": {
            "consistency": entry.get("cot_action_consistency"),
            "consistency_justification": justification,
            "reliability_flags": reliability_flags,
            "reliability_justification": as_list(
                (entry.get("cot_reliability") or {}).get("justification")
                if isinstance(entry.get("cot_reliability"), dict)
                else None
            ),
            "trajectory_safe": entry.get("planned_trajectory_safe"),
            "safety_justification": as_list(
                entry.get("planned_trajectory_safety_justification")
            ),
            "note": as_list(entry.get("note")),
            # Kept so the review build can seed its form from any of the
            # answer schemas in circulation, and write the same one back.
            "source": {
                "consistency_justification_key": justification_key,
                "cot_reliability": entry.get("cot_reliability")
                if isinstance(entry.get("cot_reliability"), dict)
                else {},
            },
        },
        "decision": {
            "longitudinal": high_level.get("longitudinal"),
            "lateral": high_level.get("lateral"),
            "causal_factor": causal,
            "component_categories": string_list(
                decision.get("critical_component_categories")
            ),
            "component_tags": facets["component_tags"],
            "hint_longitudinal": hint.get("longitudinal"),
            "hint_lateral": hint.get("lateral"),
        },
        "kinematics": {
            key: summary.get(key)
            for key in (
                "time_horizon_s",
                "total_distance_2d_m",
                "final_forward_offset_m",
                "final_lateral_offset_m",
                "max_abs_lateral_offset_m",
                "min_speed_mps",
                "mean_speed_mps",
                "max_speed_mps",
                "max_speed_kph",
                "min_longitudinal_acceleration_mps2",
                "max_abs_longitudinal_acceleration_mps2",
                "max_abs_lateral_acceleration_mps2",
                "max_abs_jerk_mps3",
                "heading_change_deg",
                "max_abs_curvature_1pm",
                "waypoint_count",
                "trajectory_status",
            )
        },
        "meta_actions": {
            "dominant": dominant,
            "longitudinal_distribution": meta_actions.get("longitudinal_distribution")
            or {},
            "lateral_distribution": meta_actions.get("lateral_distribution") or {},
            "transition_count": meta_actions.get("transition_count"),
        },
        "facets": facets,
        "points": points,
        "camera_plan": camera_plan(find_camera_calibration(scene_dir, repo_root), points)
        if include_points
        else None,
    }


def build_records(
    example_sets: Iterable[tuple[str, Path]],
    benchmark_index: dict[str, dict],
    repo_root: Path,
    output_dir: Path,
    include_points: bool,
    attribute_tables: Iterable[dict[str, dict[str, list[str]]]] = (),
    bundle: AssetBundle | None = None,
    step_s: float = DEFAULT_CLIP_STEP_S,
) -> list[dict]:
    tables = list(attribute_tables)
    records: list[dict] = []
    for set_label, path in example_sets:
        entries = extract_entries(load_json(path))
        for order, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            clip_id = str(entry.get("clip_id") or "").strip()
            attributes: dict[str, list[str]] = {}
            for table in tables:
                attributes.update(attributes_for(table, clip_id))
            records.append(
                build_record(
                    set_label=set_label,
                    order=order,
                    entry=entry,
                    benchmark_entry=benchmark_index.get(clip_id, {}),
                    repo_root=repo_root,
                    output_dir=output_dir,
                    include_points=include_points,
                    attributes=attributes,
                    bundle=bundle,
                    step_s=step_s,
                )
            )
    return records


def evenness(counts: Iterable[int]) -> float | None:
    """Normalised Shannon entropy of a value distribution (1.0 = uniform)."""
    values = [count for count in counts if count > 0]
    total = sum(values)
    if total == 0:
        return None
    if len(values) == 1:
        return 1.0
    entropy = -sum((count / total) * math.log(count / total) for count in values)
    return round(entropy / math.log(len(values)), 3)


def attribute_facet_key(column: str) -> str:
    """Namespaced facet key, so a CSV column cannot shadow a built-in facet."""
    return f"attr:{column}"


def attribute_section(section: str, columns: Iterable[str]) -> dict:
    """A diversity-panel section for one attributes CSV."""
    return {
        "section": section,
        "groups": [
            {
                "key": attribute_facet_key(column),
                "label": column.replace("_", " ").capitalize(),
            }
            for column in columns
        ],
    }


def render(
    template: str,
    title: str,
    records: list[dict],
    sources: list[dict],
    facet_groups: list[dict] | None = None,
    extra: dict | None = None,
) -> str:
    payload = {
        "title": title,
        "sources": sources,
        "records": records,
        "facet_groups": FACET_GROUPS if facet_groups is None else facet_groups,
        **(extra or {}),
    }
    # ``\/`` keeps a literal "</script>" inside the payload from closing the
    # embedded JSON block; it is a valid JSON escape for "/".
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__PAGE_TITLE__", html.escape(title)).replace(
        '"__VIEWER_DATA__"', encoded
    )


FACET_GROUPS = [
    {
        "section": "Scene (NuRec attributes)",
        "groups": [
            {"key": key, "label": label} for key, label in SCENE_FACETS
        ]
        + [{"key": "vrus", "label": "VRUs"}],
    },
    {
        "section": "Chain of thought (Alpamayo-R1 decision label)",
        "groups": [
            {"key": "longitudinal_decision", "label": "Longitudinal decision"},
            {"key": "lateral_decision", "label": "Lateral decision"},
            {"key": "causal_factor", "label": "Primary causal factor"},
            {"key": "component_tags", "label": "Critical components"},
        ],
    },
    {
        "section": "Executed trajectory",
        "groups": [
            {"key": "meta_longitudinal", "label": "Dominant longitudinal action"},
            {"key": "meta_lateral", "label": "Dominant lateral action"},
            {"key": "speed_band", "label": "Peak speed"},
            {"key": "turn_band", "label": "Heading change"},
        ],
    },
    {
        "section": "Answers",
        "groups": [
            {"key": "consistency", "label": "CoT-action consistency"},
            {"key": "reliability", "label": "CoT reliability"},
            {"key": "trajectory_safe", "label": "Trajectory safe"},
        ],
    },
]


#: Facet groups whose values restate a recorded answer, hidden from annotators.
ANSWER_FACETS = ("consistency", "reliability", "trajectory_safe")


def encode_reference(answers: dict) -> str:
    """Base64 the recorded answers so reading the page source cannot spoil them.

    Obfuscation, not secrecy: the annotator page must carry the answers to be
    able to reveal them on request, and anyone who decodes the blob deliberately
    has simply pressed the reveal button the long way round.
    """
    blob = json.dumps(answers, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(blob).decode("ascii")


#: The three reliability flags the form asks about, and the released
#: unreliability category that stands for each one having failed.
RELIABILITY_CATEGORIES = {
    "grounded": "scene_entity_hallucination",
    "safe": "unsafe_planning_from_grounded_observation",
    "coherent": "logical_coherence_error",
}


def released_answers(answers: dict) -> dict:
    """The recorded answers, normalised into the fields the form edits.

    Two reliability schemas are in circulation: the example sets carry the
    three flags the form asks about, while the released benchmark carries one
    ``reliable`` verdict plus the categories that made it false. Those
    categories are one per flag, so the second schema seeds the first exactly.
    Which schema it came from is remembered, so the export can write back the
    shape it was read from rather than silently changing it.
    """
    source = answers.get("source") or {}
    reliability = source.get("cot_reliability") or {}
    flags = answers.get("reliability_flags") or {}
    schema = "reliable" if "reliable" in flags else "flags"
    if schema == "reliable":
        reliable = flags.get("reliable")
        failed = set(reliability.get("unreliability_categories") or [])
        derived = {
            name: None if reliable is None else category not in failed
            for name, category in RELIABILITY_CATEGORIES.items()
        }
    else:
        derived = {
            "grounded": flags.get("grounded"),
            "safe": flags.get("safe"),
            "coherent": flags.get("logically_coherent"),
        }
    return {
        "consistency": answers.get("consistency"),
        "consistency_justification": answers.get("consistency_justification") or [],
        **derived,
        "reliability_justification": answers.get("reliability_justification") or [],
        "trajectory_safe": answers.get("trajectory_safe"),
        "safety_justification": answers.get("safety_justification") or [],
        "reliability_schema": schema,
        # Carried so an untouched clip exports the categories it arrived with,
        # in the order it arrived in, rather than in flag order.
        "unreliability_categories": list(
            reliability.get("unreliability_categories") or []
        ),
        "primary_unreliability_category": reliability.get(
            "primary_unreliability_category"
        ),
        "consistency_justification_key": source.get(
            "consistency_justification_key", CONSISTENCY_JUSTIFICATION_KEYS[0]
        ),
    }


def annotator_record(record: dict, prefill: bool = False) -> dict:
    """Move the recorded answers behind the reveal and drop the note field.

    The note is a private aside on the entry rather than an answer to any of
    the three questions, so the annotator version leaves it out entirely.

    ``prefill`` builds the review variant instead: the same answers travel in
    the clear, under ``released``, because the form is seeded from them and
    the page has nothing left to spoil. The answer facets stay in the panel
    there too, so a reviewer can slice the set by the verdict being checked.
    """
    answers = dict(record.pop("answers", {}))
    answers.pop("note", None)
    if prefill:
        record["released"] = released_answers(answers)
        return record
    answers.pop("source", None)
    record["reference"] = encode_reference(answers)
    for key in ANSWER_FACETS:
        record["facets"].pop(key, None)
    return record


def annotator_facet_groups(facet_groups: Iterable[dict]) -> list[dict]:
    """The facet panel minus every group that would give an answer away."""
    sections = []
    for section in facet_groups:
        groups = [
            group for group in section["groups"] if group["key"] not in ANSWER_FACETS
        ]
        if groups:
            sections.append({**section, "groups": groups})
    return sections


def default_output(
    example_sets: list[tuple[str, Path]],
    repo_root: Path,
    suffix: str = ".viewer.html",
) -> Path:
    """``<first examples path><suffix>``, so one build never clobbers another."""
    if not example_sets or [path for _, path in example_sets] == [
        path for _, path in DEFAULT_EXAMPLES
    ]:
        return repo_root / "data" / "benchmark" / suffix.lstrip(".")
    first = example_sets[0][1]
    return first.with_suffix(suffix)


def main() -> None:
    args = parse_args()
    # The review build is the annotator page with its answers already in it.
    args.annotator = args.annotator or args.prefill
    repo_root = args.repo_root.resolve()

    if args.examples:
        example_sets = [parse_example_arg(raw, repo_root) for raw in args.examples]
    else:
        example_sets = [(label, path) for label, path in DEFAULT_EXAMPLES]

    missing = [str(path) for _, path in example_sets if not path.exists()]
    if missing:
        raise SystemExit("Example set not found: " + ", ".join(missing))

    suffix = (
        ".review.html"
        if args.prefill
        else ".annotator.html" if args.annotator else ".viewer.html"
    )
    if args.bundle is not None:
        bundle_dir = args.bundle
        if not bundle_dir.is_absolute():
            bundle_dir = repo_root / bundle_dir
        output = bundle_dir.resolve() / "index.html"
    else:
        output = args.output or default_output(example_sets, repo_root, suffix)
        if not output.is_absolute():
            output = repo_root / output
        output = output.resolve()
    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = AssetBundle(output_dir / "media", output_dir) if args.bundle else None

    benchmark_index: dict[str, dict] = {}
    if args.benchmark and Path(args.benchmark).exists():
        benchmark_index = dict(
            index_by_clip_id(extract_entries(load_json(Path(args.benchmark))))
        )

    attribute_tables = []
    facet_groups = list(FACET_GROUPS)
    for raw in args.attributes or []:
        section, path = parse_example_arg(raw, repo_root)
        if not path.exists():
            raise SystemExit(f"Attributes CSV not found: {path}")
        key_column, table = load_attributes(path)
        if not table:
            print(f"  warning: no rows in {path}")
            continue
        columns = sorted({column for values in table.values() for column in values})
        attribute_tables.append(table)
        facet_groups.insert(0, attribute_section(f"{section} (by {key_column})", columns))

    records = build_records(
        example_sets,
        benchmark_index,
        repo_root,
        output_dir,
        not args.no_trajectory_points,
        attribute_tables,
        bundle=bundle,
        step_s=args.clip_step_s,
    )
    sources = [
        {
            "label": label,
            "path": os.path.relpath(path, repo_root).replace(os.sep, "/"),
            "count": sum(1 for record in records if record["set"] == label),
        }
        for label, path in example_sets
    ]

    extra: dict = {}
    if args.annotator:
        if not args.prefill:
            facet_groups = annotator_facet_groups(facet_groups)
        records = [annotator_record(record, args.prefill) for record in records]
        # Namespaces the browser autosave, so two example sets annotated on the
        # same machine cannot overwrite each other's in-progress verdicts — and
        # so a review pass never lands on top of a blind pass over the same set.
        extra["storage_key"] = (sources[0]["path"] if sources else args.title) + (
            ":review" if args.prefill else ""
        )
        extra["prefill"] = args.prefill

    template = (ANNOTATOR_TEMPLATE_PATH if args.annotator else TEMPLATE_PATH).read_text(
        encoding="utf-8"
    )
    output.write_text(
        render(template, args.title, records, sources, facet_groups, extra),
        encoding="utf-8",
    )

    missing_frames = sum(1 for r in records if not r["assets"]["frame"])
    missing_plots = sum(1 for r in records if not r["assets"]["trajectory"])
    missing_videos = sum(1 for r in records if not r["assets"]["video"])
    unlabelled = sum(1 for r in records if not r["in_benchmark"])
    if attribute_tables:
        matched = sum(
            1
            for r in records
            if any(key.startswith("attr:") and r["facets"][key] for key in r["facets"])
        )
        print(f"  attribute rows matched: {matched}/{len(records)} clips")
    print(f"Wrote {output} ({len(records)} clips)")
    for source in sources:
        print(f"  {source['label']}: {source['count']} clips from {source['path']}")
    print(
        f"  missing assets - frame: {missing_frames}, trajectory plot: {missing_plots}, "
        f"video: {missing_videos}; clips without benchmark labels: {unlabelled}"
    )
    if bundle is not None:
        print(
            f"  bundled {len(bundle.copied)} media files "
            f"({bundle.bytes / 1e6:.1f} MB) into {bundle.directory}"
        )
        print(f"  hand it over:  cd {output_dir.parent} && zip -r {output_dir.name}.zip "
              f"{output_dir.name}")
        print(f"  or host it:    python -m http.server --directory {output_dir}")


if __name__ == "__main__":
    main()
