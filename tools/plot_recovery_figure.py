#!/usr/bin/env python
"""Render the panels of the lane-relative example figure (fig:fllm-recovery).

Panels (b) and (c) show one trajectory against two references, in the map style
of the page-1 figure. They are north-up: the ego heads up the page and its left
is the page left, so + = left reads directly. LaTeX sets all of the text.

  (a) camera crop
  (b) offset from a straight ray along the ego's initial heading
  (c) offset from the routed lane centre, via the same projection the judge uses

The script stops if the offsets disagree with the rig-frame waypoints the
raw-waypoint judge reads, or with the values printed in the figure.

Two examples are kept. false_alarm is the current figure: a consistent pair that
raw waypoints flag and F-LLM accepts. recovery is the earlier figure, now commented out
in sections/fig_fllm_recovery.tex: an inconsistent pair only F-LLM flags.

Usage: python tools/plot_recovery_figure.py --out-dir <dir> [--example recovery]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpasim_utils.lane_projection import (  # noqa: E402
    compute_lane_relative_trajectory,
    lane_center_lines_from_geometry,
)

# Print size of one map panel.
PANEL_CM = (5.2, 3.6)


@dataclass(frozen=True)
class Example:
    clip_id: str
    stem: str  # output files: <stem>_ego.pdf, <stem>_lane.pdf, <stem>_scene.jpg
    # View in metres, the same in both panels: lateral (+ = right) and forward.
    view_width_m: float
    view_x_min: float
    view_y_center: float
    # Checked against the stored trajectory: (start, final) e_t in metres.
    expected_ray: tuple[float, float]
    expected_lane: tuple[float, float]
    scene_crop: tuple[int, int, int, int]  # pixels of the 1920 x 1080 frame
    # Draw the route over the trajectory when the two nearly coincide.
    route_on_top: bool


EXAMPLES = {
    "false_alarm": Example(
        clip_id="clipgt-f6de0321-2272-42ff-a50a-4edea5168565_t0035",
        stem="cot_falsealarm",
        view_width_m=106.9,  # 0.04865 cm/m
        view_x_min=-66.0,
        view_y_center=32.0,
        expected_ray=(0.0, -6.58),
        expected_lane=(-0.96, 0.54),
        scene_crop=(520, 430, 1480, 938),
        route_on_top=True,
    ),
    "recovery": Example(
        clip_id="clipgt-14a02b32-08ac-493f-b9b5-5367df0cc568_t0061",
        stem="cot_recovery",
        view_width_m=49.84,  # 0.1043 cm/m
        view_x_min=-19.17,
        view_y_center=12.75,
        expected_ray=(0.0, -0.62),
        expected_lane=(-0.41, 9.47),
        scene_crop=(468, 357, 1568, 939),
        route_on_top=False,
    ),
}

ROAD, LANE, ROUTE, PATH = "#7E9DBD", "#B9CBDD", "#14806F", "#E07B1A"
EGO, ACTOR, RAY = "#1B7837", "#C9CED6", "#6B7280"
CONNECTOR_STEP = 3


def ego_axes(points, origin: np.ndarray, yaw: float) -> np.ndarray:
    """World xy to (forward, left) of the ego's initial pose."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    q = np.asarray(points, float)[..., :2] - origin
    return np.stack([c * q[..., 0] - s * q[..., 1], s * q[..., 0] + c * q[..., 1]], -1)


def north_up(points: np.ndarray) -> np.ndarray:
    """(forward, left) to page coordinates (right, up)."""
    return np.stack([-points[..., 1], points[..., 0]], -1)


def new_panel(example: Example, geometry: dict, origin: np.ndarray, yaw: float):
    width = example.view_width_m
    height = width * PANEL_CM[1] / PANEL_CM[0]
    xlim = (example.view_x_min, example.view_x_min + width)
    ylim = (example.view_y_center - height / 2, example.view_y_center + height / 2)

    def in_view(q: np.ndarray) -> np.ndarray:
        return (
            (q[:, 0] > xlim[0] - 3)
            & (q[:, 0] < xlim[1] + 3)
            & (q[:, 1] > ylim[0] - 3)
            & (q[:, 1] < ylim[1] + 3)
        )

    fig = plt.figure(figsize=(PANEL_CM[0] / 2.54, PANEL_CM[1] / 2.54))
    ax = fig.add_axes((0, 0, 1, 1))
    for line in geometry["map_linestrings"]:
        q = north_up(ego_axes(line["coords"], origin, yaw))
        if not in_view(q).any():
            continue
        if line["type"] == "road_edge":
            ax.plot(q[:, 0], q[:, 1], color=ROAD, lw=0.45, zorder=2, solid_capstyle="round")
        elif line["type"] in ("road_lane_left_edge", "road_lane_right_edge"):
            ax.plot(q[:, 0], q[:, 1], color=LANE, lw=0.25, ls=(0, (3, 2.6)), alpha=0.8, zorder=1)
    for actor in geometry["actors"]:
        q = north_up(ego_axes(actor["polygon_coords"], origin, yaw))
        if actor["id"] == "EGO":
            ax.fill(q[:, 0], q[:, 1], fc=EGO, ec="white", lw=0.3, zorder=9)
        elif in_view(q).any():
            ax.fill(q[:, 0], q[:, 1], fc=ACTOR, ec="#8C939E", lw=0.25, zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def finish_panel(fig, ax, path: np.ndarray, out: Path) -> None:
    ax.plot(path[:, 0], path[:, 1], color="white", lw=2.1, zorder=6, solid_capstyle="round")
    ax.plot(path[:, 0], path[:, 1], color=PATH, lw=1.25, zorder=7, solid_capstyle="round")
    # The waypoints the prompts sample: every 8th step of the 10 Hz trajectory.
    ax.scatter(path[::8, 0], path[::8, 1], s=3.2, c=PATH, edgecolors="white", linewidths=0.25, zorder=8)
    ax.scatter(*path[-1], s=9, c=PATH, edgecolors="white", linewidths=0.4, zorder=8)
    fig.savefig(out, transparent=False)
    plt.close(fig)


def check_offsets(offsets: np.ndarray, expected: tuple[float, float], name: str) -> None:
    start, final = float(offsets[0]), float(offsets[-1])
    print(f"{name}: start {start:+.2f} final {final:+.2f} max {np.abs(offsets).max():.2f}")
    if (round(start, 2) + 0.0, round(final, 2)) != expected:
        raise SystemExit(f"{name} offsets no longer match the figure text {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--example", choices=sorted(EXAMPLES), default="false_alarm")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    example = EXAMPLES[args.example]

    matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
    matplotlib.rcParams["ps.fonttype"] = 42

    scene = ROOT / "data" / "media" / "scenes" / example.clip_id
    metadata = json.loads((scene / "metadata.json").read_text())
    geometry = json.loads((scene / "trajectory_plot_geometry.json").read_text())

    xy = np.array([[p["x"], p["y"]] for p in metadata["trajectory_poses"]], dtype=float)
    origin = xy[0]
    # True anchor yaw. trajectory_poses carries no orientation, and the
    # raw-waypoint judge's rig-frame waypoints are rotated by this same yaw.
    yaw = float(next(a["yaw"] for a in geometry["actors"] if a["id"] == "EGO"))

    local = ego_axes(xy, origin, yaw)
    rig_ry = np.array([p["ry"] for p in metadata["trajectory_xy_rig_frame"]])
    # Poses are stored as float32, which accounts for up to about 1.2 mm here.
    if not np.allclose(local[:, 1], rig_ry, atol=2e-3):
        raise SystemExit("ray offsets disagree with trajectory_xy_rig_frame")
    check_offsets(local[:, 1], example.expected_ray, "ego")

    result = compute_lane_relative_trajectory(
        xy, lane_center_lines_from_geometry(geometry), prefer="map_graph"
    )
    if result.method != "map_graph":
        raise SystemExit(f"expected map_graph reference, got {result.method}")
    check_offsets(result.offsets, example.expected_lane, "lane")

    path = north_up(local)
    route = north_up(ego_axes(result.reference_xy, origin, yaw))
    feet = north_up(ego_axes(result.projections, origin, yaw))

    # (b) straight ray along the ego's initial heading: the page's vertical axis.
    fig, ax = new_panel(example, geometry, origin, yaw)
    ax.plot([0, 0], [0, path[:, 1].max() + 1.5], color=RAY, lw=1.0, ls=(0, (3.2, 1.8)), zorder=5)
    for p in path[::CONNECTOR_STEP]:
        ax.plot([p[0], 0], [p[1], p[1]], color=RAY, lw=0.3, alpha=0.7, zorder=4)
    ax.plot([path[-1, 0], 0], [path[-1, 1]] * 2, color=RAY, lw=0.85, zorder=5, solid_capstyle="butt")
    finish_panel(fig, ax, path, args.out_dir / f"{example.stem}_ego.pdf")

    # (c) routed lane centre, the same reference the judge's features use.
    fig, ax = new_panel(example, geometry, origin, yaw)
    route_style = (0.8, 7.5) if example.route_on_top else (1.0, 5)
    ax.plot(
        route[:, 0],
        route[:, 1],
        color=ROUTE,
        lw=route_style[0],
        ls=(0, (3.2, 1.8)),
        zorder=route_style[1],
    )
    for p, foot in zip(path[::CONNECTOR_STEP], feet[::CONNECTOR_STEP]):
        ax.plot([p[0], foot[0]], [p[1], foot[1]], color=ROUTE, lw=0.3, alpha=0.6, zorder=4)
    ax.plot(
        [path[-1, 0], feet[-1, 0]],
        [path[-1, 1], feet[-1, 1]],
        color=ROUTE,
        lw=0.85,
        zorder=5,
        solid_capstyle="butt",
    )
    finish_panel(fig, ax, path, args.out_dir / f"{example.stem}_lane.pdf")

    # (a) the same frame: resampled only, no colour change.
    frame = Image.open(scene / "camera_front_wide_120fov.jpg").convert("RGB")
    crop = frame.crop(example.scene_crop)
    crop.resize((1200, round(1200 * crop.height / crop.width)), Image.LANCZOS).save(
        args.out_dir / f"{example.stem}_scene.jpg", quality=93, subsampling=0
    )


if __name__ == "__main__":
    main()
