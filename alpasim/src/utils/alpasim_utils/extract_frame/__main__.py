# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import argparse
import asyncio
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from alpasim_utils.lane_projection import compute_lane_relative_trajectory
from alpasim_utils.logs import async_read_pb_log
from alpasim_utils.trajectory_additional_info import (
    DEFAULT_OUTPUT_FILENAME as ADDITIONAL_INFO_FILENAME,
)
from alpasim_utils.trajectory_additional_info import build_additional_info
from matplotlib.patches import Rectangle

mpl.use("Agg")
mplstyle.use("fast")


def render_eval_visualization(
    sim_result,
    cfg,
    time_now_us_target,
    output_path,
    trajectory_world_xy=None,
    trajectory_timestamps=None,
    lane_overlay_path=None,
    cot_text=None,
):
    # This logic matches video.py map rendering logic

    import matplotlib.pyplot as plt

    try:
        from eval.schema import MapElements
        from eval.video import get_ego_transform
        from eval.video_data import ShapelyMap
    except ImportError:
        print("eval package unavailable.")
        return

    if hasattr(sim_result, "actor_polygons"):
        sim_result.actor_polygons.artists = {}
    if hasattr(sim_result, "route"):
        sim_result.route.artists = None
    if hasattr(sim_result, "driver_responses"):
        sim_result.driver_responses.artists = None

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    # Using white background to match standard eval video
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3, color="#dddddd")

    shapely_map = (
        ShapelyMap.from_vec_map(sim_result.vec_map) if sim_result.vec_map else None
    )
    ego_transform = get_ego_transform(
        sim_result=sim_result, cfg=cfg, time=time_now_us_target
    )

    image_center_xy = sim_result.actor_polygons.set_axis_limits_around_agent(
        ax, "EGO", time_now_us_target, cfg, axis_transform=ego_transform
    )

    if shapely_map:
        shapely_map.render(
            ax,
            cfg,
            center=image_center_xy,
            max_dist=cfg.video.map_video.map_radius_m + 10,
        )

    # Note: the exact objects in video.py
    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.GT_LINESTRING in cfg.video.map_video.map_elements_to_plot
    ):
        sim_result.ego_recorded_ground_truth_trajectory.set_linestring_plot_style(
            "gt_linestring", linewidth=1, style="g-", alpha=0.7
        ).render_linestring(ax)

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.AGENTS in cfg.video.map_video.map_elements_to_plot
    ):
        sim_result.actor_polygons.render_at_time(
            ax,
            time_now_us_target,
            center=image_center_xy,
            max_dist=cfg.video.map_video.map_radius_m + 10,
        )
    else:
        sim_result.actor_polygons.render_at_time(
            ax, time_now_us_target, only_agents=["EGO"]
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.DRIVER_RESPONSES in cfg.video.map_video.map_elements_to_plot
    ):
        sim_result.driver_responses.render_at_time(ax, time_now_us_target, "now")

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.ROUTE in cfg.video.map_video.map_elements_to_plot
    ) and hasattr(sim_result, "routes"):
        sim_result.routes.render_at_time(ax, time_now_us_target)

    if image_center_xy:
        r = cfg.video.map_video.map_radius_m
        ax.set_xlim(image_center_xy.x - r, image_center_xy.x + r)
        ax.set_ylim(image_center_xy.y - r + 0.3 * r, image_center_xy.y + r + 0.3 * r)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")

    # Overlay the lane-offset basis on the *same* BEV (world frame, so the
    # reference and predicted trajectory align with the rendered map) and save it
    # as a companion image. This shows how offset_m is computed: the magenta
    # reference lane line is what the trajectory's lateral offset is measured to.
    if (
        trajectory_world_xy is not None
        and lane_overlay_path is not None
        and shapely_map is not None
    ):
        try:
            lane_center_lines = _lane_center_lines_from_shapely(shapely_map)
            result = (
                draw_lane_reference_overlay(ax, trajectory_world_xy, lane_center_lines)
                if lane_center_lines
                else None
            )
            if result is not None:
                if cot_text:
                    cot = cot_text if len(cot_text) <= 110 else cot_text[:107] + "..."
                    ax.set_title(cot, fontsize=9)
                fig.savefig(
                    lane_overlay_path,
                    facecolor=fig.get_facecolor(),
                    edgecolor="none",
                )
                write_lane_relative_sidecar(
                    result,
                    trajectory_world_xy,
                    trajectory_timestamps,
                    lane_overlay_path.with_suffix(".json"),
                )
                print(f"Saved Lane-Relative Overlay: {lane_overlay_path}")
            else:
                print("Lane-relative overlay skipped: no road_lane_center reference")
        except Exception as e:
            print(f"Lane-relative overlay failed: {e}")

    plt.close(fig)

    # Extract literal geometry into a sibling JSON file
    geometry_path = output_path.with_name(f"{output_path.stem}_geometry.json")
    try:
        geometry_data = {
            "timestamp_us": int(time_now_us_target),
            "map_linestrings": [],
            "actors": [],
        }
        if shapely_map:
            for ls in shapely_map.renderable_linestrings:
                geometry_data["map_linestrings"].append(
                    {"type": str(ls.name), "coords": list(ls.linestring.coords)}
                )

        polys_at_time = sim_result.actor_polygons.get_polygons_at_time(
            time_now_us_target
        )
        for agent_id, poly, yaw in zip(
            polys_at_time.agent_ids, polys_at_time.bbox_polygons, polys_at_time.yaws
        ):
            geometry_data["actors"].append(
                {
                    "id": agent_id,
                    "yaw": float(yaw),
                    "polygon_coords": list(poly.exterior.coords),
                }
            )

        with open(geometry_path, "w") as f:
            json.dump(geometry_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to extract geometry JSON: {e}")


def save_trajectory_plot(trajectory_xy, output_path: Path):
    """Draws a neat BEV plot of the trajectory based on output visualization styles."""
    fig, ax = plt.subplots(figsize=(4.0, 6.0), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if len(trajectory_xy) > 0:
        # In rig-frame output:
        # rx is longitudinal (forward) -> plot_y
        # ry is lateral (left) -> plot_x = -ry for standard layout (right is positive in plot)
        px = [-pt["ry"] for pt in trajectory_xy]
        py = [pt["rx"] for pt in trajectory_xy]

        ax.plot(
            px,
            py,
            "o-",
            color="#4ecdc4",
            linewidth=2.5,
            markersize=5,
            alpha=0.9,
        )

    # Mark current ego position (t0) as a rectangle
    car_length = 4.5
    car_width = 1.8
    ego_rect = Rectangle(
        (-car_width / 2, -car_length / 2),
        car_width,
        car_length,
        facecolor="#ffd93d",
        edgecolor="#ffffff",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(ego_rect)

    # Set FIXED axis limits and ticks
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 80)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3, color="#555555")

    ax.tick_params(colors="#ffffff", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# Overlay colors chosen to pop on the white eval BEV and stay distinct from the
# green recorded GT trajectory and the gray map lanes.
_REF_LANE_COLOR = "#ff1f8f"  # magenta — the lane center offsets are measured from
_PRED_TRAJ_COLOR = "#1f4fff"  # blue — the predicted trajectory offset_m is for
_OFFSET_LEFT_COLOR = "#0aa3c2"  # connector when trajectory is left of lane
_OFFSET_RIGHT_COLOR = "#ff7f0e"  # connector when right of lane


def _lane_center_lines_from_shapely(shapely_map):
    """Pull ``road_lane_center`` polylines (world XY) from a rendered ShapelyMap.

    Same source the geometry sidecar dumps, so the overlay's reference matches
    the one the CoT judge rebuilds from that sidecar.
    """
    lines = []
    for ls in shapely_map.renderable_linestrings:
        if str(ls.name) != "road_lane_center":
            continue
        coords = np.asarray(ls.linestring.coords, dtype=float)
        if coords.ndim == 2 and coords.shape[0] >= 2 and coords.shape[1] >= 2:
            lines.append(coords[:, :2])
    return lines or None


def draw_lane_reference_overlay(ax, trajectory_world_xy, lane_center_lines):
    """Overlay the lane-offset basis onto the world-frame eval BEV axis.

    Draws, in colors distinct from the eval map: the lane-center reference the
    offset is measured from (magenta), the predicted trajectory it is computed
    for (blue), and the per-waypoint offset connectors between them. Returns the
    LaneRelativeResult (so the numbers can be written to a sidecar), or None if
    no reference could be built.
    """
    xy = np.asarray(trajectory_world_xy, dtype=float)
    if xy.ndim != 2 or len(xy) < 3 or not lane_center_lines:
        return None
    try:
        result = compute_lane_relative_trajectory(
            xy, lane_center_lines, prefer="map_graph"
        )
    except ValueError:
        return None

    offsets = result.offsets
    proj = result.projections
    ref = result.reference_xy

    # Offset connectors (trajectory point -> its foot on the lane center),
    # colored by which side of the lane the trajectory is on.
    for i in range(len(xy)):
        ax.plot(
            [xy[i, 0], proj[i, 0]],
            [xy[i, 1], proj[i, 1]],
            "-",
            color=_OFFSET_LEFT_COLOR if offsets[i] >= 0 else _OFFSET_RIGHT_COLOR,
            alpha=0.6,
            linewidth=0.9,
            zorder=11,
        )
    ax.plot(
        ref[:, 0],
        ref[:, 1],
        "--",
        color=_REF_LANE_COLOR,
        linewidth=3.0,
        alpha=0.95,
        zorder=12,
        label=f"Lane ref (offset basis, {result.method})",
    )
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        "-",
        color=_PRED_TRAJ_COLOR,
        linewidth=2.2,
        marker="o",
        markersize=3,
        alpha=0.95,
        zorder=13,
        label="Predicted trajectory",
    )
    ax.scatter(
        [xy[0, 0]],
        [xy[0, 1]],
        c="#2ca02c",
        s=70,
        zorder=14,
        edgecolors="white",
        linewidth=1.0,
        label="Start",
    )
    ax.scatter(
        [xy[-1, 0]],
        [xy[-1, 1]],
        c="#000000",
        s=70,
        zorder=14,
        edgecolors="white",
        linewidth=1.0,
        label="End",
    )

    stats = (
        "offset_m (+ = left of lane)\n"
        f"initial {offsets[0]:+.2f}   final {offsets[-1]:+.2f}\n"
        f"delta {offsets[-1] - offsets[0]:+.2f}   max|off| {np.abs(offsets).max():.2f}"
    )
    ax.text(
        0.015,
        0.015,
        stats,
        transform=ax.transAxes,
        fontsize=8,
        color="#111111",
        va="bottom",
        ha="left",
        family="monospace",
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor=_REF_LANE_COLOR, alpha=0.9
        ),
        zorder=15,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return result


def write_lane_relative_sidecar(
    result, trajectory_world_xy, trajectory_timestamps, json_path: Path
):
    """Write the per-waypoint lane offsets behind the overlay to a JSON sidecar."""
    xy = np.asarray(trajectory_world_xy, dtype=float)
    if trajectory_timestamps is None:
        ts = np.zeros(len(xy))
    else:
        ts = np.asarray(trajectory_timestamps, dtype=float)
    offsets = result.offsets
    proj = result.projections
    cumulative_dist = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    )
    data = {
        "reference_method": result.method,
        "lane_segments_used": result.lane_segments_used,
        "offset_convention": "signed lateral distance from lane center, + = left of lane direction",
        "stats": {
            "initial_offset_m": round(float(offsets[0]), 4),
            "final_offset_m": round(float(offsets[-1]), 4),
            "delta_offset_m": round(float(offsets[-1] - offsets[0]), 4),
            "min_offset_m": round(float(offsets.min()), 4),
            "max_offset_m": round(float(offsets.max()), 4),
            "max_abs_offset_m": round(float(np.abs(offsets).max()), 4),
            "rms_offset_m": round(float(np.sqrt(np.mean(offsets**2))), 4),
        },
        "per_waypoint": [
            {
                "timestamp_us": int(ts[i]),
                "world_x": round(float(xy[i, 0]), 4),
                "world_y": round(float(xy[i, 1]), 4),
                "lane_center_x": round(float(proj[i, 0]), 4),
                "lane_center_y": round(float(proj[i, 1]), 4),
                "offset_m": round(float(offsets[i]), 4),
                "s_lane_m": round(float(result.s_lane[i]), 4),
                "cumulative_dist_m": round(float(cumulative_dist[i]), 4),
            }
            for i in range(len(xy))
        ],
        "reference_polyline_xy": [
            [round(float(x), 4), round(float(y), 4)] for x, y in result.reference_xy
        ],
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


async def extract_inference(
    asl_path: str,
    output_dir: str,
    target_time_us: int | None = None,
    write_additional_info: bool = True,
):
    """
    Programmatically extracts inputs, Chain of Thought (CoT), trajectory outputs,
    and image frames from an ALPASIM .asl rollout file.
    Saves outputs to the specified directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Reading ASL file: {asl_path}")
    print(f"Extracting to directory: {out_path.absolute()}\n")
    # Preload the simulation result if eval package exists
    sim_result = None
    eval_cfg = None
    try:
        from alpasim_utils.artifact import Artifact
        from omegaconf import OmegaConf

        from eval.accumulator import EvalDataAccumulator
        from eval.data import SimulationResult
        from eval.schema import EvalConfig

        print("Discovering map artifacts for full visualization...")
        artifacts = {}
        import glob

        for path in glob.glob("data/**/*.usdz", recursive=True):
            try:
                # To prevent Torch/CUDA from stalling here unnecessarily, we just parse ID
                a = Artifact(path, _smooth_trajectories=True)
                if a.scene_id not in artifacts:
                    artifacts[a.scene_id] = a
            except Exception:
                pass

        eval_cfg = OmegaConf.structured(EvalConfig)
        if Path("outputs/wizard-config-loadable.yaml").exists():
            wiz_cfg = OmegaConf.load("outputs/wizard-config-loadable.yaml")
            if "eval" in wiz_cfg:
                eval_cfg = OmegaConf.merge(eval_cfg, wiz_cfg.eval)

        acc = EvalDataAccumulator(cfg=eval_cfg)
        preload_scene_id = "unknown"
        async for m in async_read_pb_log(asl_path):
            acc.handle_message(m)
            if m.WhichOneof("log_entry") == "rollout_metadata":
                if m.rollout_metadata.session_metadata is not None:
                    preload_scene_id = m.rollout_metadata.session_metadata.scene_id

        vec_map = (
            artifacts[preload_scene_id].map if preload_scene_id in artifacts else None
        )
        scenario_input = acc.build_scenario_eval_input(
            run_uuid="extract",
            run_name="extract",
            vec_map=vec_map,
        )
        sim_result = SimulationResult.from_scenario_input(scenario_input, eval_cfg)
        print("Successfully pre-loaded visualization evaluation context.")
    except Exception as e:
        print(f"Could not load full visualization context, will fallback. ({e})")

    # Extract scene_id from metadata to build clip_id
    scene_id = "unknown"
    frame_index = 0

    # Track the active driver request to associate responses
    pending_request = None

    # Store metadata grouped by time_now_us
    # Since driver_camera_image.frame_end_us typically matches driver_request.time_now_us
    frames_metadata = {}

    async for message in async_read_pb_log(asl_path):
        msg_type = message.WhichOneof("log_entry")

        if msg_type == "rollout_metadata":
            if message.rollout_metadata.session_metadata is not None:
                scene_id = message.rollout_metadata.session_metadata.scene_id

        # 1. Inputs: Related Images (these happen BEFORE driver_request in the log)
        elif msg_type == "driver_camera_image":
            cam_img = message.driver_camera_image.camera_image
            frame_end_us = (
                cam_img.frame_end_us
            )  # Matches time_now_us of the inference step

            # Skip images not matching our target frame (if specified)
            if target_time_us and frame_end_us != target_time_us:
                continue

            camera_name = cam_img.logical_id
            image_bytes = cam_img.image_bytes

            # Store images in metadata dict immediately by their end time
            # Since driver_request hasn't happened yet, we initialize a shell for this frame
            if frame_end_us not in frames_metadata:
                clip_id = f"{scene_id}_t{frame_index:04d}"
                frames_metadata[frame_end_us] = {
                    "clip_id": clip_id,
                    "time_now_us": frame_end_us,
                    "chain_of_thought": None,
                    "trajectory_poses": [],
                    "images_pending_save": [],
                }
                frame_index += 1

            frames_metadata[frame_end_us]["images_pending_save"].append(
                (camera_name, image_bytes)
            )

        # 2. Inputs: Driver Requests
        elif msg_type == "driver_request":
            time_query_us = message.driver_request.time_query_us
            time_now_us = message.driver_request.time_now_us

            # If target_time_us is specified, skip if we don't match exactly
            if (
                target_time_us
                and time_query_us != target_time_us
                and time_now_us != target_time_us
            ):
                continue

            # The images should have created the dict key already via time_now_us
            # We just need to record the time_query_us and link the pending_request
            if time_now_us not in frames_metadata:
                # Fallback if no images fired
                clip_id = f"{scene_id}_t{frame_index:04d}"
                frames_metadata[time_now_us] = {
                    "clip_id": clip_id,
                    "time_now_us": time_now_us,
                    "chain_of_thought": None,
                    "trajectory_poses": [],
                    "images_pending_save": [],
                }
                frame_index += 1

            frames_metadata[time_now_us]["time_query_us"] = time_query_us
            pending_request = time_query_us

            # Keep a reverse mapping so we can find the time_now_us from time_query_us
            # Since we just need the dictionary reference directly, let's keep a lookup pointer
            # We don't overwrite frames_metadata[time_query_us] if it conflicts.
            pass

        # 3. Outputs & Chain-Of-Thought
        elif msg_type == "driver_return":
            resp = message.driver_return

            if pending_request is None:
                continue

            # Use the currently tracked pending_request from the previous driver_request parsing
            meta_match = None
            for md in frames_metadata.values():
                if isinstance(md, dict) and md.get("time_query_us") == pending_request:
                    meta_match = md
                    break

            if not meta_match:
                continue

            meta = meta_match

            # 3a. Outputs  (Rig-frame relative XY trajectory like in output videos)
            trajectory_xy = []
            if len(resp.trajectory.poses) >= 1:
                anchor = resp.trajectory.poses[0]
                anchor_x = anchor.pose.vec.x
                anchor_y = anchor.pose.vec.y
                anchor_yaw = np.arctan2(
                    2.0
                    * (
                        anchor.pose.quat.w * anchor.pose.quat.z
                        + anchor.pose.quat.x * anchor.pose.quat.y
                    ),
                    1.0 - 2.0 * (anchor.pose.quat.y**2 + anchor.pose.quat.z**2),
                )
                cos_yaw = np.cos(-anchor_yaw)
                sin_yaw = np.sin(-anchor_yaw)

                # Starting from anchor (t=0, ego rig-frame offset [0, 0])
                for pt in resp.trajectory.poses:
                    dx = pt.pose.vec.x - anchor_x
                    dy = pt.pose.vec.y - anchor_y
                    rx = cos_yaw * dx - sin_yaw * dy
                    ry = sin_yaw * dx + cos_yaw * dy
                    trajectory_xy.append(
                        {"timestamp_us": pt.timestamp_us, "rx": rx, "ry": ry}
                    )

            meta["trajectory_xy_rig_frame"] = trajectory_xy

            # Keep keeping raw global positions for completion
            poses = []
            for p in resp.trajectory.poses:
                poses.append(
                    {
                        "timestamp_us": p.timestamp_us,
                        "x": p.pose.vec.x,
                        "y": p.pose.vec.y,
                        "z": p.pose.vec.z,
                    }
                )
            meta["trajectory_poses"] = poses

            # 3b. Chain of Thought
            dbg_bytes = resp.debug_info.unstructured_debug_info
            if dbg_bytes:
                try:
                    extra_payload = pickle.loads(dbg_bytes)
                    if isinstance(extra_payload, dict):
                        meta["chain_of_thought"] = extra_payload.get("reasoning_text")
                except Exception:
                    pass

            # We are complete! Output the clip structures directly using its clip_id
            clip_id = meta["clip_id"]
            clip_dir = out_path / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Dump the images
            for cam_name, img_bytes in meta.get("images_pending_save", []):
                image_path = clip_dir / f"{cam_name}.jpg"
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
                print(f"Saved Image: {image_path}")

            # Dump the metadata
            meta_path = clip_dir / "metadata.json"

            # Strip the temporary cached binary data before stringifying
            meta_to_save = {k: v for k, v in meta.items() if k != "images_pending_save"}

            with open(meta_path, "w") as f:
                json.dump(meta_to_save, f, indent=2)

            print(f"Saved Metadata: {meta_path}")

            if write_additional_info:
                additional_info_path = clip_dir / ADDITIONAL_INFO_FILENAME
                try:
                    additional_info = build_additional_info(
                        meta_to_save,
                        metadata_path=meta_path,
                    )
                    with open(additional_info_path, "w") as f:
                        json.dump(additional_info, f, indent=2)
                        f.write("\n")
                    print(f"Saved Additional Info: {additional_info_path}")
                except Exception as e:
                    print(f"Additional info generation failed for {meta_path}: {e}")

            # Save the plotted trajectory image. The eval BEV additionally emits
            # a lane_relative_plot.png companion: the same BEV with the
            # lane-center offset basis overlaid (see render_eval_visualization).
            plot_path = clip_dir / "trajectory_plot.png"
            lane_overlay_path = clip_dir / "lane_relative_plot.png"
            world_poses = [
                p for p in meta.get("trajectory_poses", []) if "x" in p and "y" in p
            ]
            if len(world_poses) >= 3:
                traj_world_xy = np.array(
                    [[p["x"], p["y"]] for p in world_poses], dtype=float
                )
                traj_world_ts = np.array(
                    [p.get("timestamp_us", 0) for p in world_poses], dtype=float
                )
            else:
                traj_world_xy = None
                traj_world_ts = None
            if sim_result is not None and eval_cfg is not None:
                try:
                    time_uint64 = np.uint64(meta["time_now_us"])
                    render_eval_visualization(
                        sim_result,
                        eval_cfg,
                        time_uint64,
                        plot_path,
                        trajectory_world_xy=traj_world_xy,
                        trajectory_timestamps=traj_world_ts,
                        lane_overlay_path=lane_overlay_path,
                        cot_text=meta.get("chain_of_thought"),
                    )
                except Exception as e:
                    print(
                        f"Eval visualization failed, fallback to trajectory plot. ({e})"
                    )
                    save_trajectory_plot(meta["trajectory_xy_rig_frame"], plot_path)
            else:
                save_trajectory_plot(meta["trajectory_xy_rig_frame"], plot_path)

            print(f"Saved Trajectory Plot: {plot_path}")

            pending_request = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract inferences, CoT, and images from ASL files to disk."
    )
    parser.add_argument("asl_path", type=str, help="Path to the rollout.asl file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_frames",
        help="Directory where structures will be saved",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="Target specific time_query_us or time_now_us to extract a single frame",
    )
    parser.add_argument(
        "--no-additional-info",
        action="store_true",
        help=f"Skip writing the {ADDITIONAL_INFO_FILENAME} sidecar next to each metadata.json.",
    )
    args = parser.parse_args()

    # Create an asyncio event loop to run the async generator
    asyncio.run(
        extract_inference(
            args.asl_path,
            args.output_dir,
            target_time_us=args.time,
            write_additional_info=not args.no_additional_info,
        )
    )
