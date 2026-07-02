# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
A CLI utility for extracting camera frames from alpasim logs (.asl files) and saving as video or
individual images. The results for each discovered file `path/to/<name>.asl` will be saved at
`path/to/<name>_asl_frames/<camera_id>/<mp4_or_jpegs_or_pngs>`.
The necessary dependencies for this script can be installed with the optional dependency: eg
`pip install alpasim_grpc_protobuf4[asl_to_frames]`.
"""

import argparse
import asyncio
import glob
import logging
from typing import Literal, TypeAlias

import aiofiles
import numpy as np
from aiofiles import os as aios
from alpasim_grpc.v0.egodriver_pb2 import DriveSessionRequest, RolloutCameraImage
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_utils.logs import async_read_pb_log

try:
    import imageio.v3 as iio
except ImportError:
    raise ImportError(
        "This script requires additionally installing imageio[ffmpeg] "
        + "or installing with the [asl_to_frames} optional dependency."
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

SaveFormat: TypeAlias = Literal["mp4", "frames"]


def pad_to_divisible_by_16(image: np.ndarray) -> np.ndarray:
    """
    Pads an image (h, w, 3) with zeros so that height and width are divisible by 16.

    Parameters:
    image (numpy.ndarray): Input image array of shape (h, w, 3).

    Returns:
    numpy.ndarray: Padded image with dimensions divisible by 16.
    """
    h, w, _ = image.shape

    # Calculate the padding required for height and width
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    # Calculate padding amounts
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the image
    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return padded_image


async def convert_single_log(
    log_path: str,
    save_dir: str,
    format: SaveFormat,
) -> None:
    frames_by_camera: dict[str, list[RolloutCameraImage.CameraImage]] = {}

    rollout_metadata: RolloutMetadata | None = None
    drive_session_request: DriveSessionRequest | None = None

    async for message in async_read_pb_log(log_path):
        if message.WhichOneof("log_entry") == "driver_session_request":
            drive_session_request = message.driver_session_request
        elif message.WhichOneof("log_entry") == "rollout_metadata":
            rollout_metadata = message.rollout_metadata
        elif message.WhichOneof("log_entry") == "driver_camera_image":
            image: RolloutCameraImage.CameraImage = (
                message.driver_camera_image.camera_image
            )
            frames_by_camera.setdefault(image.logical_id, []).append(image)

    if rollout_metadata is None:
        raise ValueError("RolloutMetadata not found in log; unknown rollout index.")

    if drive_session_request is None:
        raise ValueError("DriveSessionRequest not found in log; unknown camera IDs.")

    await aios.makedirs(save_dir, exist_ok=True)

    # --- GEOMETRY EXTRACTION BLOCK ---
    try:
        from omegaconf import OmegaConf
        from eval.schema import EvalConfig
        from eval.data import SimulationResult
        from eval.accumulator import EvalDataAccumulator
        from alpasim_utils.artifact import Artifact
        import os

        logger.info("Initializing evaluation models for geometry generation...")
        import glob
        artifacts = {}
        for path in glob.glob("data/**/*.usdz", recursive=True):
            try:
                a = Artifact(path, _smooth_trajectories=True)
                if a.scene_id not in artifacts:
                    artifacts[a.scene_id] = a
            except Exception:
                pass
        
        eval_cfg = OmegaConf.structured(EvalConfig)
        acc = EvalDataAccumulator(cfg=eval_cfg)
        
        async for m in async_read_pb_log(log_path):
            acc.handle_message(m)
            
        scene_id = rollout_metadata.session_metadata.scene_id if rollout_metadata and rollout_metadata.session_metadata else "unknown"
        vec_map = artifacts[scene_id].map if scene_id in artifacts else None
        scenario_input = acc.build_scenario_eval_input(
            run_uuid="extract",
            run_name="extract",
            vec_map=vec_map,
        )
        sim_result = SimulationResult.from_scenario_input(scenario_input, eval_cfg)
        
        all_timestamps = set()
        for images_list in frames_by_camera.values():
            for img in images_list:
                all_timestamps.add(img.frame_start_us)
        
        sorted_timestamps = sorted(list(all_timestamps))
        
        logger.info("Generating top-down geometry views...")
        bev_save_dir = f"{save_dir}/geometry_bev"
        await aios.makedirs(bev_save_dir, exist_ok=True)
        
        for ts in sorted_timestamps:
            out_img = f"{bev_save_dir}/{ts}.png"
            render_eval_visualization_and_extract_geometry(sim_result, eval_cfg, ts, out_img)
            
        if format == "mp4":
            import imageio.v3 as iio
            bev_images = []
            for ts in sorted_timestamps:
                img_path = f"{bev_save_dir}/{ts}.png"
                if os.path.exists(img_path):
                    # using the pad function from earlier
                    bev_images.append(pad_to_divisible_by_16(iio.imread(img_path)))
            
            if bev_images:
                duration_us = sorted_timestamps[-1] - sorted_timestamps[0] if len(sorted_timestamps) > 1 else 1e6
                average_fps = max(1, len(sorted_timestamps) / (duration_us / 1e6))
                iio.imwrite(f"{bev_save_dir}.mp4", image=bev_images, extension=".mp4", fps=average_fps)
                logger.info(f"Saved BEV geometry MP4 to {bev_save_dir}.mp4")

    except Exception as e:
        logger.warning(f"Could not extract geometry/simulate eval: {e}")
    # --- END GEOMETRY EXTRACTION BLOCK ---

    for camera_logical_id, images in frames_by_camera.items():
        camera_name = camera_logical_id
        save_path = f"{save_dir}/{camera_name}"
        images = sorted(images, key=lambda frame: frame.frame_start_us)
        timestamps_us = np.array([frame.frame_start_us for frame in images])

        match format:
            case "mp4":
                await frames_to_mp4(images, timestamps_us, save_path)
            case "frames":
                await save_frames_as_files(images, timestamps_us, save_path)
            case _:
                raise TypeError(f"Unknown {format=}")


async def frames_to_mp4(
    images: list[RolloutCameraImage.CameraImage],
    timestamps_us: np.ndarray,
    save_path: str,
) -> None:
    average_fps = 1 / (np.diff(timestamps_us).mean() / 1e6)

    bitmaps = [
        pad_to_divisible_by_16(iio.imread(image.image_bytes)) for image in images
    ]

    duration_us = timestamps_us[-1] - timestamps_us[0]
    duration_s = duration_us / 1e6

    save_path = f"{save_path}.mp4"

    logger.info(f"Saving {save_path=} (duration {duration_s:.2f}s).")
    iio.imwrite(
        save_path,
        image=bitmaps,
        extension=".mp4",
        fps=average_fps,
    )


async def _write_image(content: bytes, path: str) -> None:
    """Detects the format (JPG or PNG) and writes `content` to disk."""
    format: str
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        format = "png"
    elif content[:3] == b"\xff\xd8\xff":
        format = "jpg"
    else:
        raise ValueError("A frame could not be identified as either png or jpg")

    async with aiofiles.open(f"{path}.{format}", "wb") as file:
        await file.write(content)


async def save_frames_as_files(
    images: list[RolloutCameraImage.CameraImage],
    timestamps_us: np.ndarray,
    save_path: str,
) -> None:
    await aios.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving {save_path=} as frames.")
    await asyncio.gather(
        *[
            _write_image(image.image_bytes, f"{save_path}/{timestamp_us}")
            for image, timestamp_us in zip(images, timestamps_us)
        ]
    )


def determine_save_dir(log_path: str, log_save_dir: str | None) -> str:
    if log_save_dir is None:
        log_save_name = log_path.removesuffix(".asl")
        return f"{log_save_name}_asl_frames"
    else:
        # Note(mwatson): This code is taken from an earlier version of the kpi codebase.
        log_save_name = "/".join(
            log_path.removesuffix(".asl").split("/")[-3:]
        )  # clipgt/batch/rollout
        return f"{log_save_dir}/{log_save_name}"


async def convert_multiple_logs(
    asl_glob: str,
    format: SaveFormat,
    log_save_dir: str | None = None,
) -> None:
    assert asl_glob.endswith(".asl"), asl_glob

    log_paths = glob.glob(asl_glob, recursive=True)

    logger.info(f"Found {len(log_paths)} log files for conversion in {asl_glob=}.")

    async def convert_log_with_exception_handling(log_path: str, save_dir: str) -> None:
        try:
            await convert_single_log(
                log_path=log_path, save_dir=save_dir, format=format
            )
        except Exception as e:
            logger.error(f"Exception {e}, skipping {log_path=}.")

    tasks = []
    for (
        log_path
    ) in log_paths:  # tqdm is added in alpasim-kpi... should be added here too?
        save_dir = determine_save_dir(log_path, log_save_dir)

        tasks.append(convert_log_with_exception_handling(log_path, save_dir))

    await asyncio.gather(*tasks)

    logger.info(f"Converted {len(log_paths)} logs to {format=} in {log_save_dir=}.")


EXAMPLE_USAGE = (
    'Example usage: python -m alpasim_utils.asl_to_frames "path/to/logs/**/*.asl"'
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "asl_glob",
        type=str,
        help="Glob to find the asl files for conversion. To prevent expansion in shell quote it.",
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "frames"],
        default="mp4",
    )
    parser.add_argument(
        "--log-save-dir",
        type=str,
        default=None,
        help="Optional output directory. If not provided, saves alongside the .asl files.",
    )
    args = parser.parse_args()

    asyncio.run(
        convert_multiple_logs(
            args.asl_glob, format=args.format, log_save_dir=args.log_save_dir
        )
    )

def render_eval_visualization_and_extract_geometry(sim_result, cfg, time_now_us_target, output_image_path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    import json
    from pathlib import Path
    
    mpl.use("Agg")
    mplstyle.use("fast")
    try:
        from eval.video import get_ego_transform
        from eval.video_data import ShapelyMap
        from eval.schema import MapElements
    except ImportError:
        logger.error("eval package unavailable. Make sure 'eval' is in PYTHONPATH.")
        return

    
    if hasattr(sim_result, "actor_polygons") and sim_result.actor_polygons:
        sim_result.actor_polygons.artists = {}
    if hasattr(sim_result, "route") and sim_result.route:
        sim_result.route.artists = None
    if hasattr(sim_result, "driver_responses") and sim_result.driver_responses:
        sim_result.driver_responses.artists = None

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    # Using white background to match standard eval video
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3, color="#dddddd")

    shapely_map = ShapelyMap.from_vec_map(sim_result.vec_map) if sim_result.vec_map else None
    ego_transform = get_ego_transform(sim_result=sim_result, cfg=cfg, time=time_now_us_target)
    
    image_center_xy = sim_result.actor_polygons.set_axis_limits_around_agent(
        ax, "EGO", time_now_us_target, cfg, axis_transform=ego_transform
    )

    if shapely_map:
        shapely_map.render(
            ax, cfg, center=image_center_xy, max_dist=cfg.video.map_video.map_radius_m + 10
        )

    if cfg.video.map_video.map_elements_to_plot is None or MapElements.GT_LINESTRING in cfg.video.map_video.map_elements_to_plot:
        sim_result.ego_recorded_ground_truth_trajectory.set_linestring_plot_style(
            "gt_linestring", linewidth=1, style="g-", alpha=0.7
        ).render_linestring(ax)

    if cfg.video.map_video.map_elements_to_plot is None or MapElements.AGENTS in cfg.video.map_video.map_elements_to_plot:
        sim_result.actor_polygons.render_at_time(
            ax, time_now_us_target, center=image_center_xy, max_dist=cfg.video.map_video.map_radius_m + 10
        )
    else:
        sim_result.actor_polygons.render_at_time(ax, time_now_us_target, only_agents=["EGO"])

    if cfg.video.map_video.map_elements_to_plot is None or MapElements.DRIVER_RESPONSES in cfg.video.map_video.map_elements_to_plot:
        if hasattr(sim_result, "driver_responses"):
            sim_result.driver_responses.render_at_time(ax, time_now_us_target, "now")

    if (cfg.video.map_video.map_elements_to_plot is None or MapElements.ROUTE in cfg.video.map_video.map_elements_to_plot) and hasattr(sim_result, 'routes'):
        sim_result.routes.render_at_time(ax, time_now_us_target)

    if image_center_xy:
        r = cfg.video.map_video.map_radius_m
        ax.set_xlim(image_center_xy.x - r, image_center_xy.x + r)
        ax.set_ylim(image_center_xy.y - r + 0.3*r, image_center_xy.y + r + 0.3*r)
    fig.savefig(output_image_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Extract geometry to JSON
    geometry_path = Path(output_image_path).with_name(f"{Path(output_image_path).stem}_geometry.json")
    try:
        geometry_data = {
            "timestamp_us": int(time_now_us_target),
            "map_linestrings": [],
            "actors": []
        }
        if shapely_map:
            for ls in shapely_map.renderable_linestrings:
                geometry_data["map_linestrings"].append({
                    "type": str(ls.name),
                    "coords": list(ls.linestring.coords)
                })
        
        polys_at_time = sim_result.actor_polygons.get_polygons_at_time(time_now_us_target)
        for agent_id, poly, yaw in zip(polys_at_time.agent_ids, polys_at_time.bbox_polygons, polys_at_time.yaws):
            geometry_data["actors"].append({
                "id": agent_id,
                "yaw": float(yaw),
                "polygon_coords": list(poly.exterior.coords)
            })

        with open(geometry_path, "w") as f:
            json.dump(geometry_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to extract geometry JSON: {e}")
