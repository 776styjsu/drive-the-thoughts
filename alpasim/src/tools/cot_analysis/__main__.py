# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
CoT Consistency Analysis for AlpaSim simulation outputs.

Reads ASL files from AlpaSim rollouts, extracts per-timestep Chain-of-Thought
reasoning and predicted trajectories, then evaluates self-consistency with an
LLM judge via an OpenAI-compatible API. Produces structured per-dimension
scores.

Three model backends are selectable with --provider:
    gateway       - Kimi K2.5 via the institutional GenAI gateway (key: GENAI_GATEWAY_KEY)
    openai        - GPT-5.5 with high reasoning effort (key: OPENAI_API_KEY)
    qwen3_4b_fp8  - Qwen3-4B-FP8 via a local vLLM server
    qwen35_4b_fp8 - Qwen3.5-4B-FP8 via a local vLLM server

Keys are read from the matching environment variable, also loaded from a .env
file at/above the working directory. Backends decode deterministically
(temperature=0 plus a fixed --seed).

Usage:
    # Dry run (no API key, verifies trajectory analysis):
    uv run python -m cot_analysis \
        --asl_glob "tutorial_alpamayo/rollouts/**/*.asl" \
        --output /tmp/cot_dry.json

    # Full run with Kimi K2.5 (uses GENAI_GATEWAY_KEY from .env / environment):
    uv run python -m cot_analysis \
        --asl_glob "tutorial_alpamayo/rollouts/**/*.asl" \
        --output tutorial_alpamayo/cot_consistency.json

    # Full run with GPT-5.5 high (uses OPENAI_API_KEY from .env / environment):
    uv run python -m cot_analysis --provider openai \
        --asl_glob "tutorial_alpamayo/rollouts/**/*.asl" \
        --output tutorial_alpamayo/cot_consistency_gpt55.json

    # Full run with local Qwen3-4B FP8 non-thinking mode:
    tools/serve_qwen3_4b_fp8_vllm.sh serve
    uv run python -m cot_analysis --provider qwen3_4b_fp8 \
        --asl_glob "tutorial_alpamayo/rollouts/**/*.asl" \
        --output tutorial_alpamayo/cot_consistency_qwen3_4b_fp8.json

    # Full run with local Qwen3.5-4B FP8 non-thinking mode:
    tools/serve_qwen35_4b_fp8_vllm.sh serve
    uv run python -m cot_analysis --provider qwen35_4b_fp8 \
        --asl_glob "tutorial_alpamayo/rollouts/**/*.asl" \
        --output tutorial_alpamayo/cot_consistency_qwen35_4b_fp8.json
"""

import argparse
import asyncio
import base64
import glob
import importlib.util
import io
import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from alpasim_utils.lane_projection import lane_center_lines_from_geometry
from PIL import Image

try:
    from .trajectory import compute_trajectory_features
except ImportError:
    from trajectory import compute_trajectory_features


logger = logging.getLogger(__name__)

# Dimensions to score
DIMENSIONS = ["cot_output_alignment"]

# Prompt resolution
# -----------------
# Prompt templates live alongside this module as ``prompt_<name>.py`` files,
# each exposing ``build_prompt(cot_text, traj_features) -> str``. They are
# auto-discovered, so adding a new template needs no edits here: just drop in
# the file and pass ``--prompt <name>``. PROMPT_ALIASES covers the names that
# don't map 1:1 to a ``prompt_<name>.py`` file — ``default`` lives in
# ``prompt.py``, and ``hybrid_ego`` reuses ``prompt_hybrid``'s builder but pairs
# it with different trajectory-feature handling in process_entry().
PROMPT_ALIASES = {
    "default": "prompt",
    "hybrid_ego": "prompt_hybrid",
}

_PROMPT_DIR = Path(__file__).resolve().parent
_BUILDER_CACHE: dict = {}


def discover_prompt_names() -> list[str]:
    """List selectable prompt names: aliases plus every prompt_<name>.py file."""
    names = set(PROMPT_ALIASES)
    for path in _PROMPT_DIR.glob("prompt_*.py"):
        names.add(path.stem[len("prompt_") :])
    return sorted(names)


def _load_build_prompt(module_path: Path):
    """Import a prompt module from a file path and return its build_prompt."""
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_prompt"):
        raise AttributeError(
            f"{module_path} has no build_prompt(cot_text, traj_features)"
        )
    return module.build_prompt


def resolve_prompt_builder(name: str):
    """Resolve a --prompt value to a build_prompt callable.

    Accepts, in order: a path to a .py file (one-off experiments), a known
    alias (default, hybrid_ego), or an auto-discovered prompt_<name>.py module.
    """
    if name in _BUILDER_CACHE:
        return _BUILDER_CACHE[name]

    path_candidate = Path(name)
    if path_candidate.suffix == ".py":
        if not path_candidate.exists():
            raise FileNotFoundError(f"Prompt file not found: {name}")
        module_path = path_candidate
    else:
        stem = PROMPT_ALIASES.get(name, f"prompt_{name}")
        module_path = _PROMPT_DIR / f"{stem}.py"
        if not module_path.exists():
            raise ValueError(
                f"Unknown prompt '{name}'. Available: "
                f"{', '.join(discover_prompt_names())}, or a path to a .py file "
                f"defining build_prompt()."
            )

    builder = _load_build_prompt(module_path)
    _BUILDER_CACHE[name] = builder
    return builder


# institutional GenAI gateway (OpenAI-compatible gateway running Kimi K2.5)
DEFAULT_BASE_URL = "https://genai-gateway.example.edu/api"
DEFAULT_MODEL = "Kimi K2.5"
API_KEY_ENV = "GENAI_GATEWAY_KEY"
BASE_URL_ENV = "GENAI_GATEWAY_BASE_URL"
QWEN3_4B_FP8_MODEL = "Qwen/Qwen3-4B-FP8"
QWEN35_4B_FP8_MODEL = "RedHatAI/Qwen3.5-4B-FP8-dynamic"
QWEN3_LOCAL_BASE_URL = "http://localhost:8000/v1"

# Fixed seed for deterministic decoding. Providers combine this with their
# sampling defaults so repeated evaluations are reproducible.
DEFAULT_SEED = 42

# Selectable model backends. Each provider resolves its own API key / base URL
# from the environment (loaded from .env) and may add extra request params
# (e.g. reasoning_effort for OpenAI's GPT-5.5 reasoning model).
PROVIDERS: dict[str, dict] = {
    "gateway": {
        "label": "Institutional gateway (Kimi K2.5)",
        "model": DEFAULT_MODEL,
        "api_key_env": API_KEY_ENV,
        "base_url_env": BASE_URL_ENV,
        "base_url": DEFAULT_BASE_URL,
        "temperature": 0,
        "extra_params": {},
    },
    "openai": {
        "label": "OpenAI GPT-5.5 (high reasoning)",
        "model": "gpt-5.5",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": None,  # use the OpenAI SDK default endpoint
        # GPT-5.5 is a reasoning model: it rejects temperature != 1 (default),
        # so we omit temperature and rely on the fixed seed for determinism.
        "temperature": None,
        "extra_params": {"reasoning_effort": "high"},
    },
    "qwen3_4b_fp8": {
        "label": "Local Qwen3-4B-FP8 via vLLM (non-thinking)",
        "model": QWEN3_4B_FP8_MODEL,
        "api_key_env": "QWEN3_API_KEY",
        "base_url_env": "QWEN3_BASE_URL",
        "base_url": QWEN3_LOCAL_BASE_URL,
        "default_api_key": "EMPTY",
        "temperature": 0.0, # default 0.7
        "supports_images": False,
        "extra_params": {
            # The local server is usually launched with --max-model-len 8192.
            # Keep enough context budget for the benchmark prompt.
            "max_tokens": 1024,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    },
    "qwen35_4b_fp8": {
        "label": "Local Qwen3.5-4B-FP8 via vLLM (non-thinking)",
        "model": QWEN35_4B_FP8_MODEL,
        "api_key_env": "QWEN35_API_KEY",
        "base_url_env": "QWEN35_BASE_URL",
        "base_url": QWEN3_LOCAL_BASE_URL,
        "default_api_key": "EMPTY",
        "temperature": 0.0, # default 0.7
        "supports_images": False,
        "extra_params": {
            # The local server is usually launched with --max-model-len 8192.
            # Keep enough context budget for the benchmark prompt.
            "max_tokens": 1024,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        },
    },
}

# Some OpenAI-compatible gateways reject the response_format parameter. We
# attempt JSON mode first and disable it for the rest of the run if rejected.
_USE_JSON_MODE = True


# =============================================================================
# .env loading (no external dependency)
# =============================================================================


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file (existing vars take precedence)."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _find_and_load_dotenv() -> None:
    """Search for a .env file from the cwd and module dir upward, then load it."""
    seen: set[Path] = set()
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        current = base
        while True:
            candidate = current / ".env"
            if candidate not in seen and candidate.exists():
                _load_dotenv(candidate)
                return
            seen.add(candidate)
            if current.parent == current:
                break
            current = current.parent


# =============================================================================
# ASL Data Extraction
# =============================================================================


def _extract_debug_info(driver_response) -> dict | None:
    """Extract and unpickle debug dict from a DriveResponse."""
    try:
        dbg_bytes = driver_response.debug_info.unstructured_debug_info
        if not dbg_bytes:
            return None
        extra = pickle.loads(dbg_bytes)
        if isinstance(extra, dict):
            return extra
        return None
    except Exception:
        return None


def _trajectory_poses_to_xy(trajectory) -> np.ndarray | None:
    """Convert a gRPC Trajectory to a (T, 2) rig-frame XY array.

    The DriveResponse trajectory is in the local frame. The poses are
    expressed relative to the first pose (current ego position), so the
    first point is near origin and subsequent points are the predicted
    future path in rig-frame coordinates.
    """
    poses = trajectory.poses
    if len(poses) < 3:
        return None

    # The first pose is the current position (anchor).
    # Subsequent poses are the planned future trajectory in local frame.
    # Convert to rig-frame relative to the anchor.
    anchor = poses[0]
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

    xy = []
    for pose_at_time in poses[1:]:
        p = pose_at_time.pose
        dx = p.vec.x - anchor_x
        dy = p.vec.y - anchor_y
        # Rotate into rig frame (relative to ego heading)
        rx = cos_yaw * dx - sin_yaw * dy
        ry = sin_yaw * dx + cos_yaw * dy
        xy.append([rx, ry])

    return np.array(xy, dtype=float) if xy else None


def _trajectory_poses_to_map_xy(trajectory) -> np.ndarray | None:
    """Convert a gRPC Trajectory to raw map/local-frame XY positions."""
    poses = trajectory.poses
    if len(poses) < 3:
        return None
    xy = [
        [pose_at_time.pose.vec.x, pose_at_time.pose.vec.y] for pose_at_time in poses[1:]
    ]
    return np.array(xy, dtype=float)


def _route_waypoints_to_xy(route) -> np.ndarray | None:
    """Convert a gRPC Route to route/lane-center XY waypoints in rig frame."""
    if len(route.waypoints) < 2:
        return None
    xy = [[wp.x, wp.y] for wp in route.waypoints]
    return np.array(xy, dtype=float)


def _scene_id_from_clip_id(clip_id: str | None) -> str | None:
    if not clip_id or "_t" not in clip_id:
        return None
    return clip_id.rsplit("_t", 1)[0]


def _read_lane_center_lines(geometry_path: Path) -> list[np.ndarray] | None:
    """Read road lane-center polylines from one trajectory geometry JSON."""
    try:
        with geometry_path.open("r") as f:
            geometry = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping invalid geometry file %s: %s", geometry_path, exc)
        return None

    return lane_center_lines_from_geometry(geometry)


def _load_lane_geometry_index(root: str | None) -> dict | None:
    """Index extracted-frame lane geometry paths by scene/time."""
    if not root:
        return None

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Lane geometry root does not exist: {root}")

    by_scene_time: dict[tuple[str, int], Path] = {}
    by_time: dict[int, Path] = {}
    count = 0

    for geometry_path in root_path.rglob("trajectory_plot_geometry.json"):
        scene_id = None
        timestamp_us = None
        metadata_path = geometry_path.with_name("metadata.json")
        if metadata_path.exists():
            try:
                with metadata_path.open("r") as f:
                    metadata = json.load(f)
                timestamp_us = metadata.get("time_now_us")
                scene_id = _scene_id_from_clip_id(metadata.get("clip_id"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read metadata for %s: %s", geometry_path, exc)

        if timestamp_us is None:
            try:
                with geometry_path.open("r") as f:
                    timestamp_us = json.load(f).get("timestamp_us")
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Could not read timestamp from %s: %s", geometry_path, exc
                )
                continue

        timestamp_us = int(timestamp_us)
        by_time[timestamp_us] = geometry_path
        if scene_id is not None:
            by_scene_time[(scene_id, timestamp_us)] = geometry_path
        count += 1

    logger.info("Indexed %d lane geometry frame(s) from %s", count, root_path)
    return {
        "by_scene_time": by_scene_time,
        "by_time": by_time,
        "cache": {},
        "count": count,
    }


def _find_lane_center_lines(
    lane_geometry_index: dict | None,
    scene_id: str,
    timestamp_us: int,
) -> list[np.ndarray] | None:
    if not lane_geometry_index:
        return None
    by_scene_time = lane_geometry_index.get("by_scene_time", {})
    geometry_path = by_scene_time.get((scene_id, timestamp_us))
    if geometry_path is None:
        geometry_path = lane_geometry_index.get("by_time", {}).get(timestamp_us)
    if geometry_path is None:
        return None

    cache = lane_geometry_index.get("cache", {})
    if geometry_path not in cache:
        cache[geometry_path] = _read_lane_center_lines(geometry_path)
    return cache[geometry_path]


async def extract_entries_from_asl(
    asl_path: str,
    camera_id: str = "camera_front_wide_120fov",
    every_nth: int = 1,
    lane_geometry_index: dict | None = None,
) -> list[dict]:
    """Parse an ASL file and extract per-timestep entries for analysis.

    Args:
        asl_path: Path to the .asl file.
        camera_id: Camera logical ID for image extraction.
        every_nth: Only keep every Nth timestep (1 = keep all).

    Returns:
        List of dicts, each with keys:
            clip_id, chain_of_thought, trajectory_xy, image (PIL or None),
            timestamp_us
    """
    from alpasim_utils.logs import async_read_pb_log

    # Collect raw data from ASL messages
    scene_id = "unknown"
    pending_request: tuple[int, int] | None = None
    driver_responses: list[tuple[int, int, object]] = []  # (now_us, query_us, resp)
    camera_frames: dict[int, bytes] = {}  # timestamp_us -> image_bytes
    route_frames: dict[int, np.ndarray] = {}  # timestamp_us -> route XY waypoints

    async for message in async_read_pb_log(asl_path):
        msg_type = message.WhichOneof("log_entry")

        if msg_type == "rollout_metadata":
            if message.rollout_metadata.session_metadata is not None:
                scene_id = message.rollout_metadata.session_metadata.scene_id

        elif msg_type == "driver_request":
            pending_request = (
                message.driver_request.time_now_us,
                message.driver_request.time_query_us,
            )

        elif msg_type == "driver_return":
            if pending_request is not None:
                driver_responses.append((*pending_request, message.driver_return))
                pending_request = None

        elif msg_type == "driver_camera_image":
            cam_img = message.driver_camera_image.camera_image
            if cam_img.logical_id == camera_id:
                camera_frames[cam_img.frame_end_us] = cam_img.image_bytes

        elif msg_type == "route_request":
            route_xy = _route_waypoints_to_xy(message.route_request.route)
            if route_xy is not None:
                route_frames[message.route_request.route.timestamp_us] = route_xy

    # Build entries from driver responses
    entries = []
    for idx, (now_us, _query_us, resp) in enumerate(driver_responses):
        if every_nth > 1 and idx % every_nth != 0:
            continue

        # Skip empty trajectories (warmup period)
        if len(resp.trajectory.poses) < 3:
            continue

        # Extract CoT
        extra = _extract_debug_info(resp)
        reasoning_text = None
        if extra is not None:
            reasoning_text = extra.get("reasoning_text")
        if reasoning_text is None:
            continue  # no CoT for this timestep

        # Extract trajectory (rig-frame XY)
        trajectory_xy = _trajectory_poses_to_xy(resp.trajectory)
        if trajectory_xy is None:
            continue
        trajectory_map_xy = _trajectory_poses_to_map_xy(resp.trajectory)

        # Find closest route/lane-center reference.
        route_xy = None
        if route_frames:
            valid_ts = [ts for ts in route_frames if ts <= now_us]
            if valid_ts:
                route_xy = route_frames[max(valid_ts)]
        lane_center_lines = _find_lane_center_lines(
            lane_geometry_index,
            scene_id,
            now_us,
        )

        # Find closest camera image
        image = None
        if camera_frames:
            # Find the camera frame with timestamp closest to (but <= ) now_us
            valid_ts = [ts for ts in camera_frames if ts <= now_us]
            if valid_ts:
                closest_ts = max(valid_ts)
                try:
                    image = Image.open(io.BytesIO(camera_frames[closest_ts]))
                except Exception:
                    pass

        entry = {
            "clip_id": f"{scene_id}_t{idx:04d}",
            "chain_of_thought": reasoning_text,
            "trajectory_xy": trajectory_xy,
            "trajectory_map_xy": trajectory_map_xy,
            "route_xy": route_xy,
            "lane_center_lines": lane_center_lines,
            "image": image,
            "timestamp_us": now_us,
        }
        entries.append(entry)

    logger.info(
        "Extracted %d entries from %s (scene: %s, total driver responses: %d)",
        len(entries),
        os.path.basename(asl_path),
        scene_id,
        len(driver_responses),
    )
    return entries


# =============================================================================
# institutional GenAI gateway API (OpenAI-compatible SDK)
# =============================================================================


def _create_client(api_key: str, base_url: str):
    """Create and return an OpenAI client pointed at the institutional GenAI gateway."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _image_to_data_url(image) -> str | None:
    """Encode a PIL image as a base64 JPEG data URL for the chat API."""
    try:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_llm(
    client,
    model_name: str,
    prompt: str,
    image=None,
    *,
    seed: int = DEFAULT_SEED,
    temperature: float | None = 0,
    extra_params: dict | None = None,
    return_timing: bool = False,
) -> str | tuple[str, dict]:
    """Send prompt (+ optional image) to the model and return response text.

    Decoding is deterministic via a fixed seed. ``temperature`` is sent only
    when not None (GPT-5.5 and other reasoning models reject any value other
    than the default, so the openai provider passes None). ``extra_params``
    carries provider-specific request fields (e.g. ``reasoning_effort``).
    """
    global _USE_JSON_MODE

    user_content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        data_url = _image_to_data_url(image)
        if data_url is not None:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for autonomous vehicle reasoning "
                "systems. Respond with ONLY a single JSON object, no markdown."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    timing: dict = {
        "model": model_name,
        "stream": True,
        "started_at_utc": _utc_now_iso(),
        "attempts": [],
    }
    total_start = time.perf_counter()

    def _create(use_json: bool) -> str:
        attempt_start = time.perf_counter()
        attempt: dict = {
            "json_mode": use_json,
            "started_at_utc": _utc_now_iso(),
            "status": "started",
            "chunk_count": 0,
            "content_chunk_count": 0,
            "output_chars": 0,
        }
        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "seed": seed,
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if extra_params:
            kwargs.update(extra_params)
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        # The institutional GenAI gateway (Open WebUI) always streams SSE and ignores
        # stream=False, so consume the stream and accumulate the content delta.
        # Kimi also emits a separate `delta.reasoning` field which we drop.
        parts: list[str] = []
        first_chunk_elapsed_s = None
        first_content_elapsed_s = None
        try:
            stream = client.chat.completions.create(**kwargs)
            attempt["stream_open_elapsed_s"] = time.perf_counter() - attempt_start
            for chunk in stream:
                now = time.perf_counter()
                attempt["chunk_count"] += 1
                if first_chunk_elapsed_s is None:
                    first_chunk_elapsed_s = now - attempt_start
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    attempt["content_chunk_count"] += 1
                    if first_content_elapsed_s is None:
                        first_content_elapsed_s = now - attempt_start
                    parts.append(delta.content)
            response_text = "".join(parts)
            attempt["status"] = "ok"
            attempt["output_chars"] = len(response_text)
            return response_text
        except Exception as exc:
            attempt["status"] = "error"
            attempt["error_type"] = type(exc).__name__
            attempt["error"] = str(exc)
            raise
        finally:
            attempt["first_chunk_elapsed_s"] = first_chunk_elapsed_s
            attempt["first_content_elapsed_s"] = first_content_elapsed_s
            attempt["total_elapsed_s"] = time.perf_counter() - attempt_start
            attempt["ended_at_utc"] = _utc_now_iso()
            timing["attempts"].append(attempt)
            log_level = logging.INFO if attempt["status"] == "ok" else logging.WARNING
            logger.log(
                log_level,
                "  llm_attempt_timing: status=%s json_mode=%s "
                "stream_open=%.3fs first_chunk=%s first_content=%s "
                "total=%.3fs chunks=%d content_chunks=%d output_chars=%d",
                attempt["status"],
                use_json,
                attempt.get("stream_open_elapsed_s", 0.0),
                (
                    f"{first_chunk_elapsed_s:.3f}s"
                    if first_chunk_elapsed_s is not None
                    else "none"
                ),
                (
                    f"{first_content_elapsed_s:.3f}s"
                    if first_content_elapsed_s is not None
                    else "none"
                ),
                attempt["total_elapsed_s"],
                attempt["chunk_count"],
                attempt["content_chunk_count"],
                attempt["output_chars"],
            )

    try:
        result = _create(_USE_JSON_MODE)
        timing["status"] = "ok"
    except Exception as e:
        # The gateway may reject response_format; retry once without it and
        # disable JSON mode for the remainder of the run.
        if _USE_JSON_MODE:
            try:
                result = _create(False)
                _USE_JSON_MODE = False
                timing["status"] = "ok_after_retry"
                logger.warning(
                    "Disabling JSON response_format (server rejected it): %s", e
                )
            except Exception as e2:
                result = json.dumps({"error": str(e2)})
                timing["status"] = "error"
                timing["error_type"] = type(e2).__name__
                timing["error"] = str(e2)
        else:
            result = json.dumps({"error": str(e)})
            timing["status"] = "error"
            timing["error_type"] = type(e).__name__
            timing["error"] = str(e)

    timing["retry_count"] = max(0, len(timing["attempts"]) - 1)
    timing["total_elapsed_s"] = time.perf_counter() - total_start
    timing["ended_at_utc"] = _utc_now_iso()
    timing["output_chars"] = len(result)
    logger.info(
        "  llm_total_timing: status=%s attempts=%d total=%.3fs output_chars=%d",
        timing["status"],
        len(timing["attempts"]),
        timing["total_elapsed_s"],
        len(result),
    )
    if return_timing:
        return result, timing
    return result


def parse_response(response_text: str) -> dict:
    """Parse structured JSON response from the model."""
    if not response_text:
        return {"parse_error": True, "raw_response": response_text}
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                return {"parse_error": True, "raw_response": response_text}
        if not isinstance(parsed, dict):
            return {"parse_error": True, "raw_response": response_text}
        return parsed
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_response": response_text}


# =============================================================================
# Processing & Aggregation
# =============================================================================


def flatten_cot(cot) -> str:
    """Recursively flatten CoT from nested lists to a single string."""
    if isinstance(cot, str):
        return cot
    if isinstance(cot, list):
        return " ".join(flatten_cot(item) for item in cot)
    return str(cot)


def _resolve_benchmark_source_path(
    source_scene_file: str | None,
    benchmark_path: Path,
    source_root: Path,
) -> Path | None:
    """Resolve a benchmark source_scene_file against likely local roots."""
    if not source_scene_file:
        return None

    source_path = Path(source_scene_file)
    candidates = []
    if source_path.is_absolute():
        candidates.append(source_path)
        if not source_path.exists():
            for marker in ("tutorial_alpamayo", "tutorial_alpamayo15_first20"):
                if marker in source_path.parts:
                    marker_idx = source_path.parts.index(marker)
                    candidates.append(
                        source_root.joinpath(*source_path.parts[marker_idx + 1 :])
                    )
    else:
        candidates.extend(
            [
                Path.cwd() / source_path,
                source_root / source_path,
                benchmark_path.parent / source_path,
            ]
        )
        if source_root.exists():
            for child in source_root.iterdir():
                if child.is_dir():
                    candidates.append(child / source_path)
        if benchmark_path.parent.exists():
            for child in benchmark_path.parent.iterdir():
                if child.is_dir():
                    candidates.append(child / source_path)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _metadata_trajectory_to_xy(
    metadata_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """Load metadata.json trajectory poses as relative and map-frame XY points."""
    if not metadata_path.exists():
        return None, None, None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    poses = metadata.get("trajectory_poses") or []
    if len(poses) < 3:
        return None, None, metadata.get("time_now_us")

    map_xy = np.asarray([[pose["x"], pose["y"]] for pose in poses], dtype=float)
    relative_xy = map_xy - map_xy[0]
    return relative_xy, map_xy, metadata.get("time_now_us")


def _find_benchmark_metadata_path(
    source_path: Path | None, clip_id: str | None
) -> Path | None:
    """Find metadata.json adjacent to a benchmark source report."""
    if source_path is None or not clip_id:
        return None

    candidates = [source_path.with_name("metadata.json")]
    for parent in source_path.parents:
        candidates.append(parent / "rollouts" / "frames" / clip_id / "metadata.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_benchmark_geometry_path(
    source_path: Path | None,
    metadata_path: Path | None,
    clip_id: str | None,
) -> Path | None:
    """Find trajectory_plot_geometry.json for a benchmark clip."""
    candidates = []
    if metadata_path is not None:
        candidates.append(metadata_path.with_name("trajectory_plot_geometry.json"))

    if source_path is not None and clip_id:
        candidates.append(source_path.with_name("trajectory_plot_geometry.json"))
        for parent in source_path.parents:
            candidates.append(
                parent
                / "rollouts"
                / "frames"
                / clip_id
                / "trajectory_plot_geometry.json"
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_benchmark_additional_info_path(
    source_path: Path | None,
    metadata_path: Path | None,
    clip_id: str | None,
) -> Path | None:
    """Find additional_info.json carrying trajectory meta-action hints."""
    candidates = []
    if metadata_path is not None:
        candidates.append(metadata_path.with_name("additional_info.json"))

    if source_path is not None and clip_id:
        candidates.append(source_path.with_name("additional_info.json"))
        for parent in source_path.parents:
            candidates.append(
                parent / "rollouts" / "frames" / clip_id / "additional_info.json"
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_additional_info(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read additional trajectory info from %s: %s", path, exc
        )
        return None
    return payload if isinstance(payload, dict) else None


def _compact_trajectory_meta_actions(meta_actions: dict | None) -> dict | None:
    """Trim additional_info.meta_actions to prompt-sized trajectory hints."""
    if not isinstance(meta_actions, dict):
        return None
    compact = {
        "source": "additional_info.meta_actions",
        "coordinate_frame": meta_actions.get("coordinate_frame"),
        "lateral_side_reliable": meta_actions.get("lateral_side_reliable"),
        "dominant": meta_actions.get("dominant"),
        "longitudinal_distribution": meta_actions.get("longitudinal_distribution"),
        "lateral_distribution": meta_actions.get("lateral_distribution"),
        "transition_count": meta_actions.get("transition_count"),
    }
    transitions = meta_actions.get("transitions")
    if isinstance(transitions, list) and transitions:
        compact["first_transitions"] = transitions[:5]
    return {k: v for k, v in compact.items() if v is not None}


def _speed_delta_from_rows(features: dict) -> float | None:
    rows = features.get("table_rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    try:
        return float(rows[-1]["speed"]) - float(rows[0]["speed"])
    except (KeyError, TypeError, ValueError):
        return None


def _numeric_trajectory_action_hints(traj_features: dict) -> dict:
    """Build lightweight fallback hints from computed trajectory features."""
    stats = traj_features.get("summary_stats", {})
    ego_stats = stats.get("ego", stats) if isinstance(stats, dict) else {}
    lane_stats = stats.get("lane", {}) if isinstance(stats, dict) else {}
    ego_features = traj_features.get("ego_features", traj_features)
    speed_delta = _speed_delta_from_rows(ego_features)

    if speed_delta is None:
        longitudinal_trend = "unknown"
    elif speed_delta > 1.0:
        longitudinal_trend = "speeding_up"
    elif speed_delta < -1.0:
        longitudinal_trend = "slowing_down"
    else:
        longitudinal_trend = "roughly_steady_speed"

    lateral_basis = "ego_frame"
    lateral_value = ego_stats.get("final_lateral_m")
    if (
        isinstance(lane_stats, dict)
        and lane_stats.get("reference_frame") == "lane_center"
    ):
        lateral_basis = "lane_relative_delta_offset"
        lateral_value = lane_stats.get("delta_offset_m")

    try:
        lateral_value_float = float(lateral_value)
    except (TypeError, ValueError):
        lateral_value_float = None

    if lateral_value_float is None:
        lateral_trend = "unknown"
    elif lateral_value_float > 2.5:
        lateral_trend = "large_left_shift"
    elif lateral_value_float < -2.5:
        lateral_trend = "large_right_shift"
    elif lateral_value_float > 0.5:
        lateral_trend = "small_left_shift"
    elif lateral_value_float < -0.5:
        lateral_trend = "small_right_shift"
    else:
        lateral_trend = "minimal_lateral_shift"

    return {
        "source": "computed_trajectory_features",
        "longitudinal_trend": longitudinal_trend,
        "speed_delta_ms": round(speed_delta, 2) if speed_delta is not None else None,
        "mean_accel_ms2": ego_stats.get("mean_accel_ms2"),
        "lateral_trend": lateral_trend,
        "lateral_basis": lateral_basis,
        "lateral_value_m": lateral_value,
    }


def _trajectory_action_hints_for_prompt(traj_features: dict, entry: dict) -> dict:
    meta_hints = _compact_trajectory_meta_actions(entry.get("trajectory_meta_actions"))
    numeric_hints = _numeric_trajectory_action_hints(traj_features)
    if meta_hints is None:
        return numeric_hints
    return {
        "deterministic_meta_actions": meta_hints,
        "numeric_fallback_summary": numeric_hints,
    }


def _ego_only_trajectory_features(traj_features: dict) -> dict:
    """Return an ego-frame feature payload, stripping lane views if cached."""
    if not isinstance(traj_features, dict):
        return traj_features

    ego_features = traj_features.get("ego_features")
    if isinstance(ego_features, dict):
        return dict(ego_features)

    stats = traj_features.get("summary_stats", {})
    if isinstance(stats, dict) and stats.get("reference_frame") == "lane_center":
        return {
            "summary_stats": {
                "reference_frame": "ego_rig",
                "ego_frame_error": (
                    "Ego-only hybrid requested, but only lane-center cached "
                    "features were available."
                ),
            },
            "table_rows": [],
            "markdown_kv": "No ego-frame trajectory data available.",
        }

    return traj_features


def extract_entries_from_benchmark(
    benchmark_json: str,
    source_root: str = ".",
    load_raw_trajectory: bool = False,
) -> list[dict]:
    """Load benchmark-selected entries from prior cot_analysis result files.

    benchmark_expanded_50.json stores curated clip IDs plus source_scene_file
    pointers, not raw ASL trajectories. The source cot_consistency reports
    already include trajectory_features, so benchmark mode reuses those by
    default. Set load_raw_trajectory when the requested feature frame may need
    recomputation, e.g. lane-center features from map geometry.
    """
    benchmark_path = Path(benchmark_json)
    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if isinstance(benchmark, dict):
        benchmark_items = benchmark.get("results", benchmark.get("entries", []))
    else:
        benchmark_items = benchmark
    if not isinstance(benchmark_items, list):
        raise ValueError(
            f"Expected benchmark JSON list or results list: {benchmark_json}"
        )

    source_root_path = Path(source_root)
    report_cache: dict[Path, dict[str, dict]] = {}
    entries = []

    for item in benchmark_items:
        if not isinstance(item, dict):
            continue

        clip_id = item.get("clip_id")
        source_path = _resolve_benchmark_source_path(
            item.get("source_scene_file"),
            benchmark_path,
            source_root_path,
        )

        source_result = None
        error = None
        metadata_path = None
        if source_path is None:
            error = (
                f"could not resolve source_scene_file: {item.get('source_scene_file')}"
            )
        else:
            if source_path not in report_cache:
                with source_path.open("r", encoding="utf-8") as f:
                    source_report = json.load(f)
                if isinstance(source_report, dict) and source_report.get("clip_id"):
                    source_results = [source_report]
                else:
                    source_results = source_report.get("results", source_report)
                if not isinstance(source_results, list):
                    source_results = []
                report_cache[source_path] = {
                    result.get("clip_id"): result
                    for result in source_results
                    if isinstance(result, dict) and result.get("clip_id")
                }
            source_result = report_cache[source_path].get(clip_id)
            if source_result is None:
                error = f"clip_id not found in source report: {clip_id}"
                metadata_path = _find_benchmark_metadata_path(source_path, clip_id)
                if metadata_path is not None:
                    source_result = {"clip_id": clip_id}
                    error = None

        trajectory_xy = None
        trajectory_map_xy = None
        lane_center_lines = None
        additional_info = None
        timestamp_us = source_result.get("timestamp_us") if source_result else None
        should_load_raw = load_raw_trajectory or (
            source_result is not None
            and source_result.get("trajectory_features") is None
        )
        if source_result and should_load_raw and source_path:
            try:
                metadata_path = metadata_path or _find_benchmark_metadata_path(
                    source_path, clip_id
                )
                trajectory_xy, trajectory_map_xy, metadata_timestamp_us = (
                    _metadata_trajectory_to_xy(
                        metadata_path or source_path.with_name("metadata.json")
                    )
                )
                timestamp_us = timestamp_us or metadata_timestamp_us
                if trajectory_xy is None:
                    error = "metadata.json has no usable trajectory_poses"
                geometry_path = _find_benchmark_geometry_path(
                    source_path,
                    metadata_path,
                    clip_id,
                )
                if geometry_path is not None:
                    lane_center_lines = _read_lane_center_lines(geometry_path)
            except Exception as exc:
                error = f"failed to load trajectory from metadata.json: {exc}"
        if source_result and source_path:
            try:
                metadata_path = metadata_path or _find_benchmark_metadata_path(
                    source_path, clip_id
                )
                additional_info = _read_additional_info(
                    _find_benchmark_additional_info_path(
                        source_path,
                        metadata_path,
                        clip_id,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Could not load additional trajectory info for %s: %s",
                    clip_id,
                    exc,
                )

        entries.append(
            {
                "clip_id": clip_id,
                "chain_of_thought": item.get(
                    "chain_of_thought",
                    source_result.get("chain_of_thought") if source_result else "",
                ),
                "trajectory_features": (
                    source_result.get("trajectory_features") if source_result else None
                ),
                "trajectory_xy": trajectory_xy,
                "trajectory_map_xy": trajectory_map_xy,
                "lane_center_lines": lane_center_lines,
                "metadata_path": (
                    str(metadata_path) if metadata_path is not None else None
                ),
                "trajectory_meta_actions": (
                    additional_info.get("meta_actions") if additional_info else None
                ),
                "trajectory_dynamics_summary": (
                    additional_info.get("summary") if additional_info else None
                ),
                "timestamp_us": timestamp_us,
                "benchmark": item,
                "source_report": str(source_path) if source_path else None,
                "error": error,
            }
        )

    logger.info(
        "Loaded %d benchmark entries from %s using %d source report(s)",
        len(entries),
        benchmark_path,
        len(report_cache),
    )
    return entries


def _cot_reliability_flag(item: dict | None) -> bool | None:
    """CoT reliability from a benchmark entry, supporting both on-disk schemas.

    - flat ``cot_reliable`` (bool/str), used by benchmark_expanded_*.json
    - nested ``cot_reliability.reliable`` (bool), used by benchmark.json

    Returns True/False when a signal is present, else None (caller's default).
    """
    if not isinstance(item, dict):
        return None
    nested = item.get("cot_reliability")
    value = (
        nested.get("reliable")
        if isinstance(nested, dict) and "reliable" in nested
        else item.get("cot_reliable")
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "reliable", "yes", "1"}:
            return True
        if text in {"false", "unreliable", "no", "0"}:
            return False
    return None


def _benchmark_cot_is_unreliable(benchmark_item: dict | None) -> bool:
    """Return True when a benchmark entry explicitly marks its CoT unreliable."""
    return _cot_reliability_flag(benchmark_item) is False


def _cot_reliability_justification(item: dict) -> object:
    """Reliability justification from either schema (flat or nested)."""
    nested = item.get("cot_reliability")
    if isinstance(nested, dict) and nested.get("justification") is not None:
        return nested.get("justification")
    return item.get("cot_reliability_justification")


def _cot_unreliability_taxonomy(item: dict) -> object:
    """Unreliability taxonomy/categories from either schema (flat or nested)."""
    nested = item.get("cot_reliability")
    if isinstance(nested, dict):
        categories = nested.get("unreliability_categories")
        if categories:
            return categories
        primary = nested.get("primary_unreliability_category")
        if primary is not None:
            return primary
    return item.get("cot_unreliability_taxonomy")


def _skip_record_for_unreliable_cot(entry: dict) -> dict:
    """Build a compact audit record for an unreliable benchmark CoT skip."""
    benchmark_item = entry.get("benchmark")
    if not isinstance(benchmark_item, dict):
        benchmark_item = {}

    return {
        "clip_id": entry.get("clip_id"),
        "reason": "benchmark CoT marked unreliable",
        "cot_reliability_justification": _cot_reliability_justification(benchmark_item),
        "cot_unreliability_taxonomy": _cot_unreliability_taxonomy(benchmark_item),
        "source_report": entry.get("source_report"),
    }


def filter_unreliable_benchmark_cots(
    entries: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Drop benchmark entries with explicitly unreliable CoT annotations."""
    kept_entries = []
    skipped_entries = []
    for entry in entries:
        if _benchmark_cot_is_unreliable(entry.get("benchmark")):
            skipped_entries.append(_skip_record_for_unreliable_cot(entry))
        else:
            kept_entries.append(entry)
    return kept_entries, skipped_entries


def process_entry(
    client,
    model_name: str,
    entry: dict,
    prompt_name: str = "default",
    trajectory_frame: str = "ego_rig",
    lane_reference: str = "auto",
    use_images: bool = True,
    seed: int = DEFAULT_SEED,
    temperature: float | None = 0,
    extra_params: dict | None = None,
) -> dict:
    """Process a single extracted entry through the analysis pipeline."""
    clip_id = entry["clip_id"]
    cot_text = flatten_cot(entry["chain_of_thought"])

    cached_features = entry.get("trajectory_features")
    cached_stats = (
        cached_features.get("summary_stats", {})
        if isinstance(cached_features, dict)
        else {}
    )
    cached_frame = cached_stats.get("reference_frame")
    cached_lane_reference = cached_stats.get("lane_reference")
    cached_lane_reference_matches = cached_lane_reference == lane_reference or (
        lane_reference == "auto"
        and cached_lane_reference in {"auto", "map_graph", "route"}
    )
    use_cached_features = cached_features is not None and (
        trajectory_frame == "ego_rig"
        or (cached_frame == trajectory_frame and cached_lane_reference_matches)
    )

    if use_cached_features:
        traj_features = dict(cached_features)
    else:
        # Compute trajectory features
        traj_features = compute_trajectory_features(
            entry["trajectory_xy"],
            route_xy=entry.get("route_xy"),
            trajectory_world_xy=entry.get("trajectory_map_xy"),
            lane_center_lines=entry.get("lane_center_lines"),
            reference_frame=trajectory_frame,
            lane_reference=lane_reference,
        )

    if prompt_name == "hybrid_ego":
        traj_features = _ego_only_trajectory_features(traj_features)

    if prompt_name in {"hybrid", "hybrid_ego"}:
        traj_features = dict(traj_features)
        traj_features["trajectory_action_hints"] = _trajectory_action_hints_for_prompt(
            traj_features,
            entry,
        )

    # Build prompt
    prompt_builder = resolve_prompt_builder(prompt_name)
    prompt = prompt_builder(cot_text, traj_features)

    # Call the model
    llm_timing = None
    if client is not None:
        image = entry.get("image") if use_images else None
        raw_response, llm_timing = call_llm(
            client,
            model_name,
            prompt,
            image=image,
            seed=seed,
            temperature=temperature,
            extra_params=extra_params,
            return_timing=True,
        )
        parsed = parse_response(raw_response)
    else:
        parsed = {"dry_run": True}

    result = {
        "clip_id": clip_id,
        "chain_of_thought": cot_text,
        "trajectory_features": traj_features,
        "evaluation": parsed,
        "timestamp_us": entry.get("timestamp_us"),
        "prompt": prompt_name,
        "trajectory_frame": trajectory_frame,
        "lane_reference": lane_reference,
        "benchmark": entry.get("benchmark"),
        "source_report": entry.get("source_report"),
    }
    if llm_timing is not None:
        result["llm_timing"] = llm_timing
    return result


def dim_is_inconsistent(dim_data) -> bool | None:
    """Whether a dimension result indicates inconsistency, for either schema.

    Supports the graded schema ({"score": 1-5}, inconsistent when score <= 2)
    and the detector schema ({"verdict": "consistent" | "inconsistent"}).
    Returns None when the result carries neither signal.
    """
    if not isinstance(dim_data, dict):
        return None
    score_inconsistent = None
    if "score" in dim_data:
        try:
            score_inconsistent = float(dim_data["score"]) <= 2
        except (ValueError, TypeError):
            score_inconsistent = None

    verdict = dim_data.get("verdict")
    verdict_inconsistent = None
    if isinstance(verdict, str) and verdict.strip():
        verdict_inconsistent = verdict.strip().lower() == "inconsistent"
    if score_inconsistent is not None and verdict_inconsistent is not None:
        return score_inconsistent or verdict_inconsistent
    if verdict_inconsistent is not None:
        return verdict_inconsistent
    return score_inconsistent


def aggregate_results(results: list) -> dict:
    """Compute summary statistics across all evaluated entries."""
    scores = {d: [] for d in DIMENSIONS}
    verdicts = {d: [] for d in DIMENSIONS}
    for r in results:
        ev = r.get("evaluation", {})
        if ev.get("dry_run") or ev.get("parse_error") or ev.get("error"):
            continue
        for d in DIMENSIONS:
            dim_data = ev.get(d, {})
            if isinstance(dim_data, dict) and "score" in dim_data:
                try:
                    scores[d].append(float(dim_data["score"]))
                except (ValueError, TypeError):
                    pass
            inconsistent = dim_is_inconsistent(dim_data)
            if inconsistent is not None:
                verdicts[d].append(inconsistent)

    summary = {}
    for d in DIMENSIONS:
        s = scores[d]
        if s:
            summary[d] = {
                "mean": round(np.mean(s), 2),
                "std": round(np.std(s), 2),
                "min": round(min(s), 1),
                "max": round(max(s), 1),
                "count": len(s),
            }
        else:
            summary[d] = {"count": len(verdicts[d])}
        if verdicts[d]:
            summary[d]["inconsistent"] = sum(verdicts[d])
            summary[d]["consistent"] = len(verdicts[d]) - sum(verdicts[d])

    # Flag entries judged inconsistent (score <= 2 or verdict == inconsistent)
    flagged = []
    for r in results:
        ev = r.get("evaluation", {})
        for d in DIMENSIONS:
            dim_data = ev.get(d, {})
            if dim_is_inconsistent(dim_data):
                flagged.append(
                    {
                        "clip_id": r["clip_id"],
                        "dimension": d,
                        "score": dim_data.get("score"),
                        "verdict": dim_data.get("verdict"),
                        "inconsistency_type": dim_data.get("inconsistency_type"),
                        "justification": dim_data.get("justification", ""),
                    }
                )

    summary["flagged_entries"] = flagged
    summary["total_evaluated"] = sum(
        1
        for r in results
        if not r.get("error")
        and not r.get("evaluation", {}).get("dry_run")
        and not r.get("evaluation", {}).get("error")
    )
    return summary


def build_output_data(
    results: list,
    summary: dict,
    *,
    entries_to_analyze: int,
    original_entry_count: int | None = None,
    skipped_unreliable_cot_entries: list[dict] | None = None,
) -> dict:
    """Build the persisted report payload with run-level filter metadata."""
    skipped_unreliable_cot_entries = skipped_unreliable_cot_entries or []
    summary = dict(summary)
    summary["input_entries"] = (
        original_entry_count if original_entry_count is not None else entries_to_analyze
    )
    summary["entries_to_analyze"] = entries_to_analyze
    if skipped_unreliable_cot_entries:
        summary["skipped_unreliable_cot"] = len(skipped_unreliable_cot_entries)

    payload = {
        "results": results,
        "summary": summary,
    }
    if skipped_unreliable_cot_entries:
        payload["skipped_entries"] = {
            "unreliable_cot": skipped_unreliable_cot_entries,
        }
    return payload


# =============================================================================
# Main
# =============================================================================


def main():
    _find_and_load_dotenv()

    parser = argparse.ArgumentParser(
        description="CoT consistency analysis for AlpaSim simulation outputs"
    )
    parser.add_argument(
        "--asl_glob",
        type=str,
        default=None,
        help="Glob pattern to find ASL files (e.g. 'rollouts/**/*.asl')",
    )
    parser.add_argument(
        "--benchmark_json",
        type=str,
        default=None,
        help=(
            "Benchmark JSON with clip_id/source_scene_file entries. Reuses "
            "trajectory_features from the referenced cot_consistency reports."
        ),
    )
    parser.add_argument(
        "--benchmark_source_root",
        type=str,
        default=".",
        help=(
            "Root for resolving benchmark source_scene_file paths "
            "(default: current working directory)."
        ),
    )
    parser.add_argument(
        "--skip_unreliable_cot",
        "--skip-unreliable-cot",
        action="store_true",
        help=(
            "Benchmark mode only: skip entries whose benchmark annotation marks "
            "the CoT unreliable (flat cot_reliable=false or nested "
            "cot_reliability.reliable=false)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cot_consistency.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=sorted(PROVIDERS),
        default="gateway",
        help=(
            "Model backend. Determines the default model, API key env var, "
            "base URL, and provider-specific request parameters."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (default: the selected provider's model)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (defaults to the provider's key env var from .env/environment)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL override (defaults to the provider's base URL)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Fixed decoding seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (the gateway allows ~60/min)",
    )
    parser.add_argument(
        "--every_nth",
        type=int,
        default=1,
        help="Only analyze every Nth timestep (default: 1 = all)",
    )
    parser.add_argument(
        "--camera_id",
        type=str,
        default="camera_front_wide_120fov",
        help="Camera logical ID for image context",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="default",
        help=(
            "Prompt template: a registered name (%s), or a path to a .py file "
            "defining build_prompt(cot_text, traj_features). New prompt_<name>.py "
            "files in this package are auto-discovered."
            % ", ".join(discover_prompt_names())
        ),
    )
    parser.add_argument(
        "--trajectory_frame",
        type=str,
        choices=["ego_rig", "lane_center", "dual"],
        default="ego_rig",
        help=(
            "Feature frame for the predicted trajectory. lane_center projects "
            "onto map geometry or logged route_request waypoints and falls back "
            "to ego_rig if no lane reference is available. dual computes both "
            "ego-frame and lane-center features (pairs with --prompt "
            "center_of_lane_v3 or hybrid; lane_reference auto means map_graph "
            "in dual). --prompt hybrid_ego forces ego_rig."
        ),
    )
    parser.add_argument(
        "--lane_reference",
        type=str,
        choices=["auto", "map_graph", "map_graph_same_lane", "route"],
        default="auto",
        help=(
            "Lane-center source when --trajectory_frame lane_center. map_graph "
            "walks the lane successor graph to build one route-consistent "
            "reference path (no per-point nearest-lane switching; pairs with "
            "--prompt center_of_lane_v2). map_graph_same_lane evaluates the "
            "same candidate starting lanes without traversing to predecessor "
            "or successor lanes. auto prefers map_graph when "
            "--lane_geometry_root is provided, then falls back to logged route "
            "waypoints."
        ),
    )
    parser.add_argument(
        "--lane_geometry_root",
        type=str,
        default=None,
        help=(
            "Optional extracted-frames root containing trajectory_plot_geometry.json "
            "files. This enables true map lane-center features."
        ),
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip sending images to the model (text-only mode)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed entries",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()
    if bool(args.asl_glob) == bool(args.benchmark_json):
        parser.error("Specify exactly one of --asl_glob or --benchmark_json")
    if args.skip_unreliable_cot and not args.benchmark_json:
        parser.error("--skip_unreliable_cot is only supported with --benchmark_json")
    # Resolve (and cache) the prompt builder up front so a bad --prompt fails
    # immediately instead of erroring on every entry.
    try:
        resolve_prompt_builder(args.prompt)
    except (ValueError, FileNotFoundError, AttributeError) as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.prompt == "hybrid" and args.trajectory_frame != "dual":
        logger.info(
            "--prompt hybrid uses dual-frame trajectory features; overriding "
            "--trajectory_frame %s -> dual.",
            args.trajectory_frame,
        )
        args.trajectory_frame = "dual"
    if args.prompt == "hybrid_ego" and args.trajectory_frame != "ego_rig":
        logger.info(
            "--prompt hybrid_ego uses ego-frame trajectory features only; overriding "
            "--trajectory_frame %s -> ego_rig.",
            args.trajectory_frame,
        )
        args.trajectory_frame = "ego_rig"

    # --- Setup model client ---
    provider = PROVIDERS[args.provider]
    api_key_env = provider["api_key_env"]
    api_key = (
        args.api_key
        or os.environ.get(api_key_env)
        or provider.get("default_api_key")
    )
    base_url = (
        args.base_url
        or os.environ.get(provider["base_url_env"])
        or provider["base_url"]
    )
    model_name = args.model or provider["model"]
    extra_params = provider.get("extra_params") or {}
    temperature = provider.get("temperature", 0)
    client = None

    if api_key:
        client = _create_client(api_key, base_url)
        logger.info(
            "Using %s — model: %s (base_url=%s, seed=%d, temperature=%s)",
            provider["label"],
            model_name,
            base_url or "<openai-default>",
            args.seed,
            "default" if temperature is None else temperature,
        )
        if extra_params:
            logger.info("Extra request params: %s", extra_params)
    else:
        logger.info(
            "No API key (set $%s or --api_key). Running in DRY-RUN mode "
            "(trajectory analysis only).",
            api_key_env,
        )
    logger.info(
        "Using prompt=%s, trajectory_frame=%s, lane_reference=%s",
        args.prompt,
        args.trajectory_frame,
        args.lane_reference,
    )
    if args.prompt == "default" and args.trajectory_frame == "lane_center":
        logger.warning(
            "lane_center features are intended for --prompt center_of_lane; "
            "the default prompt describes ego-frame x/y coordinates."
        )
    if args.prompt == "center_of_lane_v3" and args.trajectory_frame != "dual":
        logger.warning(
            "--prompt center_of_lane_v3 is designed for --trajectory_frame dual; "
            "with %s it will render only a single view.",
            args.trajectory_frame,
        )

    # --- Extract entries ---
    skipped_unreliable_cot_entries = []
    original_entry_count = None
    if args.benchmark_json:
        all_entries = extract_entries_from_benchmark(
            args.benchmark_json,
            source_root=args.benchmark_source_root,
            load_raw_trajectory=args.trajectory_frame in {"lane_center", "dual"},
        )
        original_entry_count = len(all_entries)
        if args.skip_unreliable_cot:
            all_entries, skipped_unreliable_cot_entries = (
                filter_unreliable_benchmark_cots(all_entries)
            )
            logger.info(
                "Skipped %d unreliable benchmark CoT entr%s",
                len(skipped_unreliable_cot_entries),
                "y" if len(skipped_unreliable_cot_entries) == 1 else "ies",
            )
    else:
        lane_geometry_index = _load_lane_geometry_index(args.lane_geometry_root)
        if (
            args.lane_reference in {"map_graph", "map_graph_same_lane"}
            and not lane_geometry_index
        ):
            logger.warning(
                "--lane_reference %s requested without --lane_geometry_root; "
                "lane-center features will fall back to ego-frame with an error note.",
                args.lane_reference,
            )

        # --- Find ASL files ---
        asl_files = sorted(glob.glob(args.asl_glob, recursive=True))
        if not asl_files:
            logger.error("No ASL files found matching: %s", args.asl_glob)
            return

        logger.info("Found %d ASL file(s)", len(asl_files))

        # --- Extract entries from all ASL files ---
        all_entries = []
        for asl_path in asl_files:
            logger.info("Parsing: %s", asl_path)
            entries = asyncio.run(
                extract_entries_from_asl(
                    asl_path,
                    camera_id=args.camera_id,
                    every_nth=args.every_nth,
                    lane_geometry_index=lane_geometry_index,
                )
            )
            all_entries.extend(entries)

    logger.info("Total entries to analyze: %d", len(all_entries))

    completed_ids = set()
    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not all_entries:
        logger.warning("No entries extracted. Check ASL files and settings.")
        summary = aggregate_results(results)
        output_data = build_output_data(
            results,
            summary,
            entries_to_analyze=len(all_entries),
            original_entry_count=original_entry_count,
            skipped_unreliable_cot_entries=skipped_unreliable_cot_entries,
        )
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return

    # --- Resume support ---
    if args.resume and output_path.exists():
        try:
            with open(output_path, "r") as f:
                saved = json.load(f)
                loaded_results = (
                    saved.get("results", saved) if isinstance(saved, dict) else saved
                )
            if isinstance(loaded_results, dict):
                loaded_results = [loaded_results]
            elif not isinstance(loaded_results, list):
                loaded_results = []
            allowed_ids = {entry["clip_id"] for entry in all_entries}
            current_results = [
                r
                for r in loaded_results
                if isinstance(r, dict) and r.get("clip_id") in allowed_ids
            ]
            results = [
                r
                for r in current_results
                if not r.get("error")
                and not r.get("evaluation", {}).get("error")
            ]
            retry_results = len(current_results) - len(results)
            dropped_results = len(loaded_results) - len(current_results)
            completed_ids = {r["clip_id"] for r in results}
            logger.info("Resuming: %d entries already completed", len(completed_ids))
            if retry_results:
                logger.info(
                    "Retrying %d saved result(s) with prior errors",
                    retry_results,
                )
            if dropped_results:
                logger.info(
                    "Ignored %d saved result(s) outside the current input set",
                    dropped_results,
                )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Output file invalid. Starting fresh.")

    # --- Process entries ---
    use_images = not args.no_images
    if use_images and not provider.get("supports_images", True):
        logger.info(
            "%s is text-only for this workflow; disabling image inputs.",
            provider["label"],
        )
        use_images = False
    for idx, entry in enumerate(all_entries):
        clip_id = entry["clip_id"]

        if clip_id in completed_ids:
            continue

        if entry.get("error"):
            logger.error(
                "[%d/%d] %s — %s", idx + 1, len(all_entries), clip_id, entry["error"]
            )
            results.append(
                {
                    "clip_id": clip_id,
                    "error": entry["error"],
                    "benchmark": entry.get("benchmark"),
                    "source_report": entry.get("source_report"),
                }
            )
            continue

        cot_text = flatten_cot(entry["chain_of_thought"])
        logger.info(
            "[%d/%d] %s — CoT: %s",
            idx + 1,
            len(all_entries),
            clip_id,
            cot_text[:80],
        )

        try:
            result = process_entry(
                client,
                model_name,
                entry,
                prompt_name=args.prompt,
                trajectory_frame=args.trajectory_frame,
                lane_reference=args.lane_reference,
                use_images=use_images,
                seed=args.seed,
                temperature=temperature,
                extra_params=extra_params,
            )
            results.append(result)

            # Print summary
            stats = result["trajectory_features"].get("summary_stats", {})
            ev = result["evaluation"]

            if stats.get("reference_frame") == "dual":
                lane_stats = stats.get("lane", {})
                ego_stats = stats.get("ego", {})
                if lane_stats.get("reference_frame") == "lane_center":
                    logger.info(
                        "  [dual] Lane progress: %.1fm | Offset: %.1fm -> %.1fm | "
                        "Reference: %s | Final pos: (%.1fm fwd, %.1fm lat) | Speed: %.1f m/s avg",
                        lane_stats.get("lane_path_length_m", 0),
                        lane_stats.get("initial_offset_m", 0),
                        lane_stats.get("final_offset_m", 0),
                        lane_stats.get("lane_reference", "unknown"),
                        ego_stats.get("final_longitudinal_m", 0),
                        ego_stats.get("final_lateral_m", 0),
                        ego_stats.get("mean_speed_ms", 0),
                    )
                else:
                    logger.warning(
                        "  [dual] Lane-center view unavailable: %s",
                        stats.get("lane_center_error", "unknown"),
                    )
                    logger.info(
                        "  [dual] Final pos: (%.1fm fwd, %.1fm lat) | Speed: %.1f m/s avg",
                        ego_stats.get("final_longitudinal_m", 0),
                        ego_stats.get("final_lateral_m", 0),
                        ego_stats.get("mean_speed_ms", 0),
                    )
            elif stats.get("reference_frame") == "lane_center":
                logger.info(
                    "  Lane progress: %.1fm | Offset: %.1fm -> %.1fm | Reference: %s | Speed: %.1f m/s avg",
                    stats.get("lane_path_length_m", 0),
                    stats.get("initial_offset_m", 0),
                    stats.get("final_offset_m", 0),
                    stats.get("lane_reference", "unknown"),
                    stats.get("mean_speed_ms", 0),
                )
            else:
                if stats.get("lane_center_error"):
                    logger.warning(
                        "  Lane-center features unavailable: %s",
                        stats["lane_center_error"],
                    )
                logger.info(
                    "  Final pos: (%.1fm fwd, %.1fm lat) | Speed: %.1f m/s avg",
                    stats.get("final_longitudinal_m", 0),
                    stats.get("final_lateral_m", 0),
                    stats.get("mean_speed_ms", 0),
                )

            if not ev.get("dry_run"):
                for dim in DIMENSIONS:
                    d = ev.get(dim, {})
                    if isinstance(d, dict) and "score" in d:
                        logger.info(
                            "  %s: %s/5 — %s",
                            dim,
                            d["score"],
                            d.get("justification", "")[:60],
                        )
                    elif isinstance(d, dict) and d.get("verdict"):
                        label = d["verdict"]
                        if d.get("inconsistency_type"):
                            label += f" ({d['inconsistency_type']})"
                        logger.info(
                            "  %s: %s — %s",
                            dim,
                            label,
                            d.get("justification", "")[:60],
                        )

        except Exception as e:
            logger.error("  ERROR: %s", e)
            results.append({"clip_id": clip_id, "error": str(e)})

        # Incremental save
        summary = aggregate_results(results)
        output_data = build_output_data(
            results,
            summary,
            entries_to_analyze=len(all_entries),
            original_entry_count=original_entry_count,
            skipped_unreliable_cot_entries=skipped_unreliable_cot_entries,
        )
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        if args.delay > 0 and client is not None:
            time.sleep(args.delay)

    # --- Final summary ---
    summary = aggregate_results(results)
    output_data = build_output_data(
        results,
        summary,
        entries_to_analyze=len(all_entries),
        original_entry_count=original_entry_count,
        skipped_unreliable_cot_entries=skipped_unreliable_cot_entries,
    )
    summary = output_data["summary"]
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'=' * 50}")
    print(f"Results saved to: {output_path}")
    if skipped_unreliable_cot_entries:
        print(
            f"Skipped unreliable benchmark CoTs: {len(skipped_unreliable_cot_entries)}"
        )
    print(f"Total evaluated: {summary.get('total_evaluated', 0)}/{len(all_entries)}")
    for dim in DIMENSIONS:
        d = summary.get(dim, {})
        if "mean" in d:
            print(f"  {dim}: {d['mean']:.2f} ± {d['std']:.2f} (n={d['count']})")
        elif d.get("count", 0) > 0:
            print(
                f"  {dim}: {d.get('inconsistent', 0)} inconsistent / "
                f"{d.get('consistent', 0)} consistent (n={d['count']})"
            )
    flagged = summary.get("flagged_entries", [])
    if flagged:
        print(f"\n⚠ {len(flagged)} inconsistency flags:")
        for f_item in flagged[:10]:
            if f_item.get("score") is not None:
                label = f"{f_item['score']}/5"
            else:
                label = f_item.get("verdict", "inconsistent")
                if f_item.get("inconsistency_type"):
                    label += f" ({f_item['inconsistency_type']})"
            print(f"  {f_item['clip_id']} {f_item['dimension']}: {label}")


if __name__ == "__main__":
    main()
