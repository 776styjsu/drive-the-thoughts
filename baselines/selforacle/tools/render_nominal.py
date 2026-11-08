"""Generate GT camera streams on Slurm; keep proposed nominal status pending review.

Stage one pinned USDZ at a time. Persistent outputs contain raw camera JPEGs,
timestamps, source/config hashes, GT poses, render logs, and GPU measurements.
Intermediate USDZs, overlays, and ASLs are this job's temporary files only.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
CAMERA = "camera_front_wide_120fov"
HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
SIF_CACHE = os.environ.get("SIF_CACHE", str(ROOT / "alpasim" / "sif-cache"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def shard(rows, task_index, scenes_per_task, limit):
    if task_index < 0 or scenes_per_task < 1 or limit < 1:
        raise ValueError("Invalid shard bounds")
    start = task_index * scenes_per_task
    return rows[:limit][start:start + scenes_per_task]


def wizard_environment():
    # This worker already shards the cohort and launches one scene at a time.
    # Hide the outer array from the wizard so it cannot shard that scene again.
    return {key: value for key, value in os.environ.items()
            if not key.startswith("SLURM_ARRAY_")}


def lease_ports():
    """Hold a per-node file lock for a private range across container launches."""
    # Stay below Linux's ephemeral client-port range (32768+ on these nodes).
    # Otherwise downloads/gRPC clients can claim a checked-but-not-yet-bound
    # telemetry port while the renderer starts, including via TIME_WAIT.
    blocks = 100
    start = int(os.environ.get("SLURM_JOB_ID", "0")) % blocks
    for offset in range(blocks):
        base = 20000 + ((start + offset) % blocks) * 100
        stream = open(f"/tmp/selforacle-ports-{os.getuid()}-{base}.lock", "a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sockets = []
            try:
                for port in range(base, base + 16):
                    sock = socket.socket()
                    sockets.append(sock)
                    sock.bind(("0.0.0.0", port))
            finally:
                for sock in sockets:
                    sock.close()
            return base, stream
        except (BlockingIOError, OSError):
            stream.close()
    raise RuntimeError("No private service-port range available")


def gpu_info():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This worker requires exactly one Slurm-visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability(0)[0] != 8:
        raise RuntimeError("This campaign uses supported Ampere GPUs only")
    if properties.total_memory < 38 * 1024**3:
        raise RuntimeError("Provision at least a 40 GB GPU for NuRec")
    uuid = str(properties.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    result = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], text=True
    ).strip().split(",")
    return {"uuid": uuid, "name": result[0].strip(), "total_mib": int(result[1]),
            "driver": result[2].strip(), "slurm_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


class GpuMonitor:
    def __init__(self, uuid, path):
        self.uuid, self.path = uuid, path
        self.stop = threading.Event()
        self.peak_mib = 0
        self.samples = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["unix_time", "memory_used_mib", "utilization_percent"])
            while not self.stop.is_set():
                try:
                    result = subprocess.check_output(
                        ["nvidia-smi", "-i", self.uuid, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10
                    ).strip().split(",")
                    used, utilization = map(int, result)
                    self.peak_mib = max(self.peak_mib, used)
                    self.samples += 1
                    writer.writerow([time.time(), used, utilization])
                    stream.flush()
                except (subprocess.SubprocessError, ValueError):
                    pass
                self.stop.wait(1)


def camera_metadata(timestamp, scene_id, index, payload):
    return {"clip_id": f"{scene_id}_t{index:04d}", "time_now_us": timestamp,
            "chain_of_thought": None, "trajectory_poses": [],
            "image_sha256": hashlib.sha256(payload).hexdigest(), "source": "alpasim_full_gt_replay"}


async def extract_frames(asl, destination, expected_scene):
    from alpasim_utils.logs import async_read_pb_log
    from PIL import Image

    destination.mkdir(parents=True, exist_ok=True)
    timestamps = []
    source_scene = None
    dimensions = None
    async for message in async_read_pb_log(str(asl)):
        kind = message.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            source_scene = message.rollout_metadata.session_metadata.scene_id
            if source_scene != expected_scene:
                raise ValueError("ASL scene differs from the pinned source clip")
        elif kind == "driver_request":
            raise ValueError("GT-only replay unexpectedly called the driving policy")
        elif kind == "driver_camera_image":
            image = message.driver_camera_image.camera_image
            if image.logical_id != CAMERA:
                continue
            timestamp = int(image.frame_end_us)
            if timestamps and timestamp - timestamps[-1] != 100000:
                raise ValueError("Camera frames must be unique consecutive 10 Hz samples")
            payload = image.image_bytes
            with Image.open(io.BytesIO(payload)) as decoded:
                if decoded.height != 1080 or not 1880 <= decoded.width <= 1930:
                    raise ValueError(f"Camera dimensions differ from evaluation: {decoded.size}")
                if dimensions is None:
                    dimensions = decoded.size
                    decoded.load()
            index = len(timestamps)
            metadata = camera_metadata(timestamp, expected_scene, index, payload)
            folder = destination / metadata["clip_id"]
            folder.mkdir(exist_ok=True)
            (folder / (CAMERA + ".jpg")).write_bytes(payload)
            write_json(folder / "metadata.json", metadata)
            timestamps.append(timestamp)
    if source_scene != expected_scene or len(timestamps) < 140:
        raise ValueError(f"Incomplete GT stream: {len(timestamps)} frames")
    return {"frames": len(timestamps), "first_timestamp_us": timestamps[0],
            "last_timestamp_us": timestamps[-1], "cadence_us": 100000, "dimensions_wh": dimensions,
            "driver_calls": 0}


def render_scene(row, args, job_scratch, baseport, gpu):
    import yaml
    from dotenv import dotenv_values
    from huggingface_hub import get_token, hf_hub_download

    scene = row["scene_id"]
    destination = args.campaign / "scenes" / scene
    destination.mkdir(parents=True, exist_ok=True)
    complete = destination / "render_complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["artifact_uuid"] != row["uuid"] or previous["hf_revision"] != row["hf_revision"]:
            raise ValueError("Completed stream belongs to another reconstruction")
        print(f"Already complete: {scene}", flush=True)
        return
    if shutil.disk_usage(args.campaign).free < 60 * 1024**3:
        raise RuntimeError("Project filesystem has less than 60 GiB free; stop before filling it")
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="scene-", dir=job_scratch) as temporary:
        stage = Path(temporary)
        local_assets = stage / "assets"
        local_assets.mkdir()
        settings = dotenv_values(ROOT / ".env")
        token = get_token() or settings.get("HF_TOKEN") or settings.get("HUGGING_FACE_HUB_TOKEN")
        print(f"Downloading pinned USDZ: {scene}", flush=True)
        downloaded = Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=row["path"], revision=row["hf_revision"], local_dir=local_assets, token=token))
        with zipfile.ZipFile(downloaded) as archive:
            metadata = yaml.safe_load(archive.read("metadata.yaml"))
            if metadata["uuid"] != row["uuid"] or metadata["scene_id"] != scene:
                raise ValueError("Downloaded USDZ identity differs from pinned cohort")
            for filename in ["metadata.yaml", "rig_trajectories.json", "datasource_summary.json"]:
                if filename in archive.namelist():
                    (destination / filename).write_bytes(archive.read(filename))
        asset_sha = digest(downloaded)
        run_dir = stage / "run"
        overrides = [
            "deploy=local_apptainer", "topology=1gpu", "driver=alpamayo1_5",
            f"wizard.sif_caches=[{SIF_CACHE}]", f"wizard.log_dir={run_dir}",
            f"wizard.baseport={baseport}", "wizard.run_sim_services=[renderer,runtime]",
            "wizard.prometheus.start_prometheus=false", "wizard.timeout=600",
            "services.runtime.image=alpasim-base:0.111.0",
            f"scenes.local_usdz_dir={local_assets}", f"scenes.scene_cache={local_assets}",
            f"scenes.scene_ids=[{scene}]", "scenes.test_suite_id=null",
            "defines.nre_cache_size=1", "defines.nre_max_workers=1", "runtime.nr_workers=1",
            "runtime.endpoints.renderer.n_concurrent_rollouts=1",
            "+runtime.endpoints.driver.skip=true", "runtime.endpoints.physics.skip=true",
            "runtime.endpoints.trafficsim.skip=true", "runtime.endpoints.controller.skip=true",
            "runtime.simulation_config.physics_update_mode=NONE",
            "runtime.simulation_config.force_gt_duration_us=30000000",
            "runtime.simulation_config.skip_driver_during_force_gt=true",
            "runtime.simulation_config.n_sim_steps=200", "runtime.simulation_config.n_rollouts=1",
            "runtime.simulation_config.route_generator_type=NONE",
            "runtime.simulation_config.cameras=[{logical_id:camera_front_wide_120fov,height:1080,width:1900,frame_interval_us:100000,shutter_duration_us:30000}]",
            "eval.enabled=false",
        ]
        write_json(destination / "render_request.json", {"source": row, "overrides": overrides, "gpu": gpu,
            "nominal_status": "needs_visual_and_render_review", "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        monitor = GpuMonitor(gpu["uuid"], destination / "gpu_usage.csv")
        monitor.thread.start()
        try:
            with (destination / "render.log").open("a") as log:
                result = subprocess.run([sys.executable, "-m", "alpasim_wizard", *overrides], cwd=ROOT / "alpasim", env=wizard_environment(), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if result.returncode:
                raise RuntimeError(f"AlpaSim failed: exit {result.returncode}; see {destination / 'render.log'}")
            rollouts = list(run_dir.glob(f"rollouts/{scene}/*/rollout.asl"))
            if len(rollouts) != 1 or not rollouts[0].with_name("_complete").is_file():
                raise ValueError("Expected one completed GT rollout")
            stream = asyncio.run(extract_frames(rollouts[0], destination / "extracted_frames", scene))
            asl_sha = digest(rollouts[0])
        finally:
            monitor.stop.set()
            monitor.thread.join(timeout=15)
            for pattern in ["*.yaml", "*.json"]:
                for path in run_dir.glob(pattern):
                    shutil.copy2(path, destination / path.name)
            if (run_dir / "txt-logs").exists():
                shutil.copytree(run_dir / "txt-logs", destination / "txt-logs", dirs_exist_ok=True)
        if monitor.samples == 0:
            raise ValueError("No GPU telemetry recorded")
        write_json(complete, {**stream, "scene_id": scene, "artifact_uuid": row["uuid"],
            "hf_revision": row["hf_revision"], "source_usdz_sha256": asset_sha, "source_asl_sha256": asl_sha,
            "intermediate_retention": "job-created USDZ, overlays, and ASL removed after extraction; source hashes and GT poses retained",
            "proposed_split": row["proposed_split"], "nominal_status": "needs_visual_and_render_review",
            "gpu": gpu, "peak_gpu_memory_mib": monitor.peak_mib, "gpu_samples": monitor.samples,
            "elapsed_seconds": time.time() - started})
        print(f"Completed {scene}: {stream['frames']} frames; peak GPU {monitor.peak_mib} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--scenes-per-task", type=int, default=20)
    parser.add_argument("--limit", type=int, default=900)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.limit > 2:
        gate = json.loads((args.campaign / "pilot_validated.json").read_text())
        if gate["status"] != "passed" or gate["cohort_sha256"] != digest(args.campaign / "cohort.csv"):
            raise ValueError("Full rendering requires a passed pilot for this exact cohort")
    with (args.campaign / "cohort.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = shard(rows, args.task_index, args.scenes_per_task, args.limit)
    if not selected:
        return
    gpu = gpu_info()
    print(json.dumps({"gpu": gpu, "assigned_scenes": len(selected)}), flush=True)
    baseport, port_lock = lease_ports()
    scratch_parent = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch_parent).free < 12 * 1024**3:
        raise RuntimeError("Need at least 12 GiB temporary space for a streaming worker")
    failures = []
    try:
        with tempfile.TemporaryDirectory(prefix=f"selforacle-{os.environ.get('SLURM_JOB_ID','manual')}-", dir=scratch_parent) as temporary:
            for row in selected:
                try:
                    render_scene(row, args, Path(temporary), baseport, gpu)
                except Exception as error:
                    failures.append({"scene_id": row["scene_id"], "error": str(error)})
                    print(f"Failed {row['scene_id']}: {type(error).__name__}: {error}", flush=True)
    finally:
        port_lock.close()
    status = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "task_index": args.task_index,
              "assigned_scenes": len(selected), "failures": failures, "gpu": gpu}
    write_json(args.campaign / "tasks" / f"{os.environ.get('SLURM_JOB_ID','manual')}_{args.task_index}.json", status)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
