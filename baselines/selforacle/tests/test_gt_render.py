"""GT-only streams must work without driving-policy request/return messages."""
import asyncio
import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("gt_render", Path(__file__).parents[1] / "tools/render_nominal.py")
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


def test_shards_cover_cohort_once():
    rows = list(range(900))
    assigned = [render.shard(rows, index, 20, 900) for index in range(45)]
    assert [row for group in assigned for row in group] == rows
    assert render.shard(rows, 45, 20, 900) == []


def test_worker_does_not_shard_a_scene_twice(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "44")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "45")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    child = render.wizard_environment()
    assert not any(key.startswith("SLURM_ARRAY_") for key in child)
    assert child["SLURM_JOB_ID"] == "123"
    assert child["CUDA_VISIBLE_DEVICES"] == "3"


def test_gt_stream_extraction_without_driver_calls(tmp_path, monkeypatch):
    from alpasim_grpc.v0.logging_pb2 import LogEntry
    import alpasim_utils.logs

    scene = "clipgt-example"
    buffer = io.BytesIO()
    Image.new("RGB", (1916, 1080), (60, 80, 100)).save(buffer, format="JPEG")
    payload = buffer.getvalue()

    async def read(_):
        metadata = LogEntry()
        metadata.rollout_metadata.session_metadata.scene_id = scene
        yield metadata
        for index in range(140):
            entry = LogEntry()
            camera = entry.driver_camera_image.camera_image
            camera.logical_id = render.CAMERA
            camera.frame_end_us = (index + 1) * 100000
            camera.image_bytes = payload
            yield entry

    monkeypatch.setattr(alpasim_utils.logs, "async_read_pb_log", read)
    result = asyncio.run(render.extract_frames(tmp_path / "unused.asl", tmp_path / "frames", scene))
    assert result["frames"] == 140
    assert result["driver_calls"] == 0
    assert len(list((tmp_path / "frames").glob("*/metadata.json"))) == 140


def test_separate_workers_lease_different_ports():
    base1, lock1 = render.lease_ports()
    try:
        base2, lock2 = render.lease_ports()
        try:
            assert base1 != base2
        finally:
            lock2.close()
    finally:
        lock1.close()


@pytest.mark.parametrize("job_id", [173, 153, 146, 99])
def test_service_ports_do_not_overlap_ephemeral_client_ports(monkeypatch, job_id):
    # The old allocator selected 37300, 35300 and 34600 for these first three
    # values; runtime metrics then collided with outbound client sockets.
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    base, lock = render.lease_ports()
    try:
        assert 20000 <= base <= 29900
        assert base + 15 < 32768
    finally:
        lock.close()
