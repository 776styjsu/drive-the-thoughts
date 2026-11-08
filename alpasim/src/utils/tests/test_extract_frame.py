# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import asyncio
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest
from alpasim_utils.extract_frame import __main__ as extract_frame
from alpasim_utils.extract_frame.__main__ import (
    _eval_config_candidates,
    _load_eval_config,
)
from omegaconf import MISSING, OmegaConf


@dataclass
class _RequiredEvalConfig:
    vehicle: dict = MISSING


def _rollout_path(run_dir: Path) -> Path:
    return run_dir / "rollouts" / "scene-id" / "rollout-id" / "rollout.asl"


def test_eval_config_candidates_prefer_run_eval_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    asl_path = _rollout_path(run_dir)

    candidates = _eval_config_candidates(asl_path)

    assert candidates[:3] == [
        (run_dir / "eval-config.yaml", None),
        (run_dir / "wizard-config-loadable.yaml", "eval"),
        (run_dir / "wizard-config.yaml", "eval"),
    ]


def test_eval_config_candidates_find_rollouts_ancestor(tmp_path: Path) -> None:
    run_dir = tmp_path / "nested" / "run"
    asl_path = _rollout_path(run_dir)
    asl_path.parent.mkdir(parents=True)
    asl_path.touch()

    candidates = _eval_config_candidates(asl_path)

    assert candidates[0] == (run_dir / "eval-config.yaml", None)


def test_load_eval_config_uses_run_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    asl_path = _rollout_path(run_dir)
    asl_path.parent.mkdir(parents=True)
    asl_path.touch()
    (run_dir / "eval-config.yaml").write_text(
        "vehicle:\n  vehicle_corner_roundness: 0.5\n"
    )

    config = _load_eval_config(asl_path, OmegaConf, _RequiredEvalConfig)

    assert config.vehicle.vehicle_corner_roundness == 0.5


def test_load_eval_config_reads_eval_section_from_wizard_config(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    asl_path = _rollout_path(run_dir)
    asl_path.parent.mkdir(parents=True)
    asl_path.touch()
    (run_dir / "wizard-config-loadable.yaml").write_text(
        "eval:\n  vehicle:\n    vehicle_corner_roundness: 0.25\n"
    )

    config = _load_eval_config(asl_path, OmegaConf, _RequiredEvalConfig)

    assert config.vehicle.vehicle_corner_roundness == 0.25


def _driver_messages(timestamp, endpoint_scales):
    from alpasim_grpc.v0.common_pb2 import PoseAtTime, Trajectory
    from alpasim_grpc.v0.egodriver_pb2 import DriveRequest, DriveResponse
    from alpasim_grpc.v0.logging_pb2 import LogEntry

    messages = []
    for attempt, scale in enumerate(endpoint_scales):
        response = DriveResponse(trajectory=Trajectory())
        for i in range(3):
            point = PoseAtTime(timestamp_us=timestamp + i * 100_000)
            point.pose.vec.x = i * scale
            point.pose.quat.w = 1
            response.trajectory.poses.append(point)
        response.debug_info.unstructured_debug_info = pickle.dumps(
            {"reasoning_text": f"candidate_{attempt}"}
        )
        messages.extend(
            [
                LogEntry(
                    driver_request=DriveRequest(
                        time_now_us=timestamp, time_query_us=timestamp + 50_000
                    )
                ),
                LogEntry(driver_return=response),
            ]
        )
    return messages


def _mock_log(monkeypatch, messages):
    async def read_log(path):
        for message in messages:
            yield message

    monkeypatch.setattr(extract_frame, "async_read_pb_log", read_log)


def _write_decision(tmp_path, timestamp=1_000_000, chosen=0, count=2):
    (tmp_path / "consistency_monitor.jsonl").write_text(
        json.dumps(
            {"time_now_us": timestamp, "chosen_attempt": chosen, "num_attempts": count}
        )
        + "\n"
    )


def _selected_messages(asl_path, target_time_us=None):
    async def collect():
        return [
            message
            async for message in extract_frame._selected_driver_messages(
                str(asl_path), target_time_us
            )
        ]

    return asyncio.run(collect())


def test_selection_preserves_earlier_chosen_candidate_and_next_step(
    monkeypatch, tmp_path
):
    first = _driver_messages(1_000_000, [1, 3, 5])
    second = _driver_messages(2_000_000, [2])
    _mock_log(monkeypatch, first + second)
    _write_decision(tmp_path, chosen=1, count=3)

    selected = _selected_messages(tmp_path / "rollout.asl")

    assert selected == [first[2], first[3], second[0], second[1]]


def test_legacy_single_candidate_log_needs_no_audit(monkeypatch, tmp_path):
    messages = _driver_messages(1_000_000, [1])
    _mock_log(monkeypatch, messages)
    assert _selected_messages(tmp_path / "rollout.asl") == messages


def test_multiple_candidates_without_selection_fail(monkeypatch, tmp_path):
    _mock_log(monkeypatch, _driver_messages(1_000_000, [1, 3]))
    with pytest.raises(ValueError, match="missing consistency_monitor.jsonl decision"):
        _selected_messages(tmp_path / "rollout.asl")


def test_audit_must_match_logged_candidate_count(monkeypatch, tmp_path):
    _mock_log(monkeypatch, _driver_messages(1_000_000, [1, 3]))
    _write_decision(tmp_path, count=3)
    with pytest.raises(ValueError, match="candidate count mismatch"):
        _selected_messages(tmp_path / "rollout.asl")


def test_target_frame_ignores_ambiguous_unrequested_steps(monkeypatch, tmp_path):
    first = _driver_messages(1_000_000, [1, 3])
    second = _driver_messages(2_000_000, [2])
    _mock_log(monkeypatch, first + second)
    assert _selected_messages(tmp_path / "rollout.asl", 2_050_000) == second


@pytest.mark.parametrize("chosen", [-1, 2, True])
def test_invalid_selection_cannot_silently_export_another_candidate(tmp_path, chosen):
    _write_decision(tmp_path, chosen=chosen)
    with pytest.raises(ValueError, match="Invalid monitor decision"):
        extract_frame._read_monitor_decisions(str(tmp_path / "rollout.asl"))


def test_extracted_metadata_and_plot_use_selected_candidate(monkeypatch, tmp_path):
    messages = _driver_messages(1_000_000, [1, 3])
    _mock_log(monkeypatch, messages)
    _write_decision(tmp_path, chosen=0)
    plotted = []
    monkeypatch.setattr(
        extract_frame,
        "save_trajectory_plot",
        lambda points, path: plotted.append(points),
    )
    monkeypatch.setattr("glob.glob", lambda *args, **kwargs: [])

    def no_eval_context(*args):
        raise FileNotFoundError("test rollout has no evaluation configuration")

    monkeypatch.setattr(extract_frame, "_load_eval_config", no_eval_context)
    output = tmp_path / "frames"
    asyncio.run(
        extract_frame.extract_inference(
            str(tmp_path / "rollout.asl"), str(output), write_additional_info=False
        )
    )

    files = list(output.glob("*/metadata.json"))
    assert len(files) == 1
    metadata = json.loads(files[0].read_text())
    assert metadata["chain_of_thought"] == "candidate_0"
    assert metadata["trajectory_poses"][-1]["x"] == 2
    assert len(plotted) == 1 and plotted[0][-1]["rx"] == 2
