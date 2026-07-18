# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the online CoT/trajectory consistency monitor.

Covers the ConsistencyMonitor unit behaviour (lane extraction, frame
extraction, no-op fallbacks, judging) and the best-of-N resampling loop in
PolicyEvent. The judge LLM call is stubbed; no network or GPU is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
from alpasim_runtime import consistency_monitor as cm
from alpasim_runtime.config import ConsistencyMonitorConfig
from alpasim_runtime.consistency_monitor import (
    ConsistencyMonitor,
    JudgeResult,
    _lane_center_lines_from_vector_map,
    _trajectory_to_world_and_rig_xy,
)
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.policy import PolicyEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_runtime.services.driver_service import DriveOutcome
from alpasim_utils.geometry import Pose, Trajectory
from trajdata.maps.vec_map_elements import MapElementType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_world_traj(
    start: tuple[float, float] = (0.0, 0.0), n: int = 6, dx: float = 1.0
) -> Trajectory:
    """A straight trajectory heading +x in the world frame, identity rotation."""
    ts = np.array([i * 100_000 for i in range(n)], dtype=np.uint64)
    poses = [
        Pose(
            np.array([start[0] + i * dx, start[1], 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        for i in range(n)
    ]
    return Trajectory.from_poses(timestamps=ts, poses=poses)


class _FakeCenter:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float)


class _FakeLane:
    def __init__(self, xy):
        self.center = _FakeCenter(xy)


def _fake_vector_map(lanes_xy: list[np.ndarray]):
    """A minimal stand-in for a trajdata VectorMap with road-lane centers."""
    lanes = {f"lane_{i}": _FakeLane(xy) for i, xy in enumerate(lanes_xy)}
    return SimpleNamespace(elements={MapElementType.ROAD_LANE: lanes})


def _straight_lane() -> np.ndarray:
    return np.array([[x, 0.0] for x in np.linspace(-5.0, 20.0, 26)], dtype=float)


def _enabled_monitor(**overrides) -> ConsistencyMonitor:
    config = ConsistencyMonitorConfig(enabled=True, **overrides)
    monitor = ConsistencyMonitor.from_config(config)
    assert monitor is not None
    return monitor


# ---------------------------------------------------------------------------
# ConsistencyMonitor unit tests
# ---------------------------------------------------------------------------


class TestConsistencyMonitorUnit:
    def test_from_config_disabled_returns_none(self):
        assert ConsistencyMonitor.from_config(None) is None
        assert ConsistencyMonitor.from_config(ConsistencyMonitorConfig()) is None

    def test_unsupported_prompt_rejected(self):
        with pytest.raises(ValueError):
            ConsistencyMonitor.from_config(
                ConsistencyMonitorConfig(enabled=True, prompt="not_a_prompt")
            )

    def test_lane_extraction_from_vector_map(self):
        lanes = [_straight_lane(), np.array([[0, 5], [1, 5], [2, 5]], dtype=float)]
        lines = _lane_center_lines_from_vector_map(_fake_vector_map(lanes))
        assert lines is not None
        assert len(lines) == 2
        assert all(line.ndim == 2 and line.shape[1] == 2 for line in lines)

    def test_lane_extraction_none_map(self):
        assert _lane_center_lines_from_vector_map(None) is None

    def test_world_and_rig_extraction(self):
        # Anchor at (5, 3); identity rotation -> rig is world shifted by -anchor.
        traj = _make_world_traj(start=(5.0, 3.0), n=6, dx=1.0)
        world_xy, rig_xy = _trajectory_to_world_and_rig_xy(traj)
        assert world_xy is not None and rig_xy is not None
        assert world_xy.shape == (5, 2)
        np.testing.assert_allclose(world_xy[:, 0], [6, 7, 8, 9, 10], atol=1e-4)
        np.testing.assert_allclose(rig_xy, world_xy - np.array([5.0, 3.0]), atol=1e-4)

    def test_short_trajectory_extraction_returns_none(self):
        traj = _make_world_traj(n=2)
        world_xy, rig_xy = _trajectory_to_world_and_rig_xy(traj)
        assert world_xy is None and rig_xy is None

    def test_should_monitor_step_every_n(self):
        monitor = _enabled_monitor(monitor_every_n_steps=2)
        monitor.prepare_for_scene(_fake_vector_map([_straight_lane()]))
        assert [monitor.should_monitor_step() for _ in range(4)] == [
            True,
            False,
            True,
            False,
        ]

    def test_no_map_noop(self):
        monitor = _enabled_monitor()
        monitor.prepare_for_scene(None)
        assert monitor.lane_geometry_available is False
        assert monitor.should_monitor_step() is False
        result = monitor.judge(_make_world_traj(), "Keep lane.")
        assert result.judged is False
        assert result.is_consistent is True  # un-judgeable trajectories pass

    def test_no_cot_noop(self):
        monitor = _enabled_monitor()
        monitor.prepare_for_scene(_fake_vector_map([_straight_lane()]))
        for cot in ("", "   ", None):
            result = monitor.judge(_make_world_traj(), cot)
            assert result.judged is False

    def test_short_trajectory_noop(self):
        monitor = _enabled_monitor()
        monitor.prepare_for_scene(_fake_vector_map([_straight_lane()]))
        result = monitor.judge(_make_world_traj(n=2), "Keep lane.")
        assert result.judged is False

    def test_judge_happy_path(self, monkeypatch: pytest.MonkeyPatch):
        monitor = _enabled_monitor()
        monitor.prepare_for_scene(_fake_vector_map([_straight_lane()]))
        # Bypass real client construction; stub the LLM call.
        monitor._client = object()
        monitor._provider_settings = {
            "model": "stub-model",
            "temperature": 0,
            "extra_params": {},
        }
        captured = {}

        def _fake_call_llm(client, model, prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"cot_output_alignment": {"score": 4, "justification": "ok"}}'

        monkeypatch.setattr(cm, "call_llm", _fake_call_llm)

        result = monitor.judge(_make_world_traj(), "Keep lane and hold speed.")
        assert result.judged is True
        assert result.score == 4.0
        assert result.is_consistent is True
        assert monitor.is_accepted(result) is True
        assert "Keep lane and hold speed." in captured["prompt"]

    def test_judge_inconsistent_score(self, monkeypatch: pytest.MonkeyPatch):
        monitor = _enabled_monitor()
        monitor.prepare_for_scene(_fake_vector_map([_straight_lane()]))
        monitor._client = object()
        monitor._provider_settings = {
            "model": "stub-model",
            "temperature": 0,
            "extra_params": {},
        }
        monkeypatch.setattr(
            cm,
            "call_llm",
            lambda *a, **k: '{"cot_output_alignment": {"score": 1, "justification": "no"}}',
        )
        result = monitor.judge(_make_world_traj(), "Turn left now.")
        assert result.judged is True
        assert result.score == 1.0
        assert result.is_consistent is False
        assert monitor.is_accepted(result) is False

    def test_judge_client_unavailable_noop(self, monkeypatch: pytest.MonkeyPatch):
        monitor = _enabled_monitor()
        monitor.prepare_for_scene(_fake_vector_map([_straight_lane()]))
        # Force client construction to fail -> monitor no-ops gracefully.
        monkeypatch.setattr(
            cm,
            "resolve_provider",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = monitor.judge(_make_world_traj(), "Keep lane.")
        assert result.judged is False

    def test_append_audit_writes_jsonl(self, tmp_path):
        monitor = _enabled_monitor()
        audit = tmp_path / "consistency_monitor.jsonl"
        monitor.prepare_for_scene(
            _fake_vector_map([_straight_lane()]), audit_path=audit
        )
        monitor.append_audit({"time_now_us": 1, "chosen_attempt": 0})
        monitor.append_audit({"time_now_us": 2, "chosen_attempt": 1})
        lines = audit.read_text().strip().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# PolicyEvent best-of-N loop tests
# ---------------------------------------------------------------------------


class _StubMonitor:
    """Scripted monitor for exercising PolicyEvent's resampling loop."""

    def __init__(
        self,
        results: list[JudgeResult],
        *,
        accept_threshold: float = 3.0,
        max_samples: int = 3,
        do_monitor: bool = True,
    ):
        self.config = SimpleNamespace(
            max_samples=max_samples, accept_threshold=accept_threshold
        )
        self._results = list(results)
        self._idx = 0
        self._do_monitor = do_monitor
        self.audits: list[dict] = []
        self.judged_cots: list[str | None] = []

    def should_monitor_step(self) -> bool:
        return self._do_monitor

    def judge(self, trajectory, reasoning_text) -> JudgeResult:
        self.judged_cots.append(reasoning_text)
        result = self._results[min(self._idx, len(self._results) - 1)]
        self._idx += 1
        return result

    def is_accepted(self, result: JudgeResult) -> bool:
        return result.score is not None and result.score >= self.config.accept_threshold

    def append_audit(self, record: dict) -> None:
        self.audits.append(record)


def _policy_event(service_bundle: ServiceBundle, monitor: Any) -> PolicyEvent:
    return PolicyEvent(
        timestamp_us=200_000,
        policy_timestep_us=100_000,
        services=service_bundle,
        camera_ids=["cam_front"],
        route_generator=None,
        send_recording_ground_truth=False,
        consistency_monitor=cast(Any, monitor),
    )


@pytest.mark.asyncio
class TestPolicyEventResamplingLoop:
    async def test_early_stop_on_accepted(
        self,
        service_bundle: ServiceBundle,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        monitor = _StubMonitor([JudgeResult(judged=True, score=5.0)])
        mock_driver.drive_with_debug.return_value = DriveOutcome(
            simple_trajectory, "Keep lane."
        )
        event = _policy_event(service_bundle, monitor)

        await event.run(rollout_state, EventQueue())

        assert mock_driver.drive_with_debug.await_count == 1
        assert mock_driver.drive.await_count == 0
        assert monitor.audits[0]["chosen_attempt"] == 0
        assert monitor.audits[0]["chosen_score"] == 5.0
        assert rollout_state.step_context.driver_trajectory is not None

    async def test_picks_highest_when_none_accepted(
        self,
        service_bundle: ServiceBundle,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        rollout_state.data_sensorsim_to_driver = b"frame-bytes"
        monitor = _StubMonitor(
            [
                JudgeResult(judged=True, score=2.0),
                JudgeResult(judged=True, score=3.0),
                JudgeResult(judged=True, score=1.0),
            ],
            accept_threshold=4.0,
            max_samples=3,
        )
        mock_driver.drive_with_debug.return_value = DriveOutcome(
            simple_trajectory, "cot"
        )
        event = _policy_event(service_bundle, monitor)

        await event.run(rollout_state, EventQueue())

        assert mock_driver.drive_with_debug.await_count == 3
        assert monitor.audits[0]["chosen_attempt"] == 1
        assert monitor.audits[0]["chosen_score"] == 3.0
        # renderer_data only on the first attempt; consumed afterward.
        calls = mock_driver.drive_with_debug.call_args_list
        assert calls[0].kwargs["renderer_data"] == b"frame-bytes"
        assert calls[1].kwargs["renderer_data"] is None
        assert calls[2].kwargs["renderer_data"] is None
        assert rollout_state.data_sensorsim_to_driver is None

    async def test_ties_prefer_last_highest(
        self,
        service_bundle: ServiceBundle,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        monitor = _StubMonitor(
            [JudgeResult(judged=True, score=3.0), JudgeResult(judged=True, score=3.0)],
            accept_threshold=5.0,
            max_samples=2,
        )
        mock_driver.drive_with_debug.return_value = DriveOutcome(
            simple_trajectory, "cot"
        )
        event = _policy_event(service_bundle, monitor)

        await event.run(rollout_state, EventQueue())

        assert mock_driver.drive_with_debug.await_count == 2
        assert monitor.audits[0]["chosen_attempt"] == 1

    async def test_unjudgeable_executes_first_sample(
        self,
        service_bundle: ServiceBundle,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        monitor = _StubMonitor(
            [JudgeResult(judged=False, error="no chain-of-thought")], max_samples=3
        )
        mock_driver.drive_with_debug.return_value = DriveOutcome(
            simple_trajectory, None
        )
        event = _policy_event(service_bundle, monitor)

        await event.run(rollout_state, EventQueue())

        assert mock_driver.drive_with_debug.await_count == 1
        assert monitor.audits[0]["chosen_attempt"] == 0
        assert monitor.audits[0]["chosen_score"] is None
        assert rollout_state.step_context.driver_trajectory is not None

    async def test_disabled_step_uses_plain_drive(
        self,
        service_bundle: ServiceBundle,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        monitor = _StubMonitor([], do_monitor=False)
        mock_driver.drive.return_value = simple_trajectory.clip(200_000, 300_001)
        event = _policy_event(service_bundle, monitor)

        await event.run(rollout_state, EventQueue())

        assert mock_driver.drive.await_count == 1
        assert mock_driver.drive_with_debug.await_count == 0
        assert monitor.audits == []
        assert rollout_state.step_context.driver_trajectory is not None
