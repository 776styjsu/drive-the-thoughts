# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for the opt-in online CoT consistency scorer.

These stub the minimal ``SimulationResult`` surface the scorer touches
(``timestamps_us`` and ``driver_responses.get_driver_response_for_time``) so the
scoring logic is exercised without a map or a full simulation.
"""

from __future__ import annotations

import numpy as np

from eval.data import AggregationType
from eval.scorers.cot_consistency import CotConsistencyScorer, _rig_metadata


class _FakeTraj:
    def __init__(self, positions: np.ndarray, timestamps_us: np.ndarray) -> None:
        self.positions = positions
        self.timestamps_us = timestamps_us

    def __len__(self) -> int:
        return len(self.positions)


class _FakeResponse:
    def __init__(self, now_time_us: int, reasoning_text, traj) -> None:
        self.now_time_us = now_time_us
        self.reasoning_text = reasoning_text
        self.selected_trajectory = traj


class _FakeDriverResponses:
    def __init__(self, by_time: dict[int, _FakeResponse | None]) -> None:
        self._by_time = by_time

    def get_driver_response_for_time(self, time, which_time="now"):
        return self._by_time.get(int(time))


class _FakeSim:
    def __init__(self, timestamps_us, by_time) -> None:
        self.timestamps_us = np.asarray(timestamps_us, dtype=np.uint64)
        self.driver_responses = _FakeDriverResponses(by_time)


def _straight_traj(n: int = 12) -> _FakeTraj:
    xs = np.linspace(0.0, 22.0, n)
    pos = np.column_stack([xs, np.zeros(n), np.zeros(n)])
    ts = np.linspace(0, 5_000_000, n).astype(np.uint64)
    return _FakeTraj(pos, ts)


def test_rig_metadata_anchors_first_point():
    traj = _straight_traj()
    meta = _rig_metadata(
        np.asarray(traj.positions)[:, :2], np.asarray(traj.timestamps_us)
    )
    assert meta is not None
    assert abs(meta["trajectory_xy_rig_frame"][0]["rx"]) < 1e-9
    assert len(meta["trajectory_xy_rig_frame"]) == len(traj)


def test_no_cot_is_inert():
    traj = _straight_traj()
    sim = _FakeSim(
        [0, 1], {0: _FakeResponse(0, None, traj), 1: _FakeResponse(0, None, traj)}
    )
    metrics = {m.name: m for m in CotConsistencyScorer(cfg=None).calculate(sim)}
    assert set(metrics) == {"cot_inconsistent", "cot_consistency_score"}
    assert metrics["cot_inconsistent"].values == [False, False]
    assert metrics["cot_consistency_score"].values == [1.0, 1.0]
    assert metrics["cot_inconsistent"].time_aggregation == AggregationType.MAX
    assert metrics["cot_consistency_score"].time_aggregation == AggregationType.MIN


def test_missing_response_is_inert():
    sim = _FakeSim([5, 6], {5: None, 6: None})
    metrics = {m.name: m for m in CotConsistencyScorer(cfg=None).calculate(sim)}
    assert metrics["cot_inconsistent"].values == [False, False]


def test_contradictory_cot_is_flagged():
    # A straight, steady trajectory while the CoT claims a left turn + braking
    # should be scored inconsistent by the rule monitor.
    traj = _straight_traj()
    cot = "Turn left now to take the exit and brake hard to stop for the red light"
    sim = _FakeSim([0], {0: _FakeResponse(0, cot, traj)})
    metrics = {m.name: m for m in CotConsistencyScorer(cfg=None).calculate(sim)}
    assert metrics["cot_inconsistent"].values[0] is True
    assert metrics["cot_consistency_score"].values[0] == 0.0
