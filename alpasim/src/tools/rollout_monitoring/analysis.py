# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Join per-step monitor flags with per-timestep simulated safety outcomes.

The Alpasim evaluator already wrote a long-format ``metrics.parquet`` next to
each ``rollout.asl`` with one row per ``(metric name, timestep)``. That file is
*pre-aggregation*: it holds the raw per-timestep metric values before the
evaluator's aggregation modifiers are applied. To recover the same safety
verdict the evaluator reports we replicate the relevant modifiers (their exact
form is printed in ``metrics_results.txt``):

* ``RemoveTimestepsBeforeEvent(eval_relevant > 0)`` -- the scenario only starts
  counting once it becomes eval-relevant;
* ``RemoveTimestepsAfterEvent(offroad_or_collision > 0)`` -- the rollout is
  decided at the *first* off-road / any-collision event, so later timesteps do
  not count (this is why a clip whose first collision is a not-at-fault rear hit
  scores ``collision_at_fault = 0`` even if it later rear-ends into an at-fault
  geometry);
* ``RemoveTimestepsAfterEvent(dist_to_gt_trajectory >= max_dist)`` -- once the
  ego has drifted too far from the logged trajectory the rendered scene is no
  longer valid;
* ``RemoveTrajectoryWithEvent(img_is_black > 0)`` -- drop a rollout whose render
  went black.

Reconstructed at-fault outcomes (mirroring the printed modifiers)::

    collision_at_fault             = collision_front | collision_lateral
    offroad_or_collision_at_fault  = collision_at_fault | offroad

A *failure* for an outcome means one of its component metrics fired inside the
valid window ``[eval_start, window_end]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .monitor import MonitorStep

#: Outcome name -> the raw per-timestep metrics whose union defines a failure.
OUTCOME_COMPONENTS: dict[str, tuple[str, ...]] = {
    "offroad_or_collision_at_fault": (
        "offroad",
        "collision_front",
        "collision_lateral",
    ),
    "offroad_or_collision": ("offroad", "collision_any"),
    "collision_at_fault": ("collision_front", "collision_lateral"),
    "collision_any": ("collision_any",),
    "offroad": ("offroad",),
}
DEFAULT_OUTCOME = "offroad_or_collision_at_fault"
DEFAULT_MAX_DIST_TO_GT_M = 4.0

#: Metrics that terminate the valid window at their first occurrence.
_TERMINATING_EVENTS = ("collision_any", "offroad")
#: Which boolean field of :class:`MonitorStep` is used as the safety flag. The
#: exact-label matcher yields a single flag; this stays a tuple as the
#: extension point for stricter or learned step-level predictors.
PREDICTORS = ("flag",)


@dataclass
class OutcomeSeries:
    """Per-timestep simulated outcomes for one rollout with the valid window."""

    eval_start_us: int
    window_end_us: int  # truncated end: first of {collision/offroad, dist>=max, end}
    img_black: bool  # rollout should be excluded (render went black)
    raw: dict[str, tuple[np.ndarray, np.ndarray]]  # metric -> (sorted ts, values)

    def _first_event(
        self, metric: str, thresh: float = 0.0, *, ge: bool = False
    ) -> int | None:
        """First timestamp (>= eval_start) where ``metric`` crosses ``thresh``."""
        tv = self.raw.get(metric)
        if tv is None:
            return None
        ts, val = tv
        crossed = (val >= thresh) if ge else (val > thresh)
        mask = crossed & (ts >= self.eval_start_us)
        return int(ts[mask].min()) if mask.any() else None

    def _fires_in_window(self, metric: str) -> bool:
        tv = self.raw.get(metric)
        if tv is None:
            return False
        ts, val = tv
        mask = (val > 0) & (ts >= self.eval_start_us) & (ts <= self.window_end_us)
        return bool(mask.any())

    def failure_time_us(self, outcome: str) -> int | None:
        """First failure time (us) for ``outcome`` inside the valid window."""
        times: list[int] = []
        for metric in OUTCOME_COMPONENTS[outcome]:
            t = self._first_event(metric, 0.0)
            if t is not None and t <= self.window_end_us:
                times.append(t)
        return min(times) if times else None

    def failed(self, outcome: str) -> bool:
        return self.failure_time_us(outcome) is not None


def load_outcome_series(
    metrics_parquet: str | Path, *, max_dist_to_gt_m: float = DEFAULT_MAX_DIST_TO_GT_M
) -> OutcomeSeries:
    """Read the long-format metrics parquet and compute the valid window."""
    df = pl.read_parquet(str(metrics_parquet))
    df = df.filter(pl.col("timestamps_us").is_not_null())
    if df["timestamps_us"].dtype in (pl.Float32, pl.Float64):
        df = df.filter(~pl.col("timestamps_us").is_nan())

    names = df["name"].to_numpy()
    ts_all = df["timestamps_us"].to_numpy().astype(np.int64)
    vals_all = df["values"].to_numpy().astype(float)

    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in np.unique(names).tolist():
        mask = names == metric
        ts = ts_all[mask]
        order = np.argsort(ts)
        raw[metric] = (ts[order], vals_all[mask][order])

    eval_rel = raw.get("eval_relevant")
    if eval_rel is not None and (eval_rel[1] > 0).any():
        eval_start = int(eval_rel[0][eval_rel[1] > 0].min())
    else:
        eval_start = int(ts_all.min()) if ts_all.size else 0
    eval_end = int(ts_all.max()) if ts_all.size else eval_start

    base = OutcomeSeries(eval_start, eval_end, False, raw)
    ends = [eval_end]
    for metric in _TERMINATING_EVENTS:
        t = base._first_event(metric, 0.0)
        if t is not None:
            ends.append(t)
    t_dist = base._first_event("dist_to_gt_trajectory", max_dist_to_gt_m, ge=True)
    if t_dist is not None:
        ends.append(t_dist)
    window_end = min(ends)

    series = OutcomeSeries(eval_start, window_end, False, raw)
    series.img_black = series._fires_in_window("img_is_black")
    return series


@dataclass
class RolloutResult:
    """Leading-indicator result for a single closed-loop rollout."""

    clip_id: str
    rollout_id: str
    outcome: str
    predictor: str
    horizon_s: float | None
    # monitor coverage
    n_steps: int  # driver decision steps in the ASL
    n_scored: int  # steps with a usable monitor verdict in the eval window
    n_flag: int  # inconsistency flags in the eval window
    inconsistency_rate: float
    # simulated outcome
    failed: bool
    failure_time_us: int | None
    collision_any: bool
    offroad: bool
    collision_at_fault: bool
    safety_monitor_triggered: bool
    # leading-indicator alignment (predictor flags vs. the failure)
    first_flag_us: int | None
    flag_before_failure: bool  # failed & a flag fired strictly before failure
    flag_within_horizon: bool  # failed & a flag fired in [t_fail - H, t_fail)
    lead_time_s: float | None  # t_fail - earliest pre-failure flag (max lead)
    false_alarm: bool  # safe & at least one flag fired
    # per-flag precision support (aggregated across rollouts upstream)
    n_flags_followed_by_failure: int  # flags with a failure in (t, t+H]

    def to_dict(self) -> dict:
        return asdict(self)


def _flag_times(
    steps: list[MonitorStep], predictor: str, lo_us: int, hi_us: int
) -> np.ndarray:
    """Sorted flag times for ``predictor`` within ``[lo_us, hi_us]`` (inclusive)."""
    times = [
        s.now_us for s in steps if getattr(s, predictor) and lo_us <= s.now_us <= hi_us
    ]
    return np.sort(np.asarray(times, dtype=np.int64))


def analyze_rollout(
    steps: list[MonitorStep],
    series: OutcomeSeries,
    *,
    clip_id: str,
    rollout_id: str,
    outcome: str = DEFAULT_OUTCOME,
    predictor: str = "flag",
    horizon_s: float | None = 4.0,
) -> RolloutResult:
    """Compute the leading-indicator result for one rollout.

    Args:
        steps: Per-step monitor verdicts from :func:`monitor.monitor_steps`.
        series: Per-timestep outcomes from :func:`load_outcome_series`.
        outcome: Which simulated failure to predict (see ``OUTCOME_COMPONENTS``).
        predictor: Monitor flag field to use (``"flag"`` or ``"contradiction"``).
        horizon_s: Lead-time horizon (seconds) for within-horizon recall and the
            per-flag precision; ``None`` uses the whole pre-failure window.
    """
    if outcome not in OUTCOME_COMPONENTS:
        raise ValueError(
            f"unknown outcome {outcome!r}; choose from {list(OUTCOME_COMPONENTS)}"
        )
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor {predictor!r}; choose from {PREDICTORS}")

    lo, hi = series.eval_start_us, series.window_end_us
    scored = [s for s in steps if s.scored and lo <= s.now_us <= hi]
    n_scored = len(scored)
    n_flag = sum(1 for s in scored if getattr(s, predictor))
    rate = n_flag / n_scored if n_scored else 0.0

    failure_time = series.failure_time_us(outcome)
    failed = failure_time is not None
    horizon_us = int(horizon_s * 1_000_000) if horizon_s is not None else None

    first_flag_us: int | None = None
    flag_before_failure = False
    flag_within_horizon = False
    lead_time_s: float | None = None
    false_alarm = False
    n_flags_followed = 0

    if failed:
        pre = _flag_times(steps, predictor, lo, failure_time - 1)
        if pre.size:
            first_flag_us = int(pre.min())
            flag_before_failure = True
            lead_time_s = (failure_time - first_flag_us) / 1_000_000
            flag_within_horizon = (
                True
                if horizon_us is None
                else bool((pre >= failure_time - horizon_us).any())
            )
        all_flags = _flag_times(steps, predictor, lo, hi)
        if horizon_us is None:
            n_flags_followed = int((all_flags < failure_time).sum())
        else:
            n_flags_followed = int(
                (
                    (all_flags < failure_time)
                    & (all_flags >= failure_time - horizon_us)
                ).sum()
            )
    else:
        any_flags = _flag_times(steps, predictor, lo, hi)
        false_alarm = bool(any_flags.size)
        if any_flags.size:
            first_flag_us = int(any_flags.min())

    return RolloutResult(
        clip_id=clip_id,
        rollout_id=rollout_id,
        outcome=outcome,
        predictor=predictor,
        horizon_s=horizon_s,
        n_steps=len(steps),
        n_scored=n_scored,
        n_flag=n_flag,
        inconsistency_rate=round(rate, 4),
        failed=failed,
        failure_time_us=failure_time,
        collision_any=series._fires_in_window("collision_any"),
        offroad=series._fires_in_window("offroad"),
        collision_at_fault=(
            series._fires_in_window("collision_front")
            or series._fires_in_window("collision_lateral")
        ),
        safety_monitor_triggered=series._fires_in_window("safety_monitor_triggered"),
        first_flag_us=first_flag_us,
        flag_before_failure=flag_before_failure,
        flag_within_horizon=flag_within_horizon,
        lead_time_s=round(lead_time_s, 3) if lead_time_s is not None else None,
        false_alarm=false_alarm,
        n_flags_followed_by_failure=n_flags_followed,
    )
