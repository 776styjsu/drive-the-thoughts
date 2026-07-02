# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Policy event for event-based simulation loop.

Opens the per-step StepContext, gathers observations (egopose, route,
recording ground truth), queries the driver, and writes the transformed
trajectory into the context for downstream pipeline events.
"""

from __future__ import annotations

import logging

import numpy as np
from alpasim_runtime.consistency_monitor import ConsistencyMonitor
from alpasim_runtime.events.base import EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_utils import geometry

logger = logging.getLogger(__name__)


class PolicyEvent(RecurringEvent):
    """Open per-step context, gather observations, query driver.

    Handles egopose submission, sync tracking, route updates, and recording
    ground-truth submission.  Everything after the driver query (controller,
    physics, traffic, commit) is handled by downstream pipeline events.
    """

    priority: int = EventPriority.POLICY

    def __init__(
        self,
        timestamp_us: int,
        policy_timestep_us: int,
        services: ServiceBundle,
        camera_ids: list[str],
        route_generator: RouteGenerator | None,
        send_recording_ground_truth: bool,
        consistency_monitor: ConsistencyMonitor | None = None,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = policy_timestep_us
        self.services = services
        self.camera_ids = camera_ids
        self.route_generator = route_generator
        self.send_recording_ground_truth = send_recording_ground_truth
        self.consistency_monitor = consistency_monitor

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        step_start_us = self.timestamp_us
        target_time_us = step_start_us + self.interval_us
        svc = self.services

        # --- Step boundary: fill timing on existing StepContext ---
        assert (
            state.step_context is not None
        ), "StepContext must exist before PolicyEvent (created by StepEvent)"
        state.step_context.step_start_us = step_start_us
        state.step_context.target_time_us = target_time_us
        state.step_context.force_gt = target_time_us in state.unbound.force_gt_period

        # --- Sensor sync validation ---
        if state.unbound.assert_zero_decision_delay:
            assert_sensors_up_to_date(state, step_start_us, self.camera_ids)

        # --- Submit observations concurrently ---
        # Send all egomotion observations since the last update, not just the
        # latest one.  When pose_reporting_interval_us > 0 the controller
        # produces intermediate poses that StepEvent appends to the estimated
        # trajectory.  The driver should receive every one of them.
        all_ts = state.ego_trajectory_estimate.timestamps_us
        mask = (all_ts > state.last_egopose_update_us) & (all_ts <= step_start_us)
        ts_arr = all_ts[mask]
        if len(ts_arr) == 0:
            ts_arr = np.array([step_start_us], dtype=np.uint64)

        ego_trajectory = state.ego_trajectory_estimate.trajectory().interpolate(ts_arr)
        dynamics_arr = state.ego_trajectory_estimate.interpolate_dynamics(ts_arr)
        dynamic_states_in_rig = geometry.array_to_dynamic_states(dynamics_arr)

        if (self.route_generator is not None or self.send_recording_ground_truth) and (
            state.ego_trajectory.timestamps_us[-1] != step_start_us
        ):
            raise ValueError(
                f"Timestamp mismatch: {state.ego_trajectory.timestamps_us[-1]} "
                f"!= {step_start_us}"
            )

        ctx = state.step_context
        ctx.track_task(
            svc.driver.submit_trajectory(ego_trajectory, dynamic_states_in_rig)
        )

        if self.route_generator is not None:
            pose_local_to_rig = state.ego_trajectory.last_pose
            route = self.route_generator.generate_route(
                step_start_us, pose_local_to_rig
            )
            route = RouteGenerator.prepare_for_policy(route)
            ctx.track_task(svc.driver.submit_route(step_start_us, route))

        if self.send_recording_ground_truth:
            gt_traj = state.unbound.gt_ego_trajectory
            pose_local_to_rig = state.ego_trajectory.last_pose
            traj_in_rig = gt_traj.transform(pose_local_to_rig.inverse())
            ctx.track_task(
                svc.driver.submit_recording_ground_truth(step_start_us, traj_in_rig)
            )

        # Barrier: all observations (images + egopose + route + GT) must
        # reach the driver before we call drive().
        await ctx.drain_outstanding_tasks()
        state.last_egopose_update_us = step_start_us

        # --- Driver query ---
        renderer_data = state.data_sensorsim_to_driver
        state.data_sensorsim_to_driver = None  # Consumed

        monitor = self.consistency_monitor
        if monitor is not None and monitor.should_monitor_step():
            drive_trajectory = await self._drive_with_consistency_monitor(
                svc, state, step_start_us, target_time_us, renderer_data, monitor
            )
        else:
            drive_trajectory_noisy = await svc.driver.drive(
                time_now_us=step_start_us,
                time_query_us=target_time_us,
                renderer_data=renderer_data,
            )
            # --- Transform from noisy to true local frame ---
            drive_trajectory = transform_trajectory_from_noisy_to_true_local_frame(
                state, drive_trajectory_noisy
            )

        state.step_context.driver_trajectory = drive_trajectory

    async def _drive_with_consistency_monitor(
        self,
        svc: ServiceBundle,
        state: RolloutState,
        step_start_us: int,
        target_time_us: int,
        renderer_data: bytes | None,
        monitor: ConsistencyMonitor,
    ) -> geometry.Trajectory:
        """Sample-then-judge loop: pick a CoT-consistent trajectory to execute.

        Re-samples the driver up to ``max_samples`` times. Each sample is a fresh
        stochastic (CoT, trajectory). Executes the first sample whose judge score
        clears ``accept_threshold``; otherwise the highest-scored sample. If a
        sample cannot be judged (no CoT / too short / backend error), falls back
        to the best judged sample so far, or that sample when none were judged
        (today's single-call behaviour). The alpamayo driver ignores
        ``renderer_data`` and reuses cached frames, so resampling is safe; it is
        sent only on the first attempt to mirror the single-call path.
        """
        max_samples = max(1, monitor.config.max_samples)
        best_traj: geometry.Trajectory | None = None
        best_score: float | None = None
        best_attempt = -1
        last_traj: geometry.Trajectory | None = None
        attempts: list[dict] = []

        for attempt in range(max_samples):
            outcome = await svc.driver.drive_with_debug(
                time_now_us=step_start_us,
                time_query_us=target_time_us,
                renderer_data=renderer_data if attempt == 0 else None,
            )
            drive_trajectory = transform_trajectory_from_noisy_to_true_local_frame(
                state, outcome.trajectory
            )
            last_traj = drive_trajectory
            result = monitor.judge(drive_trajectory, outcome.reasoning_text)
            attempts.append(
                {
                    "attempt": attempt,
                    "judged": result.judged,
                    "score": result.score,
                    "has_cot": bool(outcome.reasoning_text),
                    "error": result.error,
                }
            )

            if result.judged and result.score is not None:
                # Prefer the later sample on score ties ("last highest scored").
                if best_score is None or result.score >= best_score:
                    best_score, best_traj, best_attempt = (
                        result.score,
                        drive_trajectory,
                        attempt,
                    )
                if monitor.is_accepted(result):
                    break
            else:
                # Cannot judge this sample: stop resampling. Keep the best judged
                # sample so far, or execute this one if none were judged.
                if best_traj is None:
                    best_traj, best_attempt = drive_trajectory, attempt
                break

        chosen_traj = best_traj if best_traj is not None else last_traj
        monitor.append_audit(
            {
                "time_now_us": step_start_us,
                "num_attempts": len(attempts),
                "chosen_attempt": best_attempt,
                "chosen_score": best_score,
                "accept_threshold": monitor.config.accept_threshold,
                "attempts": attempts,
            }
        )
        logger.info(
            "consistency_monitor: t=%dus attempts=%d chosen_attempt=%d "
            "chosen_score=%s scores=%s",
            step_start_us,
            len(attempts),
            best_attempt,
            best_score,
            [a["score"] for a in attempts],
        )
        return chosen_traj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_sensors_up_to_date(
    state: RolloutState, step_start_us: int, camera_ids: list[str]
) -> None:
    """Validate that egopose and all camera frames are current before the policy decision."""
    # --- egopose freshness ---
    latest_ego_us = int(state.ego_trajectory_estimate.timestamps_us[-1])
    if latest_ego_us != step_start_us:
        raise ValueError(
            f"Egopose not up to date at {step_start_us}: "
            f"ego_trajectory_estimate latest timestamp is {latest_ego_us}"
        )

    # --- camera freshness ---
    if not state.last_camera_frame_us:
        return  # First step — no cameras have fired yet

    stale = [
        cid
        for cid in camera_ids
        if state.last_camera_frame_us.get(cid, 0) != step_start_us
    ]
    if stale:
        raise ValueError(f"Cameras not up to date at {step_start_us}: {stale}")


def transform_trajectory_from_noisy_to_true_local_frame(
    state: RolloutState, drive_trajectory_noisy: geometry.Trajectory
) -> geometry.Trajectory:
    """Transform trajectory from noisy local frame to true local frame.

    The driver operates in the "noisy" (estimated) rig frame. To map its output
    into the true local frame we:

    1. Undo the estimated rig frame:  ``T_estimate_inv * traj``
    2. Apply the true rig frame:      ``T_true * result``

    When no egomotion noise model is active, ``ego_trajectory_estimate`` tracks
    ``ego_trajectory`` exactly and the transform is identity.  When noise is
    present the two trajectories diverge and this mapping compensates for the
    drift the driver doesn't know about.
    """
    return drive_trajectory_noisy.transform(
        state.ego_trajectory_estimate.last_pose.inverse()
    ).transform(state.ego_trajectory.last_pose)
