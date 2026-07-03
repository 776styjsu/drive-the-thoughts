# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Online CoT/trajectory consistency monitor.

Judges, in the simulation loop and *before a trajectory is executed*, whether the
driver model's chain-of-thought is consistent with the trajectory it produced,
using the F-LLM ``map_graph`` approach: project the predicted trajectory into a
route-consistent lane-center Frenet frame built from the scene map, compute
numerical trajectory features, and ask an LLM judge to score CoT-Output
alignment on a 1-5 scale (``<= 2`` = inconsistent).

The select-best-of-N resampling loop lives in
:class:`alpasim_runtime.events.policy.PolicyEvent` (it owns the driver RPC); this
module owns the per-trajectory judging and the scene-level lane geometry. The
shared judge core lives in :mod:`alpasim_utils.cot_consistency` so the runtime
can reuse it without importing ``alpasim-tools`` (which depends on the runtime).

The monitor degrades gracefully: if the scene has no map, a step has no CoT, the
trajectory is too short, or the judge endpoint is unreachable, :meth:`judge`
returns a result with ``judged=False`` and the caller falls back to executing the
first sampled trajectory unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from alpasim_runtime.config import ConsistencyMonitorConfig
from alpasim_utils.cot_consistency import (
    build_center_of_lane_prompt,
    call_llm,
    compute_trajectory_features,
    parse_response,
    resolve_provider,
    score_from_evaluation,
)
from alpasim_utils.geometry import Trajectory

logger = logging.getLogger(__name__)

# Only the turn-aware lane-center prompt is wired for online judging. Keep the
# old v5 name as a compatibility alias for existing configs and scripts.
_PROMPT_BUILDERS = {
    "center_of_lane": build_center_of_lane_prompt,
    "center_of_lane_v5": build_center_of_lane_prompt,
}

_ALIGNMENT_DIMENSION = "cot_output_alignment"


@dataclass
class JudgeResult:
    """Outcome of judging one (CoT, trajectory) pair.

    ``judged`` is False when no judge call was made (no map, no CoT, too-short
    trajectory, or backend error); in that case ``score`` is None and the
    resampling loop should treat the trajectory as acceptable (execute it).
    """

    judged: bool
    score: float | None = None
    evaluation: dict | None = None
    error: str | None = None

    @property
    def is_consistent(self) -> bool:
        """Whether this trajectory passes (or could not be judged)."""
        return (not self.judged) or self.score is None or self.score >= 3


@dataclass
class ConsistencyMonitor:
    """Per-rollout online consistency judge.

    Build with :meth:`from_config` (returns None when disabled). Call
    :meth:`prepare_for_scene` once with the scene map, then :meth:`judge` per
    candidate trajectory inside the resampling loop.
    """

    config: ConsistencyMonitorConfig
    _lane_center_lines: list[np.ndarray] | None = field(default=None, init=False)
    _client: Any = field(default=None, init=False)
    _client_failed: bool = field(default=False, init=False)
    _provider_settings: dict | None = field(default=None, init=False)
    _prompt_builder: Any = field(default=None, init=False)
    _audit_path: Path | None = field(default=None, init=False)
    _step_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.config.prompt not in _PROMPT_BUILDERS:
            raise ValueError(
                f"Unsupported online consistency prompt '{self.config.prompt}'. "
                f"Supported: {sorted(_PROMPT_BUILDERS)}"
            )
        self._prompt_builder = _PROMPT_BUILDERS[self.config.prompt]

    @classmethod
    def from_config(
        cls, config: ConsistencyMonitorConfig | None
    ) -> "ConsistencyMonitor | None":
        """Create a monitor, or None when disabled/unset."""
        if config is None or not config.enabled:
            return None
        return cls(config=config)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def prepare_for_scene(
        self, vector_map: Any, audit_path: str | Path | None = None
    ) -> None:
        """Cache the scene's lane-center geometry and reset per-scene state.

        The map is static for a scene, so lane-center polylines are extracted
        once and reused for every step's judging.
        """
        self._lane_center_lines = _lane_center_lines_from_vector_map(vector_map)
        self._step_counter = 0
        self._audit_path = Path(audit_path) if audit_path is not None else None
        if self._audit_path is not None:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._lane_center_lines:
            logger.warning(
                "ConsistencyMonitor: scene has no lane-center geometry; "
                "monitor will no-op (executing the first sample each step)."
            )
        else:
            logger.info(
                "ConsistencyMonitor ready: %d lane-center polylines, provider=%s, "
                "prompt=%s, max_samples=%d, accept_threshold=%.1f, every_n_steps=%d",
                len(self._lane_center_lines),
                self.config.provider,
                self.config.prompt,
                self.config.max_samples,
                self.config.accept_threshold,
                self.config.monitor_every_n_steps,
            )

    @property
    def lane_geometry_available(self) -> bool:
        return bool(self._lane_center_lines)

    def should_monitor_step(self) -> bool:
        """Whether this control step should be judged (every Nth step).

        Stateful: advances an internal per-scene step counter, so call exactly
        once per control step. Always False when no lane geometry is available.
        """
        if not self._lane_center_lines:
            return False
        every_n = max(1, self.config.monitor_every_n_steps)
        do_monitor = (self._step_counter % every_n) == 0
        self._step_counter += 1
        return do_monitor

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    def judge(
        self, drive_trajectory: Trajectory, reasoning_text: str | None
    ) -> JudgeResult:
        """Score one candidate (CoT, trajectory) pair.

        Returns ``judged=False`` when judging is not possible/needed so the
        caller executes the trajectory unchanged.
        """
        if not self._lane_center_lines:
            return JudgeResult(judged=False, error="no lane geometry")

        cot_text = (reasoning_text or "").strip()
        if not cot_text:
            return JudgeResult(judged=False, error="no chain-of-thought")

        world_xy, rig_xy = _trajectory_to_world_and_rig_xy(drive_trajectory)
        if world_xy is None or rig_xy is None or len(world_xy) < 3:
            return JudgeResult(judged=False, error="trajectory too short")

        try:
            features = compute_trajectory_features(
                rig_xy,
                trajectory_world_xy=world_xy,
                lane_center_lines=self._lane_center_lines,
                reference_frame=self.config.trajectory_frame,
                lane_reference=self.config.lane_reference,
            )
            prompt = self._prompt_builder(cot_text, features)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ConsistencyMonitor: feature/prompt build failed: %s", exc)
            return JudgeResult(judged=False, error=f"feature build failed: {exc}")

        client = self._get_client()
        if client is None or self._provider_settings is None:
            return JudgeResult(judged=False, error="judge client unavailable")

        try:
            raw = call_llm(
                client,
                self._provider_settings["model"],
                prompt,
                seed=self.config.seed,
                temperature=self._provider_settings["temperature"],
                extra_params=self._provider_settings["extra_params"],
            )
        except Exception as exc:
            logger.warning("ConsistencyMonitor: judge call failed: %s", exc)
            return JudgeResult(judged=False, error=f"judge call failed: {exc}")

        parsed = parse_response(raw)
        score = score_from_evaluation(parsed, _ALIGNMENT_DIMENSION)
        if score is None:
            return JudgeResult(
                judged=False, evaluation=parsed, error="judge returned no score"
            )
        return JudgeResult(judged=True, score=score, evaluation=parsed)

    def is_accepted(self, result: JudgeResult) -> bool:
        """Whether a judged score clears the acceptance threshold."""
        return result.score is not None and result.score >= self.config.accept_threshold

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def append_audit(self, record: dict) -> None:
        """Append one per-step decision record to the sidecar JSONL, if enabled."""
        if self._audit_path is None:
            return
        try:
            with self._audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - non-fatal
            logger.warning("ConsistencyMonitor: could not write audit record: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily build the OpenAI-compatible client; cache failures to no-op."""
        if self._client is not None:
            return self._client
        if self._client_failed:
            return None
        try:
            from alpasim_utils.cot_consistency import build_client

            self._provider_settings = resolve_provider(
                self.config.provider,
                model=self.config.model,
                base_url=self.config.base_url,
            )
            self._client = build_client(
                self._provider_settings["api_key"],
                self._provider_settings["base_url"],
            )
            logger.info(
                "ConsistencyMonitor judge client: %s (model=%s, base_url=%s)",
                self._provider_settings["label"],
                self._provider_settings["model"],
                self._provider_settings["base_url"] or "<openai-default>",
            )
            return self._client
        except Exception as exc:
            logger.error(
                "ConsistencyMonitor: failed to build judge client (%s); "
                "monitor will no-op for this rollout.",
                exc,
            )
            self._client_failed = True
            return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _lane_center_lines_from_vector_map(vector_map: Any) -> list[np.ndarray] | None:
    """Extract world-frame ``road_lane_center`` polylines from a trajdata VectorMap.

    Mirrors the lane-center source used by the offline judge (the road-lane
    centers ``ShapelyMap.from_vec_map`` renders), so online and offline judging
    project onto the same reference.
    """
    if vector_map is None:
        return None
    try:
        from trajdata.maps.vec_map_elements import MapElementType

        lane_elements = vector_map.elements[MapElementType.ROAD_LANE]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ConsistencyMonitor: could not read lanes from map: %s", exc)
        return None

    lines: list[np.ndarray] = []
    for element in lane_elements.values():
        try:
            coords = np.asarray(element.center.xy, dtype=float)
        except Exception:
            continue
        if coords.ndim == 2 and coords.shape[0] >= 2 and coords.shape[1] >= 2:
            lines.append(coords[:, :2])
    return lines or None


def _trajectory_to_world_and_rig_xy(
    trajectory: Trajectory,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (world_xy, rig_xy) future positions from a drive trajectory.

    The trajectory's first pose is the current ego pose (anchor). World XY are
    the future positions in the local/map frame; rig XY are those positions
    expressed relative to the anchor (matching the offline judge's ego frame:
    X forward, Y left).
    """
    try:
        positions = np.asarray(trajectory.positions, dtype=float)
    except Exception:
        return None, None
    if positions.ndim != 2 or positions.shape[0] < 3:
        return None, None

    world_xy = positions[1:, :2]

    try:
        anchor_inverse = trajectory.first_pose.inverse()
        rig_positions = np.asarray(
            trajectory.transform(anchor_inverse).positions, dtype=float
        )
        rig_xy = rig_positions[1:, :2]
    except Exception:
        return world_xy, None
    return world_xy, rig_xy
