# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Per-entry judging pipeline and result aggregation."""

from __future__ import annotations

import logging

import numpy as np
from alpasim_utils.cot_consistency import (
    DEFAULT_SEED,
    call_llm,
    compute_trajectory_features,
    parse_response,
)
from benchmark_analysis import flatten_cot

from .prompts import resolve_prompt_builder

logger = logging.getLogger(__name__)

# Dimensions to score.
DIMENSIONS = ["cot_output_alignment"]


def _use_cached_features(
    cached_features: dict | None, trajectory_frame: str, lane_reference: str
) -> bool:
    """Whether features cached in a source report match the requested frames."""
    if not isinstance(cached_features, dict):
        return False
    cached_stats = cached_features.get("summary_stats", {})
    if not isinstance(cached_stats, dict):
        cached_stats = {}
    if trajectory_frame == "ego_rig":
        return True
    cached_lane_reference = cached_stats.get("lane_reference")
    lane_reference_matches = cached_lane_reference == lane_reference or (
        lane_reference == "auto"
        and cached_lane_reference in {"auto", "map_graph", "route"}
    )
    return cached_stats.get("reference_frame") == trajectory_frame and (
        lane_reference_matches
    )


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
    """Judge one extracted entry: features -> prompt -> LLM -> parsed result."""
    clip_id = entry["clip_id"]
    cot_text = flatten_cot(entry["chain_of_thought"])

    cached_features = entry.get("trajectory_features")
    if _use_cached_features(cached_features, trajectory_frame, lane_reference):
        traj_features = dict(cached_features)
    else:
        traj_features = compute_trajectory_features(
            entry["trajectory_xy"],
            route_xy=entry.get("route_xy"),
            trajectory_world_xy=entry.get("trajectory_map_xy"),
            lane_center_lines=entry.get("lane_center_lines"),
            reference_frame=trajectory_frame,
            lane_reference=lane_reference,
        )

    prompt_builder = resolve_prompt_builder(prompt_name)
    prompt = prompt_builder(cot_text, traj_features)

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


def log_feature_summary(stats: dict) -> None:
    """One-line log of the computed trajectory features for an entry."""
    if stats.get("reference_frame") == "dual":
        lane_stats = stats.get("lane", {})
        ego_stats = stats.get("ego", {})
        if lane_stats.get("reference_frame") == "lane_center":
            logger.info(
                "  [dual] Lane progress: %.1fm | Offset: %.1fm -> %.1fm | "
                "Reference: %s | Final pos: (%.1fm fwd, %.1fm lat) | "
                "Speed: %.1f m/s avg",
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
            "  Lane progress: %.1fm | Offset: %.1fm -> %.1fm | "
            "Reference: %s | Speed: %.1f m/s avg",
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


def log_evaluation_summary(evaluation: dict) -> None:
    """One-line log of the judge's verdict/score for an entry."""
    if evaluation.get("dry_run"):
        return
    for dim in DIMENSIONS:
        dim_data = evaluation.get(dim, {})
        if not isinstance(dim_data, dict):
            continue
        if "score" in dim_data:
            logger.info(
                "  %s: %s/5 — %s",
                dim,
                dim_data["score"],
                dim_data.get("justification", "")[:60],
            )
        elif dim_data.get("verdict"):
            label = dim_data["verdict"]
            if dim_data.get("inconsistency_type"):
                label += f" ({dim_data['inconsistency_type']})"
            logger.info(
                "  %s: %s — %s",
                dim,
                label,
                dim_data.get("justification", "")[:60],
            )
