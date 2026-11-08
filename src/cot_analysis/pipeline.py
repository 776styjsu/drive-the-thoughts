# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Per-entry judging pipeline and result aggregation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alpasim_utils.cot_consistency import (
    DEFAULT_SEED,
    call_llm,
    compute_trajectory_features,
    parse_response,
)
from alpasim_utils.cot_consistency.llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    image_to_data_url,
)
from benchmark_analysis import flatten_cot

from .features import cached_features_for_frame
from .prompts import resolve_prompt_builder

logger = logging.getLogger(__name__)

# Dimensions to score.
DIMENSIONS = ["cot_output_alignment"]


@dataclass(frozen=True)
class JudgeInput:
    """Prepared evidence reused for resume checks and the actual judge call."""

    cot_text: str
    trajectory_features: dict
    prompt: str
    image: Any
    fingerprint: str


def prepare_judge_input(
    entry: dict,
    *,
    prompt_name: str = "default",
    trajectory_frame: str = "ego_rig",
    lane_reference: str = "auto",
    use_images: bool = True,
) -> JudgeInput:
    cot_text = flatten_cot(entry["chain_of_thought"])
    features = cached_features_for_frame(
        entry.get("trajectory_features"), trajectory_frame, lane_reference
    )
    if features is None:
        features = compute_trajectory_features(
            entry.get("trajectory_xy"),
            route_xy=entry.get("route_xy"),
            trajectory_world_xy=entry.get("trajectory_map_xy"),
            lane_center_lines=entry.get("lane_center_lines"),
            reference_frame=trajectory_frame,
            lane_reference=lane_reference,
        )
    if features.get("error"):
        raise ValueError(f"Cannot prepare trajectory evidence: {features['error']}")
    prompt = resolve_prompt_builder(prompt_name)(cot_text, features)
    image = entry.get("image") if use_images else None
    # Hash the evidence actually presented to the judge, including prompt-file
    # edits and image contents. Never hash file names in place of their data.
    evidence = {
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "prompt": prompt,
        "trajectory_features": features,
        "image": image_to_data_url(image) if image is not None else None,
        "timestamp_us": entry.get("timestamp_us"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return JudgeInput(cot_text, features, prompt, image, fingerprint)


def result_is_complete(result: dict) -> bool:
    """Only successful, usable evaluations qualify for resume or completion."""
    evaluation = result.get("evaluation")
    if result.get("error") or not isinstance(evaluation, dict):
        return False
    if any(evaluation.get(key) for key in ("dry_run", "parse_error", "error")):
        return False
    for dimension in DIMENSIONS:
        data = evaluation.get(dimension)
        if not isinstance(data, dict):
            return False
        if "score" in data:
            if isinstance(data["score"], bool):
                return False
            try:
                score = float(data["score"])
            except (TypeError, ValueError):
                return False
            if not math.isfinite(score) or not 1 <= score <= 5:
                return False
        else:
            verdict = data.get("verdict")
            if not isinstance(verdict, str) or verdict.strip().lower() not in {
                "consistent",
                "inconsistent",
            }:
                return False
    return True


def process_entry(
    client,
    model_name: str,
    entry: dict,
    variant_name: str | None = None,
    prompt_name: str = "default",
    trajectory_frame: str = "ego_rig",
    lane_reference: str = "auto",
    use_images: bool = True,
    seed: int = DEFAULT_SEED,
    temperature: float | None = 0,
    extra_params: dict | None = None,
    *,
    judge_input: JudgeInput | None = None,
) -> dict:
    """Judge one extracted entry: features -> prompt -> LLM -> parsed result."""
    clip_id = entry["clip_id"]
    prepared = judge_input or prepare_judge_input(
        entry,
        prompt_name=prompt_name,
        trajectory_frame=trajectory_frame,
        lane_reference=lane_reference,
        use_images=use_images,
    )

    llm_timing = None
    raw_response = None
    image = None
    if client is not None:
        image = prepared.image
        raw_response, llm_timing = call_llm(
            client,
            model_name,
            prepared.prompt,
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
        "chain_of_thought": prepared.cot_text,
        "trajectory_features": prepared.trajectory_features,
        "evaluation": parsed,
        # Verbatim judge exchange. Parsing keeps only the scored dimensions, so
        # without these a disagreement, a refusal, or a malformed response
        # cannot be audited once the run is over.
        "judge_prompt": prepared.prompt,
        "judge_input_fingerprint": prepared.fingerprint,
        "judge_raw_response": raw_response,
        "judge_image_attached": image is not None,
        "timestamp_us": entry.get("timestamp_us"),
        "variant": variant_name,
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
        if not result_is_complete(r):
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
        if not result_is_complete(r):
            continue
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
    summary["total_evaluated"] = sum(result_is_complete(r) for r in results)
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
