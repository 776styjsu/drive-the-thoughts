# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared CoT/trajectory consistency core.

Reusable pieces of the CoT-consistency judge that both the offline
``cot_analysis`` CLI (in ``alpasim-tools``) and the in-loop runtime
``ConsistencyMonitor`` (in ``alpasim_runtime``) depend on. It lives in
``alpasim_utils`` because ``alpasim-tools`` already depends on
``alpasim_runtime``; putting the shared core here avoids an import cycle.
"""

from .llm_judge import (
    DEFAULT_SEED,
    PROVIDERS,
    build_client,
    call_llm,
    find_and_load_dotenv,
    parse_response,
    resolve_provider,
    score_from_evaluation,
)
from .prompt_center_of_lane import build_prompt as build_center_of_lane_prompt
from .trajectory_features import compute_trajectory_features

build_center_of_lane_v5_prompt = build_center_of_lane_prompt

__all__ = [
    "DEFAULT_SEED",
    "PROVIDERS",
    "build_center_of_lane_prompt",
    "build_center_of_lane_v5_prompt",
    "build_client",
    "call_llm",
    "compute_trajectory_features",
    "find_and_load_dotenv",
    "parse_response",
    "resolve_provider",
    "score_from_evaluation",
]
