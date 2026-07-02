# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared analysis library for the Drive the Thoughts benchmark artifact.

Single home for the logic every analysis CLI needs when reading the released
benchmark and monitor-output JSON files:

- :mod:`benchmark_analysis.loading` — JSON loading and per-clip entry
  extraction for the handful of on-disk container shapes.
- :mod:`benchmark_analysis.schema` — accessors for benchmark entry fields
  (CoT reliability, ground-truth consistency, scene labels, CoT text).
- :mod:`benchmark_analysis.judgments` — normalization of monitor predictions
  (LLM judge scores/verdicts, rule-based reports) into binary judgments.
- :mod:`benchmark_analysis.metrics` — binary confusion counts and the derived
  classification metrics used in the paper (F1, balanced accuracy, kappa).

The scripts in ``tools/`` are thin CLIs over this package.
"""

from .judgments import (
    CONSISTENCY_TYPES,
    DEFAULT_SCORE_THRESHOLD,
    INVALID_PARSE_LABEL,
    Judgment,
    consistency_from_value,
    judgment_for,
    llm_judgment,
    rule_judgment,
)
from .loading import extract_entries, index_by_clip_id, load_json
from .metrics import BinaryConfusion, classification_metrics
from .schema import (
    clean_cot,
    cot_is_reliable,
    cot_is_unreliable,
    cot_reliability_flag,
    flatten_cot,
    ground_truth_is_consistent,
    nurec_scene,
    reliability_justification,
    unreliability_taxonomy,
)

__all__ = [
    "BinaryConfusion",
    "CONSISTENCY_TYPES",
    "DEFAULT_SCORE_THRESHOLD",
    "INVALID_PARSE_LABEL",
    "Judgment",
    "classification_metrics",
    "clean_cot",
    "consistency_from_value",
    "cot_is_reliable",
    "cot_is_unreliable",
    "cot_reliability_flag",
    "extract_entries",
    "flatten_cot",
    "ground_truth_is_consistent",
    "index_by_clip_id",
    "judgment_for",
    "llm_judgment",
    "load_json",
    "nurec_scene",
    "reliability_justification",
    "rule_judgment",
    "unreliability_taxonomy",
]
