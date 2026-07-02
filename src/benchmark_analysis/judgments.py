# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Normalization of monitor predictions into binary consistency judgments.

Two monitor output families are supported:

- LLM judge results (``cot_output_alignment``): either the detector schema
  ``{"verdict": "consistent" | "inconsistent"}`` or the graded schema
  ``{"score": 1-5}`` where ``score > threshold`` counts as consistent.
- Rule-based results: a ``report`` object (or the entry itself) with a
  ``label``/``result`` string or a binary ``score``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Strings (lowercased) that read as a "consistent" verdict.
CONSISTENT_STRINGS = frozenset(
    {
        "consistent",
        "true",
        "pass",
        "passed",
        "match",
        "matched",
        "aligned",
        "yes",
    }
)

#: Strings (lowercased) that read as an "inconsistent" verdict.
INCONSISTENT_STRINGS = frozenset(
    {
        "inconsistent",
        "contradictory",
        "invalid_parse",
        "false",
        "fail",
        "failed",
        "mismatch",
        "mismatched",
        "not_consistent",
        "not consistent",
        "partial",
        "partial_consistency",
        "partially_consistent",
        "partially consistent",
        "unaligned",
        "no",
    }
)

#: Rule-based label emitted when the CoT yields no parseable ego intent.
#: These are parse failures rather than genuine consistency judgments.
INVALID_PARSE_LABEL = "invalid_parse"

#: LLM score cutoff: ``score > threshold`` classifies as consistent.
DEFAULT_SCORE_THRESHOLD = 2.0

# Recognized --consistency-type values and the rule-based aliases among them.
COT_OUTPUT_ALIGNMENT = "cot_output_alignment"
RULE_BASED_ALIASES = frozenset(
    {"alpasim_cot_consistency", "alpasim_cot_consistency_report", "rule_based"}
)
CONSISTENCY_TYPES = (COT_OUTPUT_ALIGNMENT, *sorted(RULE_BASED_ALIASES))


def consistency_from_value(value: Any) -> bool | None:
    """Interpret a bool/number/verdict-string as a binary consistency signal."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in CONSISTENT_STRINGS:
            return True
        if normalized in INCONSISTENT_STRINGS:
            return False
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Judgment:
    """A monitor's normalized binary prediction for one clip."""

    is_consistent: bool
    label: str
    score: float | None = None
    category: str = ""
    justification: str = ""
    #: False for rule-based invalid_parse results (the parser could not judge).
    valid_parse: bool = True


def _label_from_bool(is_consistent: bool) -> str:
    return "consistent" if is_consistent else "inconsistent"


def alignment_payload(entry: dict) -> dict:
    """The ``cot_output_alignment`` dict from a judge result entry."""
    alignment = entry.get(COT_OUTPUT_ALIGNMENT)
    if not isinstance(alignment, dict):
        evaluation = entry.get("evaluation")
        alignment = (
            evaluation.get(COT_OUTPUT_ALIGNMENT) if isinstance(evaluation, dict) else None
        )
    return alignment if isinstance(alignment, dict) else {}


def llm_judgment(
    entry: dict, score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> Judgment:
    """Normalize an LLM judge result entry.

    Uses the verdict field when present, else the graded score against
    ``score_threshold``. Raises ValueError when neither signal exists.
    """
    alignment = alignment_payload(entry)
    justification = alignment.get("justification", entry.get("justification", ""))
    category = (
        alignment.get("inconsistency_type")
        or alignment.get("category", entry.get("category", ""))
        or ""
    )

    is_consistent = consistency_from_value(alignment.get("verdict"))
    score = _to_float(alignment.get("score"))
    if score is None:
        score = _to_float(entry.get("score"))

    if is_consistent is None:
        if score is None:
            raise ValueError(
                "Missing cot_output_alignment verdict/score for "
                f"clip_id={entry.get('clip_id')}"
            )
        is_consistent = score > score_threshold

    return Judgment(
        is_consistent=is_consistent,
        label=_label_from_bool(is_consistent),
        score=score,
        category=category,
        justification=justification,
    )


def rule_judgment(entry: dict) -> Judgment:
    """Normalize a rule-based consistency result entry.

    Reads the ``report`` object (or the entry itself) and classifies from its
    ``label``/``result`` string, falling back to the binary ``score``. Raises
    ValueError when no signal exists.
    """
    report = entry.get("report") if isinstance(entry.get("report"), dict) else entry

    label = report.get("label")
    result = report.get("result")
    score = _to_float(report.get("score"))

    is_consistent = consistency_from_value(label)
    if is_consistent is None:
        is_consistent = consistency_from_value(result)
    if is_consistent is None and score is not None:
        is_consistent = score > 0
    if is_consistent is None:
        raise ValueError(
            f"Missing rule-based result/label/score for clip_id={entry.get('clip_id')}"
        )

    raw_label = str(label or result or "").strip().lower()
    return Judgment(
        is_consistent=is_consistent,
        label=label or result or _label_from_bool(is_consistent),
        score=score,
        category=label or result or "",
        justification="",
        valid_parse=raw_label != INVALID_PARSE_LABEL,
    )


def judgment_for(
    entry: dict,
    consistency_type: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> Judgment:
    """Dispatch to the extractor matching a ``--consistency-type`` value."""
    if consistency_type == COT_OUTPUT_ALIGNMENT:
        return llm_judgment(entry, score_threshold)
    if consistency_type in RULE_BASED_ALIASES:
        return rule_judgment(entry)
    raise ValueError(f"Unsupported consistency type: {consistency_type}")
