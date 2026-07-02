# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Accessors for benchmark entry fields.

Two on-disk annotation schemas coexist in the benchmark's history and both are
supported everywhere:

- flat: ``cot_reliable`` (bool/str), ``cot_reliability_justification``,
  ``cot_unreliability_taxonomy`` — used by ``benchmark_expanded_*.json``.
- nested: a ``cot_reliability`` object with ``reliable``, ``justification``,
  ``unreliability_categories``, ``primary_unreliability_category`` — used by
  the released ``benchmark.json``.
"""

from __future__ import annotations

import ast
from typing import Any

_TRUE_STRINGS = {"true", "reliable", "yes", "1"}
_FALSE_STRINGS = {"false", "unreliable", "no", "0"}


def cot_reliability_flag(entry: Any) -> bool | None:
    """CoT reliability from a benchmark entry, supporting both schemas.

    Returns True/False when a signal is present, else None so callers choose
    how to treat unannotated entries (see :func:`cot_is_reliable` and
    :func:`cot_is_unreliable`).
    """
    if not isinstance(entry, dict):
        return None
    nested = entry.get("cot_reliability")
    value = (
        nested.get("reliable")
        if isinstance(nested, dict) and "reliable" in nested
        else entry.get("cot_reliable")
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    return None


def cot_is_reliable(entry: Any) -> bool:
    """Strict check: the entry is explicitly annotated reliable."""
    return cot_reliability_flag(entry) is True


def cot_is_unreliable(entry: Any) -> bool:
    """Strict check: the entry is explicitly annotated unreliable."""
    return cot_reliability_flag(entry) is False


def reliability_justification(entry: dict) -> Any:
    """Reliability justification from either schema (flat or nested)."""
    nested = entry.get("cot_reliability")
    if isinstance(nested, dict) and nested.get("justification") is not None:
        return nested.get("justification")
    return entry.get("cot_reliability_justification")


def unreliability_taxonomy(entry: dict) -> Any:
    """Unreliability categories from either schema.

    For the nested schema this returns ``{"categories": [...],
    "primary_category": ...}``; for the flat schema, whatever the entry
    stored under ``cot_unreliability_taxonomy``.
    """
    nested = entry.get("cot_reliability")
    if isinstance(nested, dict) and (
        nested.get("unreliability_categories")
        or nested.get("primary_unreliability_category")
    ):
        return {
            "categories": nested.get("unreliability_categories") or [],
            "primary_category": nested.get("primary_unreliability_category"),
        }
    return entry.get("cot_unreliability_taxonomy")


def ground_truth_is_consistent(entry: dict) -> bool:
    """Read the benchmark ground-truth consistency label.

    Raises ValueError when the entry carries no recognizable label.
    """
    from .judgments import consistency_from_value

    for key in ("cot_action_consistency", "is_consistent", "consistent", "label", "result"):
        if key in entry:
            parsed = consistency_from_value(entry[key])
            if parsed is not None:
                return parsed
    raise ValueError(
        f"Benchmark entry has no consistency label: {entry.get('clip_id')}"
    )


def flatten_cot(cot: Any) -> str:
    """Recursively flatten CoT from nested lists to a single string."""
    if isinstance(cot, str):
        return cot
    if isinstance(cot, list):
        return " ".join(flatten_cot(item) for item in cot)
    return str(cot)


def clean_cot(raw: Any, separator: str = " | ") -> str:
    """Render chain_of_thought (often a stringified Python list) as text."""
    if raw is None:
        return ""
    if isinstance(raw, list):
        return separator.join(str(item) for item in raw)
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, (list, tuple)):
            return separator.join(str(item) for item in parsed)
        return str(parsed)
    return text


def nurec_scene(entry: dict) -> dict:
    """The entry's NuRec scene attribute dict (empty when absent)."""
    labels = entry.get("labels")
    if not isinstance(labels, dict):
        return {}
    scene = labels.get("nurec_scene")
    return scene if isinstance(scene, dict) else {}


def cot_decision_label(entry: dict) -> dict:
    """The entry's CoT decision label dict (empty when absent)."""
    labels = entry.get("labels")
    if not isinstance(labels, dict):
        return {}
    decision = labels.get("cot_decision_label")
    return decision if isinstance(decision, dict) else {}
