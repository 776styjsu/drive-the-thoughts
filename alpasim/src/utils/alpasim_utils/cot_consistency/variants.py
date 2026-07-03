# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Named CoT-consistency prompt/feature configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConsistencyVariant:
    """Prompt and trajectory-feature settings that must be used together."""

    name: str
    prompt: str
    trajectory_frame: str
    lane_reference: str


_CONSISTENCY_VARIANTS: dict[str, ConsistencyVariant] = {
    "llm": ConsistencyVariant(
        name="llm",
        prompt="default",
        trajectory_frame="ego_rig",
        lane_reference="auto",
    ),
    "center_of_lane": ConsistencyVariant(
        name="center_of_lane",
        prompt="center_of_lane",
        trajectory_frame="dual",
        lane_reference="map_graph",
    ),
}

_CONSISTENCY_VARIANT_ALIASES = {
    "default": "llm",
    "f_llm_map_graph": "center_of_lane",
    "center_of_lane_v5": "center_of_lane",
}

CONSISTENCY_VARIANTS = dict(_CONSISTENCY_VARIANTS)


def consistency_variant_names(include_aliases: bool = False) -> list[str]:
    """Return selectable consistency variant names."""
    names = set(_CONSISTENCY_VARIANTS)
    if include_aliases:
        names.update(_CONSISTENCY_VARIANT_ALIASES)
    return sorted(names)


def normalize_consistency_variant_name(name: str) -> str:
    """Map a public variant name or accepted alias to its canonical name."""
    if name in _CONSISTENCY_VARIANTS:
        return name
    if name in _CONSISTENCY_VARIANT_ALIASES:
        return _CONSISTENCY_VARIANT_ALIASES[name]
    available = ", ".join(consistency_variant_names(include_aliases=True))
    raise ValueError(f"Unknown consistency variant '{name}'. Available: {available}")


def resolve_consistency_variant(name: str) -> ConsistencyVariant:
    """Resolve a variant name or accepted alias to its coupled settings."""
    return _CONSISTENCY_VARIANTS[normalize_consistency_variant_name(name)]
