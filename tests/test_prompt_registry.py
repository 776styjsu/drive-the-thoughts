# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from alpasim_utils.cot_consistency import resolve_consistency_variant
from cot_analysis.prompts import discover_prompt_names, resolve_prompt_builder


def test_center_of_lane_prompt_name_resolves_with_alias() -> None:
    names = discover_prompt_names()

    assert "center_of_lane" in names
    assert "center_of_lane_v5" in names
    assert resolve_prompt_builder("center_of_lane") is resolve_prompt_builder(
        "center_of_lane_v5"
    )


def test_consistency_variant_couples_prompt_frame_and_lane_reference() -> None:
    variant = resolve_consistency_variant("center_of_lane")
    alias_variant = resolve_consistency_variant("f_llm_map_graph")

    assert variant.prompt == "center_of_lane"
    assert variant.trajectory_frame == "dual"
    assert variant.lane_reference == "map_graph"
    assert alias_variant == variant
