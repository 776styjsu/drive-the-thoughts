# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from cot_analysis.__main__ import build_parser, _resolve_variant_settings


def _parse(*args: str):
    return build_parser().parse_args(["--benchmark_json", "benchmark.json", *args])


def test_default_variant_uses_plain_prompt_and_ego_features() -> None:
    assert _resolve_variant_settings(_parse()) == (
        "llm",
        "default",
        "ego_rig",
        "auto",
    )


def test_center_of_lane_variant_couples_prompt_frame_and_lane_reference() -> None:
    assert _resolve_variant_settings(_parse("--variant", "center_of_lane")) == (
        "center_of_lane",
        "center_of_lane",
        "dual",
        "map_graph",
    )


def test_center_of_lane_prompt_alias_infers_coupled_variant() -> None:
    assert _resolve_variant_settings(_parse("--prompt", "center_of_lane")) == (
        "center_of_lane",
        "center_of_lane",
        "dual",
        "map_graph",
    )


def test_advanced_overrides_keep_variant_identity() -> None:
    assert _resolve_variant_settings(
        _parse(
            "--variant",
            "center_of_lane",
            "--trajectory_frame",
            "lane_center",
            "--lane_reference",
            "route",
        )
    ) == ("center_of_lane", "center_of_lane", "lane_center", "route")
