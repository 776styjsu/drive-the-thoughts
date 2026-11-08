# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Select cached trajectory features only when their coordinate frames match."""


def cached_features_for_frame(
    cached: dict | None, trajectory_frame: str, lane_reference: str
) -> dict | None:
    if not isinstance(cached, dict) or cached.get("error"):
        return None
    stats = cached.get("summary_stats")
    if not isinstance(stats, dict):
        return None
    if trajectory_frame == "ego_rig":
        if stats.get("reference_frame") == "dual":
            return cached_features_for_frame(
                cached.get("ego_features"), trajectory_frame, lane_reference
            )
        return cached if stats.get("reference_frame") == "ego_rig" else None
    reference_matches = stats.get("lane_reference") == lane_reference or (
        lane_reference == "auto"
        and stats.get("lane_reference") in {"auto", "map_graph", "route"}
    )
    if stats.get("reference_frame") == trajectory_frame and reference_matches:
        return cached
    return None
