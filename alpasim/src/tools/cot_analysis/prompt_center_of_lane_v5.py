# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Backwards-compatible shim for prompt auto-discovery.

The lane-center v5 prompt builder moved to
``alpasim_utils.cot_consistency.prompt_center_of_lane_v5`` so the in-loop
runtime ``ConsistencyMonitor`` can reuse it without importing ``alpasim-tools``.
The ``cot_analysis`` prompt loader discovers this file by path and calls its
``build_prompt``; re-exporting keeps ``--prompt center_of_lane_v5`` working.
"""

from alpasim_utils.cot_consistency.prompt_center_of_lane_v5 import (  # noqa: F401
    build_prompt,
)
