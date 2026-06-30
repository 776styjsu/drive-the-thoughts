# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Backwards-compatible shim.

The trajectory feature computation moved to
``alpasim_utils.cot_consistency.trajectory_features`` so the in-loop runtime
``ConsistencyMonitor`` can reuse it without importing ``alpasim-tools`` (which
would create a dependency cycle). This module re-exports the public API so
``from cot_analysis.trajectory import compute_trajectory_features`` keeps working.
"""

from alpasim_utils.cot_consistency.trajectory_features import *  # noqa: F401,F403
from alpasim_utils.cot_consistency.trajectory_features import (  # noqa: F401
    compute_trajectory_features,
)
