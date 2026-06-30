# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Geometric safety outcomes for planned ego trajectories (Experiment A).

Replaces the static human safety annotation of a planned trajectory with an
objective, reproducible *simulated* outcome (road departure / collision proxy)
computed from the per-frame map geometry the benchmark already ships.
"""
