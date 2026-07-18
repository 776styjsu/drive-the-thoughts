# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Closed-loop online CoT monitoring analysis (Experiment B).

Where ``trajectory_safety`` (Experiment A) scores a single *static, open-loop*
planned trajectory against the map, this package operates on the **closed-loop
rollouts** that Alpasim already produced: for every driver decision step in a
``rollout.asl`` it recovers the ``(CoT_t, tau_t)`` pair the policy emitted, runs
the deterministic rule-based consistency monitor on it, and lines the resulting
per-step inconsistency flags up against the per-timestep simulated safety
outcomes (off-road / collision) that the Alpasim evaluator already wrote to
``metrics.parquet``.

That join answers the genuinely *downstream* RQ4 question that the static
analysis cannot: **does a CoT/trajectory inconsistency flag at step ``t``
lead a real simulated safety failure**, and by how much (lead-time
distribution)?  It also yields an offline, defensible upper bound on the
fraction of failures an at-flag fallback could have averted -- a proxy for the
closed-loop counterfactual intervention that needs a GPU re-run to measure for
real.
"""
