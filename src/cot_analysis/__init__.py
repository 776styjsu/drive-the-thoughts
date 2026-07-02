# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""CoT/trajectory consistency judging over the released benchmark.

- ``python -m cot_analysis`` — the LLM-judge pipeline (see ``__main__``).
- ``python -m cot_analysis.consistency_check`` — the deterministic rule-based
  counterpart.
- :mod:`cot_analysis.prompts` — prompt template registry.
- :mod:`cot_analysis.benchmark_source` — benchmark entry loading.
- :mod:`cot_analysis.pipeline` — per-entry judging and aggregation.
"""
