# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from argparse import Namespace
from pathlib import Path

from tools.run_benchmark_llm_matrix import RunSpec, _command_for_spec, _variant_args


def test_child_command_requests_llm_extra() -> None:
    args = Namespace(
        benchmark_source_root=Path("."),
        delay=1.0,
        include_unreliable_cot=False,
        log_level="INFO",
        module="cot_analysis",
    )
    spec = RunSpec(
        provider_label="qwen",
        provider="qwen35_4b_fp8",
        variant="llm",
        cot_args=("--variant", "llm"),
    )

    cmd = _command_for_spec(
        args,
        Path("subset.json"),
        Path("output.json"),
        spec,
        seed=42,
    )

    assert cmd[:4] == ["uv", "run", "--extra", "llm"]
    assert cmd[4:7] == ["python", "-m", "cot_analysis"]
    assert cmd[-2:] == ["--variant", "llm"]


def test_matrix_variants_forward_coupled_cot_analysis_variant() -> None:
    args = Namespace()

    assert _variant_args(args, "llm") == ("--variant", "llm")
    assert _variant_args(args, "f_llm_map_graph") == (
        "--variant",
        "f_llm_map_graph",
    )
