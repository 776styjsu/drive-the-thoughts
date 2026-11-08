# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from argparse import Namespace
from pathlib import Path

import pytest

from tools.run_benchmark_llm_matrix import (
    RunSpec,
    _build_specs,
    _command_for_spec,
    _variant_args,
)


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


def test_kimi_endpoint_overrides_reach_only_kimi_runs() -> None:
    args = Namespace(
        benchmark_source_root=Path("."),
        delay=0.0,
        include_unreliable_cot=False,
        log_level="INFO",
        module="cot_analysis",
        kimi_model="Kimi K2.5",
        kimi_base_url="https://gateway.example.org/api",
    )
    kimi = RunSpec("kimi", "kimi", "llm", ("--variant", "llm"))
    qwen = RunSpec("qwen", "qwen35_4b_fp8", "llm", ("--variant", "llm"))

    kimi_cmd = _command_for_spec(args, Path("s.json"), Path("o.json"), kimi, 42)
    qwen_cmd = _command_for_spec(args, Path("s.json"), Path("o.json"), qwen, 42)

    assert kimi_cmd[kimi_cmd.index("--model") + 1] == "Kimi K2.5"
    assert kimi_cmd[kimi_cmd.index("--base_url") + 1] == args.kimi_base_url
    assert kimi_cmd[-2:] == ["--variant", "llm"]
    assert "--model" not in qwen_cmd and "--base_url" not in qwen_cmd


def test_deepseek_overrides_reach_only_deepseek_runs() -> None:
    args = Namespace(
        benchmark_source_root=Path("."),
        delay=0.0,
        include_unreliable_cot=False,
        log_level="INFO",
        module="cot_analysis",
        deepseek_model="deepseek-v4.1-flash",
        deepseek_base_url="https://api.deepseek.com",
    )
    deepseek = RunSpec("deepseek", "deepseek", "llm", ("--variant", "llm"))
    kimi = RunSpec("kimi", "kimi", "llm", ("--variant", "llm"))

    deepseek_cmd = _command_for_spec(args, Path("s.json"), Path("o.json"), deepseek, 42)
    kimi_cmd = _command_for_spec(args, Path("s.json"), Path("o.json"), kimi, 42)

    assert deepseek_cmd[deepseek_cmd.index("--provider") + 1] == "deepseek"
    assert deepseek_cmd[deepseek_cmd.index("--model") + 1] == "deepseek-v4.1-flash"
    assert deepseek_cmd[deepseek_cmd.index("--base_url") + 1] == args.deepseek_base_url
    assert "--model" not in kimi_cmd and "--base_url" not in kimi_cmd


def test_provider_override_swaps_only_the_named_label() -> None:
    args = Namespace(
        providers="qwen,kimi",
        variants="llm",
        provider_override=["kimi=kimi_recommended"],
    )

    specs = _build_specs(args)

    assert [(s.provider_label, s.provider) for s in specs] == [
        ("qwen", "qwen35_4b_fp8"),
        ("kimi", "kimi_recommended"),
    ]


@pytest.mark.parametrize("override", ["kimi", "nope=kimi", "kimi=nope"])
def test_provider_override_rejects_unknown_labels_and_providers(override: str) -> None:
    args = Namespace(providers="kimi", variants="llm", provider_override=[override])

    with pytest.raises(ValueError):
        _build_specs(args)
