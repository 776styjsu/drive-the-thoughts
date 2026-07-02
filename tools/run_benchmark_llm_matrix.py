#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Run benchmark CoT judge matrix across providers and feature variants.

Runs Qwen (local vLLM), Kimi (Moonshot API), and GPT (OpenAI API) for:
- llm: default prompt.py with ego-frame trajectory features.
- f_llm_map_graph: prompt center_of_lane_v5 with dual/map_graph features.

API keys come from --kimi-api-key/--gpt-api-key/--qwen-api-key, falling back
to each provider's environment variable (also loaded from .env by
cot_analysis). Keys given as arguments reach the runs through the subprocess
environment, never through the command line, so they do not appear in logs or
the manifest.

Each run delegates to ``cot_analysis`` with ``--resume`` enabled and writes a
separate JSON result plus a log file under the output directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alpasim_utils.cot_consistency import PROVIDERS as JUDGE_PROVIDERS
from benchmark_analysis import cot_reliability_flag, extract_entries, load_json


PROVIDERS: tuple[tuple[str, str], ...] = (
    ("qwen", "qwen35_4b_fp8"),
    ("kimi", "kimi"),
    ("gpt", "openai"),
)


@dataclass(frozen=True)
class RunSpec:
    provider_label: str
    provider: str
    variant: str
    cot_args: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run qwen/kimi/gpt CoT judge experiments for plain LLM and "
            "feature-LLM map_graph variants."
        )
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=repo_root / "data" / "benchmark" / "benchmark.json",
        help="Input benchmark JSON. Default: %(default)s",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=3,
        help="Number of independent repeats over the selected benchmark subset. Default: %(default)s",
    )
    parser.add_argument(
        "--limit-entries",
        "--first-entries",
        dest="limit_entries",
        type=int,
        default=None,
        help=(
            "Limit the selected benchmark subset to the first K entries. "
            "By default all reliable entries are selected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "runs" / "llm_matrix",
        help=(
            "Directory for subset, outputs, logs, and manifest (kept separate "
            "from the released data/results). Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--benchmark-source-root",
        type=Path,
        default=repo_root,
        help="Root for resolving benchmark source_scene_file paths. Default: %(default)s",
    )
    parser.add_argument(
        "--prompt",
        default="default",
        help="Prompt for the plain LLM variant. Default: %(default)s",
    )
    parser.add_argument(
        "--f-llm-prompt",
        default="center_of_lane_v5",
        help="Prompt for the f-LLM map_graph variant. Default: %(default)s",
    )
    parser.add_argument(
        "--module",
        default="cot_analysis",
        help="Python module to run. Default: %(default)s",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls, forwarded to cot_analysis. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base seed forwarded to cot_analysis. Repeats use seed+i unless "
            "--fixed-seed is set. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="Use the same seed for every repeat instead of seed+i.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level forwarded to cot_analysis. Default: %(default)s",
    )
    parser.add_argument(
        "--uv-cache-dir",
        type=Path,
        default=repo_root / ".uv-cache",
        help="Writable uv cache directory. Default: %(default)s",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=repo_root / ".hf-cache",
        help="Hugging Face cache directory for local Qwen/vLLM state. Default: %(default)s",
    )
    parser.add_argument(
        "--kimi-api-key",
        default=None,
        help=(
            "API key for the kimi provider. Overrides the MOONSHOT_API_KEY "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--gpt-api-key",
        default=None,
        help=(
            "API key for the gpt provider. Overrides the OPENAI_API_KEY "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--qwen-api-key",
        default=None,
        help=(
            "API key for the qwen provider. Overrides the QWEN35_API_KEY "
            "environment variable; the local vLLM default is EMPTY, so this "
            "is only needed for a secured server."
        ),
    )
    parser.add_argument(
        "--providers",
        default="qwen,kimi,gpt",
        help=(
            "Comma-separated provider labels to run. Choices: qwen,kimi,gpt. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--variants",
        default="llm,f_llm_map_graph",
        help=(
            "Comma-separated variants to run. Choices: llm,f_llm_map_graph. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--include-unreliable-cot",
        action="store_true",
        help=(
            "Select from all benchmark entries and do not pass "
            "--skip-unreliable-cot. By default the subset contains all reliable "
            "entries and cot_analysis still receives --skip-unreliable-cot."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of cot_analysis subprocesses to run concurrently. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the subset/manifest and print commands without executing them.",
    )
    return parser.parse_args()


def _load_benchmark_items(path: Path) -> tuple[object, list[dict]]:
    payload = load_json(path)
    return payload, extract_entries(payload)


def _cot_is_reliable(item: dict) -> bool:
    # Absent/unknown reliability is treated as reliable (do not drop the entry).
    return cot_reliability_flag(item) is not False


def _write_subset(
    benchmark_path: Path,
    output_dir: Path,
    limit_entries: int | None,
    include_unreliable: bool,
) -> tuple[Path, int, int, int]:
    if limit_entries is not None and limit_entries <= 0:
        raise ValueError("--limit-entries must be positive")

    payload, items = _load_benchmark_items(benchmark_path)
    candidates = (
        items
        if include_unreliable
        else [item for item in items if _cot_is_reliable(item)]
    )
    if limit_entries is None:
        subset = candidates
        subset_name = f"{benchmark_path.stem}.all"
    else:
        subset = candidates[:limit_entries]
        if len(subset) < limit_entries:
            raise ValueError(
                f"Requested {limit_entries} entries but only found "
                f"{len(subset)} matching entries"
            )
        subset_name = f"{benchmark_path.stem}.first_{limit_entries}"
    subset_name += "_raw" if include_unreliable else "_reliable"
    subset_path = output_dir / f"{subset_name}.json"

    if isinstance(payload, dict):
        subset_payload = dict(payload)
        if "results" in subset_payload:
            subset_payload["results"] = subset
        elif "entries" in subset_payload:
            subset_payload["entries"] = subset
        else:
            subset_payload["results"] = subset
    else:
        subset_payload = subset

    with subset_path.open("w", encoding="utf-8") as f:
        json.dump(subset_payload, f, indent=2)
        f.write("\n")

    return subset_path, len(items), len(candidates), len(subset)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _selected_providers(value: str) -> list[tuple[str, str]]:
    requested = _split_csv(value)
    known = dict(PROVIDERS)
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unknown provider label(s): {', '.join(unknown)}")
    return [(label, known[label]) for label in requested]


def _variant_args(args: argparse.Namespace, variant: str) -> tuple[str, ...]:
    if variant == "llm":
        return ("--prompt", args.prompt)
    if variant == "f_llm_map_graph":
        return (
            "--prompt",
            args.f_llm_prompt,
            "--trajectory_frame",
            "dual",
            "--lane_reference",
            "map_graph",
        )
    raise ValueError(f"Unknown variant: {variant}")


def _build_specs(args: argparse.Namespace) -> list[RunSpec]:
    variants = _split_csv(args.variants)
    known_variants = {"llm", "f_llm_map_graph"}
    unknown = sorted(set(variants) - known_variants)
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")

    specs = []
    for provider_label, provider in _selected_providers(args.providers):
        for variant in variants:
            specs.append(
                RunSpec(
                    provider_label=provider_label,
                    provider=provider,
                    variant=variant,
                    cot_args=_variant_args(args, variant),
                )
            )
    return specs


def _command_for_spec(
    args: argparse.Namespace,
    subset_path: Path,
    output_path: Path,
    spec: RunSpec,
    seed: int,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        args.module,
        "--benchmark_json",
        str(subset_path),
        "--benchmark_source_root",
        str(args.benchmark_source_root),
        "--output",
        str(output_path),
        "--provider",
        spec.provider,
        "--delay",
        str(args.delay),
        "--seed",
        str(seed),
        "--log_level",
        args.log_level,
        "--resume",
    ]
    if not args.include_unreliable_cot:
        cmd.append("--skip-unreliable-cot")
    cmd.extend(spec.cot_args)
    return cmd


def _run_with_log(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
    *,
    stream_to_console: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"start_utc: {start.isoformat()}\n")
        log.write(f"cwd: {Path.cwd()}\n")
        log.write(f"cmd: {shlex.join(cmd)}\n")
        log.write("=" * 100 + "\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if stream_to_console:
                print(line, end="")
            log.write(line)
        return_code = process.wait()

        end = datetime.now(timezone.utc)
        log.write("=" * 100 + "\n")
        log.write(f"end_utc: {end.isoformat()}\n")
        log.write(f"return_code: {return_code}\n")
        log.flush()
        return return_code


def main() -> int:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")

    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    args.uv_cache_dir.mkdir(parents=True, exist_ok=True)
    args.hf_home.mkdir(parents=True, exist_ok=True)

    subset_path, total_items, candidate_items, selected_items = _write_subset(
        args.benchmark_json,
        output_dir,
        args.limit_entries,
        args.include_unreliable_cot,
    )
    specs = _build_specs(args)

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(args.uv_cache_dir.resolve())
    env["HF_HOME"] = str(args.hf_home.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("QWEN35_BASE_URL", "http://localhost:8000/v1")
    env.setdefault("QWEN35_API_KEY", "EMPTY")
    env.pop("VIRTUAL_ENV", None)
    # Keys given as arguments travel via the environment, not the command
    # line, so they stay out of the printed commands, logs, and manifest.
    for label, provider in PROVIDERS:
        api_key = getattr(args, f"{label}_api_key")
        if api_key:
            env[JUDGE_PROVIDERS[provider]["api_key_env"]] = api_key

    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_json": str(args.benchmark_json),
        "benchmark_source_root": str(args.benchmark_source_root),
        "subset_path": str(subset_path),
        "repeats": args.n,
        "limit_entries": args.limit_entries,
        "total_benchmark_items": total_items,
        "candidate_items": candidate_items,
        "selected_items": selected_items,
        "include_unreliable_cot": args.include_unreliable_cot,
        "base_seed": args.seed,
        "fixed_seed": args.fixed_seed,
        "jobs": args.jobs,
        "runs": [],
    }

    planned_runs = []
    for repeat_index in range(1, args.n + 1):
        seed = args.seed if args.fixed_seed else args.seed + repeat_index - 1
        for spec in specs:
            run_name = (
                f"{spec.provider_label}.{spec.variant}.run_{repeat_index:03d}"
            )
            output_path = output_dir / f"{run_name}.json"
            log_path = logs_dir / f"{run_name}.log"
            cmd = _command_for_spec(args, subset_path, output_path, spec, seed)

            run_record = {
                "name": run_name,
                "repeat_index": repeat_index,
                "seed": seed,
                "provider_label": spec.provider_label,
                "provider": spec.provider,
                "variant": spec.variant,
                "output": str(output_path),
                "log": str(log_path),
                "command": cmd,
            }
            manifest["runs"].append(run_record)
            planned_runs.append(
                {
                    "name": run_name,
                    "output_path": output_path,
                    "log_path": log_path,
                    "command": cmd,
                    "record": run_record,
                }
            )

    failures: list[tuple[str, int]] = []
    for planned in planned_runs:
        print(f"\n[{planned['name']}] output={planned['output_path']}")
        print(f"[{planned['name']}] log={planned['log_path']}")
        print(f"[{planned['name']}] cmd={shlex.join(planned['command'])}")
        if args.dry_run:
            planned["record"]["return_code"] = None

    if not args.dry_run and args.jobs == 1:
        for planned in planned_runs:
            return_code = _run_with_log(
                planned["command"],
                planned["log_path"],
                env,
                stream_to_console=True,
            )
            planned["record"]["return_code"] = return_code
            if return_code != 0:
                failures.append((planned["name"], return_code))
    elif not args.dry_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    _run_with_log,
                    planned["command"],
                    planned["log_path"],
                    env,
                    stream_to_console=False,
                ): planned
                for planned in planned_runs
            }
            for future in concurrent.futures.as_completed(futures):
                planned = futures[future]
                return_code = future.result()
                planned["record"]["return_code"] = return_code
                print(
                    f"[{planned['name']}] completed with return_code={return_code}; "
                    f"log={planned['log_path']}"
                )
                if return_code != 0:
                    failures.append((planned["name"], return_code))

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nManifest: {manifest_path}")
    if failures:
        print("Failed runs:")
        for run_name, return_code in failures:
            print(f"  {run_name}: {return_code}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
