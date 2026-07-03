# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""CoT-consistency judging over the released benchmark.

Loads benchmark entries (CoT text plus per-clip trajectory/lane geometry from
the artifact's scene directories), computes trajectory features, and evaluates
CoT/trajectory self-consistency with an LLM judge via an OpenAI-compatible
API. Produces structured per-dimension scores.

This artifact tool is benchmark-driven; extracting entries directly from
AlpaSim ASL rollout logs requires the full simulator workspace (see the pinned
``alpasim/`` tree).

Model backends are selectable with --provider (see
``alpasim_utils.cot_consistency.llm_judge.PROVIDERS``):
    kimi          - Kimi K2.5 via the Moonshot API (key: MOONSHOT_API_KEY)
    openai        - GPT-5.5 with high reasoning effort (key: OPENAI_API_KEY)
    qwen3_4b_fp8  - Qwen3-4B-FP8 via a local vLLM server
    qwen35_4b_fp8 - Qwen3.5-4B-FP8 via a local vLLM server

Keys come from --api_key or the matching environment variable, also loaded
from a .env file at/above the working directory. Any other OpenAI-compatible
host works through --base_url/--model overrides (e.g. OpenRouter with
``--provider kimi --base_url https://openrouter.ai/api/v1 --model
moonshotai/kimi-k2.5``). Backends decode deterministically (temperature=0
plus a fixed --seed). Without a key the tool runs in dry-run mode (trajectory
analysis only).

Usage:
    # Dry run (no API key, verifies trajectory analysis):
    uv run python -m cot_analysis \
        --benchmark_json data/benchmark/benchmark.json \
        --output /tmp/cot_dry.json

    # Full run with GPT-5.5 high on the lane-relative feature variant:
    uv run --extra llm python -m cot_analysis --provider openai \
        --benchmark_json data/benchmark/benchmark.json \
        --variant center_of_lane \
        --output cot_consistency_gpt55.json

    # Full run with local Qwen3.5-4B FP8 (serve it first):
    tools/serve_qwen_vllm.sh qwen35 serve
    uv run --extra llm python -m cot_analysis --provider qwen35_4b_fp8 \
        --benchmark_json data/benchmark/benchmark.json \
        --output cot_consistency_qwen35_4b_fp8.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

from alpasim_utils.cot_consistency import (
    DEFAULT_SEED,
    PROVIDERS,
    build_client,
    consistency_variant_names,
    normalize_consistency_variant_name,
    resolve_provider,
    resolve_consistency_variant,
)
from benchmark_analysis import flatten_cot

from .benchmark_source import (
    extract_entries_from_benchmark,
    filter_unreliable_benchmark_cots,
)
from .pipeline import (
    DIMENSIONS,
    aggregate_results,
    build_output_data,
    log_evaluation_summary,
    log_feature_summary,
    process_entry,
)
from .prompts import discover_prompt_names, resolve_prompt_builder

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CoT consistency judging over the released benchmark"
    )
    parser.add_argument(
        "--benchmark_json",
        type=str,
        required=True,
        help=(
            "Benchmark JSON with clip_id/source_scene_file entries. Reuses "
            "trajectory_features from the referenced cot_consistency reports."
        ),
    )
    parser.add_argument(
        "--benchmark_source_root",
        type=str,
        default=".",
        help=(
            "Root for resolving benchmark source_scene_file paths "
            "(default: current working directory)."
        ),
    )
    parser.add_argument(
        "--skip_unreliable_cot",
        "--skip-unreliable-cot",
        action="store_true",
        help=(
            "Skip entries whose benchmark annotation marks the CoT unreliable "
            "(flat cot_reliable=false or nested cot_reliability.reliable=false)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cot_consistency.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=sorted(PROVIDERS),
        default="kimi",
        help=(
            "Model backend. Determines the default model, API key env var, "
            "base URL, and provider-specific request parameters."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (default: the selected provider's model)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (defaults to the provider's key env var from .env/environment)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL override (defaults to the provider's base URL)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Fixed decoding seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (for rate-limited endpoints)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Coupled prompt/feature configuration. Public choices: %s. "
            "Legacy aliases are also accepted."
            % ", ".join(consistency_variant_names())
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Advanced override for --variant: registered prompt name (%s), "
            "or a path to a .py file defining build_prompt(cot_text, "
            "traj_features)."
            % ", ".join(discover_prompt_names())
        ),
    )
    parser.add_argument(
        "--trajectory_frame",
        type=str,
        choices=["ego_rig", "lane_center", "dual"],
        default=None,
        help=(
            "Advanced override for --variant: feature frame for the predicted "
            "trajectory."
        ),
    )
    parser.add_argument(
        "--lane_reference",
        type=str,
        choices=["auto", "map_graph", "map_graph_same_lane", "route"],
        default=None,
        help=(
            "Advanced override for --variant: lane-center source when "
            "lane-center features are computed."
        ),
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip sending images to the model (text-only mode)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed entries",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser


def _infer_variant_name(args: argparse.Namespace) -> str:
    """Choose a coupled variant, accepting prompt-based calls as aliases."""
    if args.variant:
        return normalize_consistency_variant_name(args.variant)
    if args.prompt in {"center_of_lane", "center_of_lane_v5"}:
        return "center_of_lane"
    return "llm"


def _resolve_variant_settings(args: argparse.Namespace) -> tuple[str, str, str, str]:
    """Resolve public variant plus optional advanced overrides."""
    variant = resolve_consistency_variant(_infer_variant_name(args))
    prompt = args.prompt or variant.prompt
    trajectory_frame = args.trajectory_frame or variant.trajectory_frame
    lane_reference = args.lane_reference or variant.lane_reference
    return variant.name, prompt, trajectory_frame, lane_reference


def load_resumable_results(
    output_path: Path, all_entries: list[dict]
) -> tuple[list[dict], set[str]]:
    """Load prior results for --resume, keeping only clean, in-scope entries."""
    try:
        with output_path.open("r") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Output file invalid. Starting fresh.")
        return [], set()

    loaded_results = saved.get("results", saved) if isinstance(saved, dict) else saved
    if isinstance(loaded_results, dict):
        loaded_results = [loaded_results]
    elif not isinstance(loaded_results, list):
        loaded_results = []

    allowed_ids = {entry["clip_id"] for entry in all_entries}
    current_results = [
        r
        for r in loaded_results
        if isinstance(r, dict) and r.get("clip_id") in allowed_ids
    ]
    results = [
        r
        for r in current_results
        if not r.get("error") and not r.get("evaluation", {}).get("error")
    ]
    completed_ids = {r["clip_id"] for r in results}

    logger.info("Resuming: %d entries already completed", len(completed_ids))
    retry_results = len(current_results) - len(results)
    if retry_results:
        logger.info("Retrying %d saved result(s) with prior errors", retry_results)
    dropped_results = len(loaded_results) - len(current_results)
    if dropped_results:
        logger.info(
            "Ignored %d saved result(s) outside the current input set",
            dropped_results,
        )
    return results, completed_ids


def main():
    args = build_parser().parse_args()

    try:
        (
            args.variant,
            args.prompt,
            args.trajectory_frame,
            args.lane_reference,
        ) = _resolve_variant_settings(args)
    except ValueError as exc:
        raise SystemExit(str(exc))

    # Resolve (and cache) the prompt builder up front so a bad --prompt fails
    # immediately instead of erroring on every entry.
    try:
        resolve_prompt_builder(args.prompt)
    except (ValueError, FileNotFoundError, AttributeError) as exc:
        raise SystemExit(str(exc))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --- Setup model client ---
    provider = resolve_provider(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    client = None
    if provider["api_key"]:
        client = build_client(provider["api_key"], provider["base_url"])
        logger.info(
            "Using %s — model: %s (base_url=%s, seed=%d, temperature=%s)",
            provider["label"],
            provider["model"],
            provider["base_url"] or "<openai-default>",
            args.seed,
            "default" if provider["temperature"] is None else provider["temperature"],
        )
        if provider["extra_params"]:
            logger.info("Extra request params: %s", provider["extra_params"])
    else:
        logger.info(
            "No API key (set $%s or --api_key). Running in DRY-RUN mode "
            "(trajectory analysis only).",
            PROVIDERS[args.provider]["api_key_env"],
        )
    logger.info(
        "Using variant=%s, prompt=%s, trajectory_frame=%s, lane_reference=%s",
        args.variant,
        args.prompt,
        args.trajectory_frame,
        args.lane_reference,
    )
    if args.prompt == "default" and args.trajectory_frame != "ego_rig":
        logger.warning(
            "%s features are intended for --variant center_of_lane; the default "
            "prompt describes ego-frame x/y coordinates.",
            args.trajectory_frame,
        )

    # --- Extract entries ---
    all_entries = extract_entries_from_benchmark(
        args.benchmark_json,
        source_root=args.benchmark_source_root,
        load_raw_trajectory=args.trajectory_frame in {"lane_center", "dual"},
    )
    original_entry_count = len(all_entries)
    skipped_unreliable_cot_entries = []
    if args.skip_unreliable_cot:
        all_entries, skipped_unreliable_cot_entries = filter_unreliable_benchmark_cots(
            all_entries
        )
        logger.info(
            "Skipped %d unreliable benchmark CoT entr%s",
            len(skipped_unreliable_cot_entries),
            "y" if len(skipped_unreliable_cot_entries) == 1 else "ies",
        )

    logger.info("Total entries to analyze: %d", len(all_entries))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(results: list[dict]) -> dict:
        output_data = build_output_data(
            results,
            aggregate_results(results),
            entries_to_analyze=len(all_entries),
            original_entry_count=original_entry_count,
            skipped_unreliable_cot_entries=skipped_unreliable_cot_entries,
        )
        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2, default=str)
        return output_data["summary"]

    results: list[dict] = []
    completed_ids: set[str] = set()

    if not all_entries:
        logger.warning("No entries extracted. Check the benchmark file and settings.")
        save(results)
        print(f"Results saved to: {output_path}")
        return

    if args.resume and output_path.exists():
        results, completed_ids = load_resumable_results(output_path, all_entries)

    # --- Process entries ---
    use_images = not args.no_images
    if use_images and not provider["supports_images"]:
        logger.info(
            "%s is text-only for this workflow; disabling image inputs.",
            provider["label"],
        )
        use_images = False

    summary: dict = {}
    for idx, entry in enumerate(all_entries):
        clip_id = entry["clip_id"]
        if clip_id in completed_ids:
            continue

        if entry.get("error"):
            logger.error(
                "[%d/%d] %s — %s", idx + 1, len(all_entries), clip_id, entry["error"]
            )
            results.append(
                {
                    "clip_id": clip_id,
                    "error": entry["error"],
                    "benchmark": entry.get("benchmark"),
                    "source_report": entry.get("source_report"),
                }
            )
            continue

        logger.info(
            "[%d/%d] %s — CoT: %s",
            idx + 1,
            len(all_entries),
            clip_id,
            flatten_cot(entry["chain_of_thought"])[:80],
        )

        try:
            result = process_entry(
                client,
                provider["model"],
                entry,
                variant_name=args.variant,
                prompt_name=args.prompt,
                trajectory_frame=args.trajectory_frame,
                lane_reference=args.lane_reference,
                use_images=use_images,
                seed=args.seed,
                temperature=provider["temperature"],
                extra_params=provider["extra_params"],
            )
            results.append(result)
            log_feature_summary(result["trajectory_features"].get("summary_stats", {}))
            log_evaluation_summary(result["evaluation"])
        except Exception as exc:
            logger.error("  ERROR: %s", exc)
            results.append({"clip_id": clip_id, "error": str(exc)})

        summary = save(results)  # incremental save after every entry

        if args.delay > 0 and client is not None:
            time.sleep(args.delay)

    summary = save(results)

    # --- Final summary ---
    print(f"\n{'=' * 50}")
    print(f"Results saved to: {output_path}")
    if skipped_unreliable_cot_entries:
        print(
            f"Skipped unreliable benchmark CoTs: {len(skipped_unreliable_cot_entries)}"
        )
    print(f"Total evaluated: {summary.get('total_evaluated', 0)}/{len(all_entries)}")
    for dim in DIMENSIONS:
        d = summary.get(dim, {})
        if "mean" in d:
            print(f"  {dim}: {d['mean']:.2f} ± {d['std']:.2f} (n={d['count']})")
        elif d.get("count", 0) > 0:
            print(
                f"  {dim}: {d.get('inconsistent', 0)} inconsistent / "
                f"{d.get('consistent', 0)} consistent (n={d['count']})"
            )
    flagged = summary.get("flagged_entries", [])
    if flagged:
        print(f"\n⚠ {len(flagged)} inconsistency flags:")
        for f_item in flagged[:10]:
            if f_item.get("score") is not None:
                label = f"{f_item['score']}/5"
            else:
                label = f_item.get("verdict", "inconsistent")
                if f_item.get("inconsistency_type"):
                    label += f" ({f_item['inconsistency_type']})"
            print(f"  {f_item['clip_id']} {f_item['dimension']}: {label}")


if __name__ == "__main__":
    main()
