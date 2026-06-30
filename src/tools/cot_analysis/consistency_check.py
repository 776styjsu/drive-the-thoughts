# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Deterministic Alpamayo-R1-style CoT/trajectory consistency check.

Walks an ``extracted_frames`` directory produced by
``alpasim_utils.extract_frame``, and for each clip reads ``metadata.json``
(for the chain-of-thought) plus ``additional_info.json`` (for the trajectory
meta-action labels), then writes a per-clip ``cot_consistency.json`` sidecar
and an aggregate report.

This is the rule-based counterpart to the LLM-judge pipeline in
``cot_analysis/__main__.py``: same input, no API calls, fully reproducible.

Usage::

    # Score every clip under an extracted_frames directory:
    uv run python -m cot_analysis.consistency_check tutorial_alpamayo15_first20/extracted_frames

    # Score only the clips selected by a benchmark JSON (resolves each entry's
    # source_scene_file to its metadata.json):
    uv run python -m cot_analysis.consistency_check \
        --benchmark_json benchmark_expanded_100.json \
        --output benchmark_expanded_100.rule_consistency.json

    # Exact-label variant (no direction families; gentle_decelerate only matches
    # gentle_decelerate). Defaults the output to
    # benchmark_expanded_100.rule_consistency.exact.json:
    uv run python -m cot_analysis.consistency_check \
        --benchmark_json benchmark_expanded_100.json --match-mode exact
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alpasim_utils.consistency import (
    MATCH_MODES,
    ConsistencyReport,
    match_cot_to_trajectory,
)
from alpasim_utils.meta_actions_types import MetaActionThresholds
from alpasim_utils.trajectory_additional_info import build_additional_info

try:
    from .__main__ import _resolve_benchmark_source_path
except ImportError:  # pragma: no cover - fallback when run as a loose script
    from cot_analysis.__main__ import _resolve_benchmark_source_path


METADATA_FILENAME = "metadata.json"
ADDITIONAL_INFO_FILENAME = "additional_info.json"
DEFAULT_OUTPUT_FILENAME = "cot_consistency.json"
DEFAULT_AGGREGATE_FILENAME = "cot_consistency_report.json"


@dataclass
class ClipResult:
    clip_id: str
    metadata_path: Path
    cot_text: str | None
    report: ConsistencyReport | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "metadata_path": str(self.metadata_path),
            "cot_text": self.cot_text,
            "report": self.report.to_dict() if self.report is not None else None,
            "error": self.error,
        }


def _flatten_cot(value: Any) -> str | None:
    """Mirrors the CoT flattening used by the LLM-judge tool."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(part for item in value if (part := _flatten_cot(item)))
    return str(value)


def _process_clip(
    metadata_path: Path,
    *,
    rederive_meta_actions: bool = False,
    meta_action_thresholds: MetaActionThresholds | None = None,
    match_mode: str = "family",
) -> ClipResult:
    clip_dir = metadata_path.parent
    clip_id = clip_dir.name

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        return ClipResult(
            clip_id=clip_id,
            metadata_path=metadata_path,
            cot_text=None,
            report=None,
            error=f"failed to read metadata: {exc}",
        )

    cot_text = _flatten_cot(metadata.get("chain_of_thought"))

    if rederive_meta_actions:
        # Derive meta_actions straight from the trajectory in metadata.json so
        # threshold overrides take effect without re-running extract_frame (the
        # stored additional_info.json bakes in the extraction-time thresholds).
        try:
            additional_info = build_additional_info(
                metadata,
                metadata_path=str(metadata_path),
                meta_action_thresholds=meta_action_thresholds,
            )
        except Exception as exc:
            return ClipResult(
                clip_id=clip_id,
                metadata_path=metadata_path,
                cot_text=cot_text,
                report=None,
                error=f"failed to derive meta_actions from metadata: {exc}",
            )
        meta_actions = additional_info.get("meta_actions") or {}
        missing_meta_error = "could not derive meta_actions from metadata trajectory"
    else:
        additional_info_path = clip_dir / ADDITIONAL_INFO_FILENAME
        if not additional_info_path.exists():
            return ClipResult(
                clip_id=clip_id,
                metadata_path=metadata_path,
                cot_text=cot_text,
                report=None,
                error=(
                    f"missing {ADDITIONAL_INFO_FILENAME}; run extract_frame first "
                    "or pass --rederive-meta-actions"
                ),
            )
        try:
            with additional_info_path.open("r", encoding="utf-8") as f:
                additional_info = json.load(f)
        except Exception as exc:
            return ClipResult(
                clip_id=clip_id,
                metadata_path=metadata_path,
                cot_text=cot_text,
                report=None,
                error=f"failed to read additional_info: {exc}",
            )
        meta_actions = additional_info.get("meta_actions") or {}
        missing_meta_error = (
            "additional_info.json has no meta_actions block (re-run extract_frame)"
        )

    if not meta_actions:
        return ClipResult(
            clip_id=clip_id,
            metadata_path=metadata_path,
            cot_text=cot_text,
            report=None,
            error=missing_meta_error,
        )

    report = match_cot_to_trajectory(cot_text or "", meta_actions, match_mode=match_mode)
    return ClipResult(
        clip_id=clip_id,
        metadata_path=metadata_path,
        cot_text=cot_text,
        report=report,
    )


def _meta_action_provenance(
    rederive_meta_actions: bool,
    meta_action_thresholds: MetaActionThresholds | None,
    match_mode: str = "family",
) -> dict:
    """Record how meta_actions were sourced, for report reproducibility."""
    return {
        "match_mode": match_mode,
        "meta_action_source": (
            "rederived" if rederive_meta_actions else ADDITIONAL_INFO_FILENAME
        ),
        "meta_action_thresholds": (
            asdict(meta_action_thresholds)
            if rederive_meta_actions and meta_action_thresholds is not None
            else None
        ),
    }


def _write_per_clip_sidecar(result: ClipResult, output_name: str) -> Path | None:
    if result.report is None:
        return None
    sidecar_path = result.metadata_path.with_name(output_name)
    payload = {
        "schema_version": 1,
        "kind": "alpasim_cot_consistency",
        "clip_id": result.clip_id,
        **result.report.to_dict(),
    }
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return sidecar_path


def _aggregate(results: list[ClipResult]) -> dict:
    scored = [r for r in results if r.report is not None]
    score_values = [r.report.score for r in scored]
    label_counts = Counter(r.report.label for r in scored)
    long_verdict_counts = Counter(r.report.longitudinal.verdict for r in scored)
    lat_verdict_counts = Counter(r.report.lateral.verdict for r in scored)
    contradictions = [
        {
            "clip_id": r.clip_id,
            "score": r.report.score,
            "longitudinal": r.report.longitudinal.to_dict(),
            "lateral": r.report.lateral.to_dict(),
            "cot_text": r.cot_text,
        }
        for r in scored
        if "contradict" in {r.report.longitudinal.verdict, r.report.lateral.verdict}
    ]
    errors = [
        {"clip_id": r.clip_id, "error": r.error}
        for r in results
        if r.error is not None
    ]

    return {
        "total_clips": len(results),
        "scored_clips": len(scored),
        "errored_clips": len(errors),
        "mean_score": sum(score_values) / len(score_values) if score_values else None,
        "min_score": min(score_values) if score_values else None,
        "max_score": max(score_values) if score_values else None,
        "label_distribution": dict(label_counts),
        "longitudinal_verdict_distribution": dict(long_verdict_counts),
        "lateral_verdict_distribution": dict(lat_verdict_counts),
        "contradictions": contradictions,
        "errors": errors,
    }


def run(
    extracted_frames_dir: Path,
    *,
    output_path: Path,
    per_clip_sidecars: bool,
    sidecar_name: str,
    rederive_meta_actions: bool = False,
    meta_action_thresholds: MetaActionThresholds | None = None,
    match_mode: str = "family",
) -> dict:
    if not extracted_frames_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {extracted_frames_dir}")
    if not extracted_frames_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {extracted_frames_dir}")

    metadata_paths = sorted(
        p for p in extracted_frames_dir.rglob(METADATA_FILENAME) if p.is_file()
    )

    results: list[ClipResult] = []
    for path in metadata_paths:
        result = _process_clip(
            path,
            rederive_meta_actions=rederive_meta_actions,
            meta_action_thresholds=meta_action_thresholds,
            match_mode=match_mode,
        )
        if per_clip_sidecars:
            _write_per_clip_sidecar(result, sidecar_name)
        results.append(result)

    aggregate = _aggregate(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "alpasim_cot_consistency_report",
        "extracted_frames_dir": str(extracted_frames_dir),
        **_meta_action_provenance(
            rederive_meta_actions, meta_action_thresholds, match_mode
        ),
        "summary": aggregate,
        "results": [r.to_dict() for r in results],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return aggregate


def _load_benchmark_items(benchmark_json: Path) -> list[dict]:
    """Load benchmark entries from a list or a {results|entries: [...]} object."""
    with benchmark_json.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)
    if isinstance(benchmark, dict):
        items = benchmark.get("results", benchmark.get("entries", []))
    else:
        items = benchmark
    if not isinstance(items, list):
        raise ValueError(f"Expected a benchmark list or results/entries list: {benchmark_json}")
    return [item for item in items if isinstance(item, dict)]


def _process_benchmark_item(
    item: dict,
    benchmark_json: Path,
    source_root: Path,
    *,
    rederive_meta_actions: bool = False,
    meta_action_thresholds: MetaActionThresholds | None = None,
    match_mode: str = "family",
) -> ClipResult:
    """Resolve one benchmark entry to its metadata.json and score that clip."""
    clip_id = str(item.get("clip_id", "")).strip()
    source_scene_file = item.get("source_scene_file")
    metadata_path = _resolve_benchmark_source_path(
        source_scene_file, benchmark_json, source_root
    )
    if metadata_path is None:
        return ClipResult(
            clip_id=clip_id or "<unknown>",
            metadata_path=Path(str(source_scene_file or "")),
            cot_text=None,
            report=None,
            error=(
                f"could not resolve source_scene_file: {source_scene_file!r}"
                if source_scene_file
                else "benchmark entry has no source_scene_file"
            ),
        )

    result = _process_clip(
        metadata_path,
        rederive_meta_actions=rederive_meta_actions,
        meta_action_thresholds=meta_action_thresholds,
        match_mode=match_mode,
    )
    # Prefer the benchmark's clip_id so downstream tools (e.g.
    # check_consistency_accuracy.py) match entries by the benchmark key.
    if clip_id:
        result.clip_id = clip_id
    return result


def run_benchmark(
    benchmark_json: Path,
    *,
    source_root: Path,
    output_path: Path,
    rederive_meta_actions: bool = False,
    meta_action_thresholds: MetaActionThresholds | None = None,
    match_mode: str = "family",
) -> dict:
    """Score only the clips selected by a benchmark JSON and write a report.

    Per-clip sidecars are intentionally not written in this mode: the source
    frame directories already hold an LLM-judge ``cot_consistency.json`` and
    the benchmark report is the deliverable.
    """
    if not benchmark_json.exists():
        raise FileNotFoundError(f"Benchmark file does not exist: {benchmark_json}")

    items = _load_benchmark_items(benchmark_json)
    results = [
        _process_benchmark_item(
            item,
            benchmark_json,
            source_root,
            rederive_meta_actions=rederive_meta_actions,
            meta_action_thresholds=meta_action_thresholds,
            match_mode=match_mode,
        )
        for item in items
    ]
    aggregate = _aggregate(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "alpasim_cot_consistency_report",
        "benchmark_file": str(benchmark_json),
        "benchmark_source_root": str(source_root),
        **_meta_action_provenance(
            rederive_meta_actions, meta_action_thresholds, match_mode
        ),
        "summary": aggregate,
        "results": [r.to_dict() for r in results],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score CoT/trajectory consistency for every clip under an "
            "extracted_frames directory using the deterministic Alpamayo-R1 "
            "meta-action vocabulary."
        )
    )
    parser.add_argument(
        "extracted_frames_dir",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Directory to recursively search for metadata.json files. "
            "Omit when using --benchmark_json."
        ),
    )
    parser.add_argument(
        "--benchmark_json",
        type=Path,
        default=None,
        help=(
            "Benchmark JSON whose entries carry clip_id + source_scene_file. "
            "Scores only those clips by resolving each source_scene_file to its "
            "metadata.json. Mutually exclusive with extracted_frames_dir."
        ),
    )
    parser.add_argument(
        "--benchmark_source_root",
        type=Path,
        default=Path("."),
        help="Root for resolving benchmark source_scene_file paths (default: cwd).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Aggregate report path. "
            f"Default: <extracted_frames_dir>/{DEFAULT_AGGREGATE_FILENAME}"
        ),
    )
    parser.add_argument(
        "--no-per-clip",
        action="store_true",
        help=f"Skip writing the per-clip {DEFAULT_OUTPUT_FILENAME} sidecars.",
    )
    parser.add_argument(
        "--sidecar-name",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Per-clip sidecar filename (default: {DEFAULT_OUTPUT_FILENAME}).",
    )
    parser.add_argument(
        "--match-mode",
        choices=MATCH_MODES,
        default="family",
        help=(
            "How a CoT label matches a trajectory label. 'family' (default) is "
            "the Alpamayo-R1 behaviour where same-direction labels (e.g. "
            "gentle/strong decelerate) are interchangeable; 'exact' matches by "
            "exact label only (gentle_decelerate != strong_decelerate) and never "
            "emits the 'contradictory' label."
        ),
    )
    parser.add_argument(
        "--rederive-meta-actions",
        action="store_true",
        help=(
            "Re-derive trajectory meta_actions from each metadata.json instead "
            "of reading the precomputed additional_info.json. Required for the "
            "threshold overrides below to take effect (no extract_frame re-run "
            "needed). Any --*-mps2/-mps/-1pm override implies this flag."
        ),
    )
    thresholds_group = parser.add_argument_group(
        "meta-action thresholds",
        "Override MetaActionThresholds fields; each implies --rederive-meta-actions.",
    )
    for field_name, default in asdict(MetaActionThresholds()).items():
        flag = "--" + field_name.replace("_", "-")
        thresholds_group.add_argument(
            flag,
            dest=field_name,
            type=float,
            default=None,
            help=f"Override MetaActionThresholds.{field_name} (default: {default}).",
        )
    args = parser.parse_args(argv)

    has_dir = args.extracted_frames_dir is not None
    has_benchmark = args.benchmark_json is not None
    if has_dir == has_benchmark:
        parser.error("Specify exactly one of extracted_frames_dir or --benchmark_json")

    threshold_overrides = {
        field_name: getattr(args, field_name)
        for field_name in asdict(MetaActionThresholds())
        if getattr(args, field_name) is not None
    }
    rederive_meta_actions = args.rederive_meta_actions or bool(threshold_overrides)
    meta_action_thresholds = (
        MetaActionThresholds(**threshold_overrides) if rederive_meta_actions else None
    )

    # Tag the default output with the match mode so an exact-mode run does not
    # clobber the family-mode report (and vice versa).
    mode_suffix = "" if args.match_mode == "family" else f".{args.match_mode}"

    if has_benchmark:
        output_path = args.output or (
            args.benchmark_json.parent
            / f"{args.benchmark_json.stem}.rule_consistency{mode_suffix}.json"
        )
        summary = run_benchmark(
            args.benchmark_json,
            source_root=args.benchmark_source_root,
            output_path=output_path,
            rederive_meta_actions=rederive_meta_actions,
            meta_action_thresholds=meta_action_thresholds,
            match_mode=args.match_mode,
        )
    else:
        output_path = args.output or (args.extracted_frames_dir / DEFAULT_AGGREGATE_FILENAME)
        summary = run(
            args.extracted_frames_dir,
            output_path=output_path,
            per_clip_sidecars=not args.no_per_clip,
            sidecar_name=args.sidecar_name,
            rederive_meta_actions=rederive_meta_actions,
            meta_action_thresholds=meta_action_thresholds,
            match_mode=args.match_mode,
        )

    print(f"Match mode: {args.match_mode}")
    if rederive_meta_actions:
        print(f"Meta-actions re-derived with thresholds: {asdict(meta_action_thresholds)}")
    print(f"Scored {summary['scored_clips']}/{summary['total_clips']} clips.")
    if summary["mean_score"] is not None:
        print(
            f"Mean consistency score: {summary['mean_score']:.3f} "
            f"(min {summary['min_score']:.3f}, max {summary['max_score']:.3f})"
        )
    print(f"Label distribution: {summary['label_distribution']}")
    if summary["contradictions"]:
        print(f"WARNING: {len(summary['contradictions'])} clip(s) flagged as contradictory.")
    if summary["errored_clips"]:
        print(f"Errors on {summary['errored_clips']} clip(s) (see report).")
    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
