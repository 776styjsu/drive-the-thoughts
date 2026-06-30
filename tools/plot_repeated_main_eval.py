#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Generate the repeated-run main-evaluation F1 figure.

The plot is derived from ``runs/llm_matrix`` so the figure stays aligned with
the merged run artifacts and Table~\\ref{tab:repeated-main-eval}.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONSISTENT_STRINGS = {
    "consistent",
    "true",
    "pass",
    "passed",
    "match",
    "matched",
    "aligned",
    "yes",
}
INCONSISTENT_STRINGS = {
    "inconsistent",
    "contradictory",
    "invalid_parse",
    "false",
    "fail",
    "failed",
    "mismatch",
    "mismatched",
    "not_consistent",
    "not consistent",
    "partial",
    "partial_consistency",
    "partially_consistent",
    "partially consistent",
    "unaligned",
    "no",
}

MODEL_ROWS = (
    ("gpt", "GPT-5.5"),
    ("kimi", "Kimi K2.5"),
    ("qwen", "Qwen3.5-4B-FP8"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=repo_root / "runs" / "llm_matrix",
        help="Merged run directory. Default: %(default)s",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=repo_root / "benchmark.json",
        help="Canonical benchmark JSON. Default: %(default)s",
    )
    parser.add_argument(
        "--rule-result",
        type=Path,
        default=repo_root / "benchmark.rule_consistency.json",
        help="Rule-based consistency output. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "SafeVLA" / "figures" / "repeated_main_eval.pdf",
        help="Output figure path. Default: %(default)s",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cot_is_reliable(entry: dict[str, Any]) -> bool:
    nested = entry.get("cot_reliability")
    if isinstance(nested, dict) and "reliable" in nested:
        return nested["reliable"] is True
    value = entry.get("cot_reliable")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "reliable", "yes", "1"}
    return False


def benchmark_truth(benchmark: Path) -> dict[str, bool]:
    data = load_json(benchmark)
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("Benchmark must be a JSON list or contain entries")
    return {
        entry["clip_id"]: bool(entry["cot_action_consistency"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("clip_id") and _cot_is_reliable(entry)
    }


def prediction_is_consistent(entry: dict[str, Any]) -> bool:
    alignment = entry.get("evaluation", {}).get("cot_output_alignment", {})
    if not isinstance(alignment, dict):
        alignment = {}

    verdict = alignment.get("verdict")
    if isinstance(verdict, str):
        normalized = verdict.strip().lower()
        if normalized in CONSISTENT_STRINGS:
            return True
        if normalized in INCONSISTENT_STRINGS:
            return False

    score = alignment.get("score", entry.get("score"))
    if score is None:
        raise ValueError(f"Missing score/verdict for clip_id={entry.get('clip_id')}")
    return float(score) > 2


def f1_for_result(path: Path, truth: dict[str, bool]) -> float:
    data = load_json(path)
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError(f"Result file has no results list: {path}")

    tp = fp = fn = 0
    for entry in results:
        if not isinstance(entry, dict):
            continue
        clip_id = entry.get("clip_id")
        if clip_id not in truth:
            continue
        try:
            predicted_consistent = prediction_is_consistent(entry)
        except ValueError:
            continue
        actual_consistent = truth[clip_id]
        if not actual_consistent and not predicted_consistent:
            tp += 1
        elif actual_consistent and not predicted_consistent:
            fp += 1
        elif not actual_consistent and predicted_consistent:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def summarize_series(run_dir: Path, truth: dict[str, bool], prefix: str, variant: str) -> tuple[float, float]:
    values = [
        f1_for_result(path, truth)
        for path in sorted(run_dir.glob(f"{prefix}.{variant}.run_*.json"))
    ]
    if not values:
        raise FileNotFoundError(f"No result files found for {prefix}.{variant}")
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def rule_f1_for_result(path: Path, truth: dict[str, bool]) -> float:
    data = load_json(path)
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError(f"Rule result file has no results list: {path}")

    tp = fp = fn = 0
    for entry in results:
        if not isinstance(entry, dict):
            continue
        clip_id = entry.get("clip_id")
        if clip_id not in truth:
            continue
        report = entry.get("report")
        if not isinstance(report, dict):
            continue
        predicted_consistent = prediction_is_consistent(
            {"evaluation": {"cot_output_alignment": {"verdict": report.get("label")}}}
        )
        actual_consistent = truth[clip_id]
        if not actual_consistent and not predicted_consistent:
            tp += 1
        elif actual_consistent and not predicted_consistent:
            fp += 1
        elif not actual_consistent and predicted_consistent:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def plot(run_dir: Path, benchmark: Path, rule_result: Path, output: Path) -> None:
    truth = benchmark_truth(benchmark)
    labels = [label for _, label in MODEL_ROWS] + ["Rule-based"]
    y_positions = list(range(len(labels)))
    row_index = {label: idx for idx, label in enumerate(labels)}

    llm = {
        label: summarize_series(run_dir, truth, prefix, "llm")
        for prefix, label in MODEL_ROWS
    }
    fllm = {
        label: summarize_series(run_dir, truth, prefix, "f_llm_map_graph")
        for prefix, label in MODEL_ROWS
    }
    rule_f1 = rule_f1_for_result(rule_result, truth)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.9, 2.55), constrained_layout=True)
    llm_color = "#4E79A7"
    fllm_color = "#2F855A"
    rule_color = "#6B7280"
    bar_height = 0.24
    bar_containers = []

    for label in labels[:-1]:
        y = row_index[label]
        mean, sd = llm[label]
        bar_containers.append(ax.barh(
            y - bar_height / 1.7,
            mean,
            height=bar_height,
            color=llm_color,
            edgecolor="#315C86",
            linewidth=0.6,
            xerr=sd if sd else None,
            error_kw={"elinewidth": 0.8, "ecolor": "#4B5563", "capsize": 2},
            label="LLM (raw waypoints)" if label == labels[0] else None,
        ))
        mean, sd = fllm[label]
        bar_containers.append(ax.barh(
            y + bar_height / 1.7,
            mean,
            height=bar_height,
            color=fllm_color,
            edgecolor="#236542",
            linewidth=0.6,
            xerr=sd if sd else None,
            error_kw={"elinewidth": 0.8, "ecolor": "#4B5563", "capsize": 2},
            label="F-LLM (lane-relative)" if label == labels[0] else None,
        ))

    bar_containers.append(ax.barh(
        row_index["Rule-based"],
        rule_f1,
        height=bar_height,
        color=rule_color,
        edgecolor="#4B5563",
        linewidth=0.6,
        label="Rule-based",
    ))

    for container in bar_containers:
        labels_for_bars = [f"{patch.get_width():.2f}" for patch in container.patches]
        ax.bar_label(
            container,
            labels=labels_for_bars,
            padding=3,
            fontsize=7.5,
            color="#111827",
        )

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.82)
    ax.set_title("Repeated-run main evaluation (F1)", pad=7, fontsize=9.5)
    ax.set_xlabel("F1 score (inconsistent positive class)")
    ax.xaxis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="both", color="#9CA3AF", length=3)
    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=1,
        handlelength=1.2,
        borderpad=0.2,
        labelspacing=0.35,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.run_dir, args.benchmark, args.rule_result, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
