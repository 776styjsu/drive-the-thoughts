#!/usr/bin/env python3
"""Build a standalone, interactive HTML comparison from temporal evaluation reports."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="temporal-plot-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


POLICY_NAMES = {
    "single_frame": "Single step",
    "two_consecutive": "2 consecutive",
    "three_consecutive": "3 consecutive",
    "two_of_three": "2 of 3",
    "three_of_five": "3 of 5",
    "weighted_decay": "Weighted decay",
}
METRICS = {
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "precision": "Inconsistency precision",
    "recall": "Inconsistency recall",
    "f1": "Inconsistency F1",
}
BASELINE_COLOR = "#6554a4"
TEMPORAL_COLOR = "#147d79"


def plot_metric(report: dict, metric: str) -> str:
    """Embed publication-friendly SVGs; the HTML can download each figure."""
    policies = report["policies"]
    names = list(POLICY_NAMES)
    has_intervals = metric in ("accuracy", "balanced_accuracy")
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#d8dedd",
            "text.color": "#283e46",
            "axes.labelcolor": "#52666d",
            "xtick.color": "#52666d",
            "ytick.color": "#283e46",
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 4.2),
            gridspec_kw={"width_ratios": [1.4, 1]},
            layout="constrained",
        )
        left, right = axes
        baseline = policies["single_frame"][metric]
        for y, name in enumerate(names):
            value = policies[name][metric]
            if value is None:
                left.text(0.02, y, "Undefined", va="center", color="#71818a")
                continue
            color = BASELINE_COLOR if name == "single_frame" else TEMPORAL_COLOR
            left.barh(y, value, height=0.43, color=color, alpha=0.88)
            left.text(value + 0.016, y, f"{value:.1%}", va="center", fontsize=10)
        if baseline is not None:
            left.axvline(baseline, color=BASELINE_COLOR, ls="--", lw=1, alpha=0.65)
        left.set(
            yticks=range(len(names)),
            yticklabels=[POLICY_NAMES[n] for n in names],
            xlim=(0, 1.12),
            ylim=(len(names) - 0.4, -0.6),
            xlabel=METRICS[metric],
        )
        left.set_xticks([0, 0.25, 0.5, 0.75, 1])
        left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        left.set_title("Absolute performance", loc="left", pad=16, fontweight="bold")
        left.tick_params(axis="y", length=0, pad=8)
        left.set_axisbelow(True)
        left.grid(axis="x", color="#e9eded", lw=0.7)

        bound = 5.0
        for y, name in enumerate(names[1:]):
            value = policies[name][metric]
            if value is None or baseline is None:
                right.text(0, y, "Undefined", ha="center", va="center")
                continue
            delta = 100 * (value - baseline)
            interval = (
                policies[name]["difference_vs_single_frame"].get(metric, {}).get("ci95")
            )
            low, high = [100 * x for x in interval] if interval else (delta, delta)
            bound = max(bound, abs(low), abs(high))
            color = "#147d79" if delta >= 0 else "#b05b3b"
            if interval:
                right.hlines(y, low, high, color=color, lw=2)
                right.vlines([low, high], y - 0.08, y + 0.08, color=color, lw=1.3)
            right.plot(delta, y, "o", color=color, ms=6)
            right.annotate(
                f"{delta:+.1f} pp",
                (delta, y),
                xytext=(0, -17),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        right.axvline(0, color="#819195", lw=1, ls="--")
        right.set(
            xlim=(-bound * 1.35, bound * 1.35),
            ylim=(len(names) - 1.4, -0.55),
            yticks=range(len(names) - 1),
            yticklabels=[POLICY_NAMES[n] for n in names[1:]],
            xlabel="Change vs single step (percentage points)",
        )
        right.tick_params(axis="y", length=0, pad=8)
        right.set_title(
            "Paired change · 95% CI" if has_intervals else "Change vs single step",
            loc="left",
            pad=16,
            fontweight="bold",
        )
        right.set_axisbelow(True)
        right.grid(axis="x", color="#e9eded", lw=0.7)
        svg = io.StringIO()
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    return (
        "data:image/svg+xml;base64,"
        + base64.b64encode(svg.getvalue().encode()).decode()
    )


def build(reports: list[tuple[str, Path]], output: Path) -> None:
    datasets = []
    cohort = None
    for label, path in reports:
        report = json.loads(path.read_text())
        if report.get("kind") != "temporal_consistency_evaluation":
            raise ValueError(f"Not a temporal evaluation report: {path}")
        if report["n_evaluated"] != len(report["targets"]) or not report["targets"]:
            raise ValueError(f"Report has no complete target cohort: {path}")
        signature = {
            row["clip_id"]: (
                row["is_consistent"],
                tuple(row["frame_ids"]),
                tuple(row["timestamps_us"]),
            )
            for row in report["targets"]
        }
        if len(signature) != len(report["targets"]):
            raise ValueError(f"Duplicate target IDs: {path}")
        if cohort is not None and signature != cohort:
            raise ValueError(
                "All reports must use identical targets, labels, and windows"
            )
        cohort = signature
        datasets.append(
            {
                "label": label,
                "source": str(path),
                "report": report,
                "charts": {metric: plot_metric(report, metric) for metric in METRICS},
            }
        )
    payload = json.dumps(
        {"datasets": datasets, "policyNames": POLICY_NAMES, "metrics": METRICS},
        allow_nan=False,
    )
    # A source label or exclusion reason must not be able to close the script tag.
    payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    template = Path(__file__).with_name("temporal_comparison_template.html").read_text()
    html = template.replace("__TEMPORAL_DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Wrote {output}: {len(datasets)} monitors, {len(cohort)} shared targets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for specification in args.report:
        label, separator, path = specification.partition("=")
        if not separator or not label or not path:
            parser.error("--report must be LABEL=PATH")
        reports.append((label, Path(path)))
    build(reports, args.output)


if __name__ == "__main__":
    main()
