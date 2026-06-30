#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Plot CoT reliability coverage as a scenario-by-CoT heatmap.

Default axes:
- x-axis: individual NuRec scene attribute values, excluding behavior by default.
- y-axis: CoT high_level_decision, using longitudinal and lateral decisions.

Examples:
    uv run python tools/plot_cot_reliability_heatmap.py \
        --benchmark benchmark_expanded_100.json

    uv run python tools/plot_cot_reliability_heatmap.py \
        --benchmark benchmark_expanded_100.json \
        --scenario-axis scene_uuid \
        --output benchmark_expanded_100.scene_uuid_cot_reliability_heatmap.html
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import textwrap
from collections import Counter, defaultdict
from copy import copy
from pathlib import Path
from typing import Any, Iterable


SCENARIO_AXES = (
    "scene_attributes",
    "nurec_scene",
    "nurec_summary",
    "scene_uuid",
    "scene_folder",
    "clip_id",
    "behavior",
    "layout",
    "lighting",
    "road_types",
    "surface_conditions",
    "traffic_density",
    "weather",
    "vru",
)
COT_AXES = (
    "chain_of_thought",
    "decision",
    "meta_action",
)
ORDERINGS = (
    "count",
    "field",
    "first_seen",
    "name",
)
OUTPUT_FORMATS = (
    "auto",
    "html",
    "png",
)
NUREC_SCENE_FIELDS = (
    "behavior",
    "layout",
    "lighting",
    "road_types",
    "surface_conditions",
    "traffic_density",
    "weather",
    "vru",
)
NUREC_ATTRIBUTE_FIELDS = (
    "layout",
    "lighting",
    "road_types",
    "surface_conditions",
    "traffic_density",
    "weather",
    "vru",
)
CAUSAL_FACTOR_FIELDS = (
    "critical_objects",
    "traffic_lights",
    "yield_stop_control",
    "road_events",
    "lane_lanelines",
    "routing_intent",
    "odd_constraints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a heatmap whose columns are scene axis values, rows are "
            "CoT types, color is CoT reliability, and text is reliable/total counts."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark.json"),
        help="Benchmark JSON file to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Heatmap output path. Defaults to <benchmark>.cot_reliability_heatmap.html.",
    )
    parser.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        default="auto",
        help="Output format. auto uses the --output suffix, defaulting to HTML.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="CSV output path for the aggregated cell counts.",
    )
    parser.add_argument(
        "--scenario-axis",
        choices=SCENARIO_AXES,
        default="scene_attributes",
        help=(
            "Column grouping. scene_attributes counts each layout, lighting, "
            "road type, surface, traffic density, weather, and VRU value separately."
        ),
    )
    parser.add_argument(
        "--cot-axis",
        choices=COT_AXES,
        default="decision",
        help="Row grouping. decision uses labels.cot_decision_label.high_level_decision.",
    )
    parser.add_argument(
        "--order",
        choices=ORDERINGS,
        default="field",
        help=(
            "Axis ordering. field groups scene_attributes by field name; "
            "count sorts by descending coverage, then name."
        ),
    )
    parser.add_argument(
        "--wrap-x",
        type=int,
        default=24,
        help="Maximum wrapped x-axis label line width.",
    )
    parser.add_argument(
        "--wrap-y",
        type=int,
        default=58,
        help="Maximum wrapped y-axis label line width.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG DPI.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title.",
    )
    parser.add_argument(
        "--max-label-lines",
        type=int,
        default=4,
        help="Maximum wrapped lines shown for each axis label.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]

    if isinstance(data, dict):
        for key in ("examples", "entries", "results", "data"):
            entries = data.get(key)
            if isinstance(entries, list):
                return [entry for entry in entries if isinstance(entry, dict)]

    raise ValueError(f"{path} must contain a benchmark list or an object with entries")


def first_seen(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def order_labels(labels: list[str], ordering: str) -> list[str]:
    if ordering == "first_seen":
        return first_seen(labels)
    if ordering == "name":
        return sorted(set(labels))
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], label))


def order_scene_labels(labels: list[str], ordering: str) -> list[str]:
    if ordering != "field":
        return order_labels(labels, ordering)

    counts = Counter(labels)
    field_order = {
        field: idx
        for idx, field in enumerate(("behavior", *NUREC_ATTRIBUTE_FIELDS))
    }

    def sort_key(label: str) -> tuple[int, int, str]:
        field = label.split("=", 1)[0]
        return (field_order.get(field, len(field_order)), -counts[label], label)

    return sorted(counts, key=sort_key)


def list_label(value: Any) -> str:
    if isinstance(value, list):
        return "+".join(str(item) for item in value) if value else "unknown"
    if value is None:
        return "unknown"
    return str(value)


def nurec_scene(entry: dict[str, Any]) -> dict[str, Any]:
    labels = entry.get("labels")
    if not isinstance(labels, dict):
        return {}
    scene = labels.get("nurec_scene")
    return scene if isinstance(scene, dict) else {}


def vru_label(scene: dict[str, Any]) -> str:
    return "vru" if scene.get("vrus") is True else "no-vru"


def vru_axis_label(scene: dict[str, Any]) -> str:
    if scene.get("vrus") is True:
        return "vru=true"
    if scene.get("vrus") is False:
        return "vru=false"
    return "vru=unknown"


def field_value_labels(scene: dict[str, Any], field: str) -> list[str]:
    if field == "vru":
        return [vru_axis_label(scene)]

    value = scene.get(field)
    values = value if isinstance(value, list) else [value]
    if not values:
        values = [None]

    labels = []
    for item in values:
        item_label = "unknown" if item in (None, "") else str(item)
        labels.append(f"{field}={item_label}")
    return first_seen(labels)


def scene_field_label(scene: dict[str, Any], field: str) -> str:
    if field == "vru":
        return vru_label(scene)
    return list_label(scene.get(field))


def scene_descriptor_label(scene: dict[str, Any], fields: Iterable[str]) -> str:
    return " | ".join(
        f"{field}={scene_field_label(scene, field)}"
        for field in fields
    )


def scenario_label(entry: dict[str, Any], axis: str) -> str:
    scene = nurec_scene(entry)

    if axis == "nurec_scene":
        return scene_descriptor_label(scene, NUREC_SCENE_FIELDS)
    if axis == "nurec_summary":
        behavior = list_label(scene.get("behavior"))
        layout = list_label(scene.get("layout"))
        traffic = list_label(scene.get("traffic_density"))
        vru = vru_label(scene)
        return f"{behavior} | {layout} | {traffic} | {vru}"
    if axis == "scene_uuid":
        return str(scene.get("scene_uuid") or scene_uuid_from_clip(entry) or "unknown")
    if axis == "scene_folder":
        scene_folder = str(entry.get("scene_folder") or "unknown")
        return Path(scene_folder).name if scene_folder != "unknown" else scene_folder
    if axis == "clip_id":
        return str(entry.get("clip_id") or "unknown")
    if axis == "behavior":
        return list_label(scene.get("behavior"))
    if axis == "layout":
        return list_label(scene.get("layout"))
    if axis == "lighting":
        return list_label(scene.get("lighting"))
    if axis == "road_types":
        return list_label(scene.get("road_types"))
    if axis == "surface_conditions":
        return list_label(scene.get("surface_conditions"))
    if axis == "traffic_density":
        return list_label(scene.get("traffic_density"))
    if axis == "weather":
        return list_label(scene.get("weather"))
    if axis == "vru":
        return vru_label(scene)

    raise ValueError(f"Unsupported scenario axis: {axis}")


def scenario_labels_for_entry(entry: dict[str, Any], axis: str) -> list[str]:
    scene = nurec_scene(entry)

    if axis == "scene_attributes":
        labels: list[str] = []
        for field in NUREC_ATTRIBUTE_FIELDS:
            labels.extend(field_value_labels(scene, field))
        return first_seen(labels)

    if axis in {
        "behavior",
        "layout",
        "lighting",
        "road_types",
        "surface_conditions",
        "traffic_density",
        "weather",
        "vru",
    }:
        return field_value_labels(scene, axis)

    return [scenario_label(entry, axis)]


def scene_uuid_from_clip(entry: dict[str, Any]) -> str | None:
    clip_id = entry.get("clip_id")
    if not isinstance(clip_id, str):
        return None
    if "_t" in clip_id:
        return clip_id.rsplit("_t", 1)[0]
    return clip_id


def cot_text(entry: dict[str, Any]) -> str:
    value = entry.get("chain_of_thought")
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
        if isinstance(parsed, list):
            return " | ".join(str(item) for item in parsed)
        return str(parsed)
    if value is None:
        return "unknown"
    return str(value)


def cot_axis_label(entry: dict[str, Any], axis: str) -> str:
    decision_label = (
        entry.get("labels", {})
        .get("cot_decision_label", {})
        if isinstance(entry.get("labels"), dict)
        else {}
    )

    if axis == "chain_of_thought":
        return cot_text(entry)

    if axis == "decision":
        decision = decision_label.get("high_level_decision", {})
        if not isinstance(decision, dict):
            decision = {}
        longitudinal = decision.get("longitudinal", "unknown")
        lateral = decision.get("lateral", "unknown")
        return f"{longitudinal} | {lateral}"

    if axis == "meta_action":
        action = decision_label.get("atomic_meta_action_hint", {})
        if not isinstance(action, dict):
            action = {}
        longitudinal = action.get("longitudinal", "unknown")
        lateral = action.get("lateral", "unknown")
        return f"{longitudinal} | {lateral}"

    raise ValueError(f"Unsupported CoT axis: {axis}")


def cot_axis_components(entry: dict[str, Any], axis: str) -> dict[str, str]:
    decision_label = (
        entry.get("labels", {})
        .get("cot_decision_label", {})
        if isinstance(entry.get("labels"), dict)
        else {}
    )
    if not isinstance(decision_label, dict):
        decision_label = {}

    if axis == "chain_of_thought":
        longitudinal = cot_text(entry)
        lateral = "unknown"
    elif axis == "decision":
        decision = decision_label.get("high_level_decision", {})
        if not isinstance(decision, dict):
            decision = {}
        longitudinal = str(decision.get("longitudinal", "unknown"))
        lateral = str(decision.get("lateral", "unknown"))
    elif axis == "meta_action":
        action = decision_label.get("atomic_meta_action_hint", {})
        if not isinstance(action, dict):
            action = {}
        longitudinal = str(action.get("longitudinal", "unknown"))
        lateral = str(action.get("lateral", "unknown"))
    else:
        raise ValueError(f"Unsupported CoT axis: {axis}")

    return {
        "longitudinal_cot_type": longitudinal,
        "lateral_cot_type": lateral,
        "causal_factor_type": causal_factor_label(entry),
    }


def cot_axis_facets(entry: dict[str, Any], axis: str) -> list[dict[str, str]]:
    components = cot_axis_components(entry, axis)

    if axis == "chain_of_thought":
        return [
            {
                "row_label": f"chain_of_thought={components['longitudinal_cot_type']}",
                "longitudinal_cot_type": components["longitudinal_cot_type"],
                "lateral_cot_type": "",
                "causal_factor_type": "",
            }
        ]

    facets = [
        {
            "row_label": f"longitudinal={components['longitudinal_cot_type']}",
            "longitudinal_cot_type": components["longitudinal_cot_type"],
            "lateral_cot_type": "",
            "causal_factor_type": "",
        },
        {
            "row_label": f"lateral={components['lateral_cot_type']}",
            "longitudinal_cot_type": "",
            "lateral_cot_type": components["lateral_cot_type"],
            "causal_factor_type": "",
        },
    ]

    causal_factors = [
        factor
        for factor in components["causal_factor_type"].split("+")
        if factor and factor != "none"
    ]
    facets.extend(
        {
            "row_label": f"causal_factor={factor}",
            "longitudinal_cot_type": "",
            "lateral_cot_type": "",
            "causal_factor_type": factor,
        }
        for factor in causal_factors
    )
    return facets


def causal_factor_label(entry: dict[str, Any]) -> str:
    decision_label = (
        entry.get("labels", {})
        .get("cot_decision_label", {})
        if isinstance(entry.get("labels"), dict)
        else {}
    )
    if not isinstance(decision_label, dict):
        return "none"

    raw_categories = decision_label.get("critical_component_categories", [])
    categories = raw_categories if isinstance(raw_categories, list) else [raw_categories]
    allowed_categories = set(CAUSAL_FACTOR_FIELDS)
    normalized = [
        str(category)
        for category in categories
        if category and str(category) in allowed_categories
    ]
    if not normalized:
        return "none"

    order = {category: idx for idx, category in enumerate(CAUSAL_FACTOR_FIELDS)}
    return "+".join(
        sorted(
            first_seen(normalized),
            key=lambda category: (order.get(category, len(order)), category),
        )
    )


def reliability_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "reliable", "yes", "1"}:
            return True
        if normalized in {"false", "unreliable", "no", "0"}:
            return False
    return None


def entry_reliability(entry: dict[str, Any]) -> bool | None:
    """Resolve a benchmark entry's reliability across both on-disk schemas.

    - flat ``cot_reliable`` (bool/str), used by benchmark_expanded_*.json
    - nested ``cot_reliability.reliable`` (bool), used by benchmark.json
    """
    nested = entry.get("cot_reliability")
    if isinstance(nested, dict) and "reliable" in nested:
        return reliability_value(nested["reliable"])
    return reliability_value(entry.get("cot_reliable"))


def aggregate(
    entries: list[dict[str, Any]],
    *,
    scenario_axis: str,
    cot_axis: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "reliable": 0,
            "unreliable": 0,
            "unknown": 0,
            "clip_ids": [],
            "longitudinal_cot_type": "unknown",
            "lateral_cot_type": "",
            "causal_factor_type": "",
        }
    )
    for entry in entries:
        cot_facets = cot_axis_facets(entry, cot_axis)
        reliable = entry_reliability(entry)
        for cot_facet in cot_facets:
            cot = cot_facet["row_label"]
            for scenario in scenario_labels_for_entry(entry, scenario_axis):
                cell = cells[(cot, scenario)]
                cell["total"] += 1
                cell["clip_ids"].append(str(entry.get("clip_id", "")))
                cell["longitudinal_cot_type"] = cot_facet["longitudinal_cot_type"]
                cell["lateral_cot_type"] = cot_facet["lateral_cot_type"]
                cell["causal_factor_type"] = cot_facet["causal_factor_type"]

                if reliable is True:
                    cell["reliable"] += 1
                elif reliable is False:
                    cell["unreliable"] += 1
                else:
                    cell["unknown"] += 1

    return cells


def aggregate_row_summaries(
    entries: list[dict[str, Any]],
    *,
    cot_axis: str,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "reliable": 0,
            "unreliable": 0,
            "unknown": 0,
            "clip_ids": [],
        }
    )
    for entry in entries:
        reliable = entry_reliability(entry)
        for cot_facet in cot_axis_facets(entry, cot_axis):
            summary = summaries[cot_facet["row_label"]]
            summary["total"] += 1
            summary["clip_ids"].append(str(entry.get("clip_id", "")))

            if reliable is True:
                summary["reliable"] += 1
            elif reliable is False:
                summary["unreliable"] += 1
            else:
                summary["unknown"] += 1

    return summaries


def default_output_path(benchmark_path: Path) -> Path:
    return benchmark_path.with_suffix(".cot_reliability_heatmap.html")


def default_summary_path(output_path: Path) -> Path:
    return output_path.with_suffix(".csv")


def write_summary_csv(
    path: Path,
    *,
    cells: dict[tuple[str, str], dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "longitudinal_cot_type",
                "lateral_cot_type",
                "causal_factor_type",
                "scene_axis_value",
                "total",
                "reliable",
                "unreliable",
                "unknown",
                "reliability_rate",
                "clip_ids",
            ]
        )
        for (cot, scenario), cell in sorted(cells.items()):
            total = cell["total"]
            rate = cell["reliable"] / total if total else ""
            writer.writerow(
                [
                    cell["longitudinal_cot_type"],
                    cell["lateral_cot_type"],
                    cell["causal_factor_type"],
                    scenario,
                    total,
                    cell["reliable"],
                    cell["unreliable"],
                    cell["unknown"],
                    f"{rate:.6f}" if rate != "" else "",
                    " ".join(clip_id for clip_id in cell["clip_ids"] if clip_id),
                ]
            )


def fit_wrapped_label(label: str, width: int, max_lines: int) -> str:
    parts: list[str] = []
    for chunk in str(label).split(" | "):
        wrapped = textwrap.wrap(chunk, width=width, break_long_words=False)
        parts.extend(wrapped or [chunk])

    if len(parts) <= max_lines:
        return "\n".join(parts)

    visible = parts[:max_lines]
    visible[-1] = visible[-1].rstrip(". ") + "..."
    return "\n".join(visible)


def build_matrices(
    cells: dict[tuple[str, str], dict[str, Any]],
    *,
    cot_labels: list[str],
    scenario_labels: list[str],
) -> tuple[Any, list[list[str]]]:
    import numpy as np

    matrix = np.full((len(cot_labels), len(scenario_labels)), np.nan)
    annotations: list[list[str]] = [
        ["" for _ in scenario_labels] for _ in cot_labels
    ]

    cot_index = {label: idx for idx, label in enumerate(cot_labels)}
    scenario_index = {label: idx for idx, label in enumerate(scenario_labels)}

    for (cot, scenario), cell in cells.items():
        row = cot_index[cot]
        col = scenario_index[scenario]
        total = cell["total"]
        reliable = cell["reliable"]
        unknown = cell["unknown"]
        matrix[row, col] = reliable / total if total else np.nan
        annotations[row][col] = f"{reliable}/{total}"
        if unknown:
            annotations[row][col] += f"\n?{unknown}"

    return matrix, annotations


def plot_heatmap(
    output_path: Path,
    *,
    matrix: Any,
    annotations: list[list[str]],
    cot_labels: list[str],
    scenario_labels: list[str],
    title: str,
    wrap_x: int,
    wrap_y: int,
    max_label_lines: int,
    dpi: int,
) -> None:
    try:
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PNG output requires matplotlib and numpy. Either write .html output "
            "or run in an environment with the tools extra installed."
        ) from exc

    rows, cols = matrix.shape
    fig_width = min(max(12.0, cols * 0.62 + 9.0), 48.0)
    fig_height = min(max(7.0, rows * 0.44 + 4.2), 40.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = copy(plt.get_cmap("RdYlGn"))
    cmap.set_bad("#d9d9d9")

    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(
        [
            fit_wrapped_label(label, wrap_x, max_label_lines)
            for label in scenario_labels
        ],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=7,
    )
    ax.set_yticklabels(
        [fit_wrapped_label(label, wrap_y, max_label_lines) for label in cot_labels],
        fontsize=7,
    )

    ax.set_xlabel("Scene axis value")
    ax.set_ylabel("CoT type")
    ax.set_title(title)

    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(rows):
        for col in range(cols):
            text = annotations[row][col]
            if not text:
                continue
            value = matrix[row, col]
            color = "white" if value < 0.25 or value > 0.80 else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=6, color=color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Reliable / total examples")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["unreliable", "mixed", "reliable"])

    fig.subplots_adjust(left=0.31, right=0.98, top=0.92, bottom=0.28)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def resolved_output_format(output_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    if output_path.suffix.lower() == ".png":
        return "png"
    return "html"


def interpolate_channel(start: int, end: int, amount: float) -> int:
    return round(start + (end - start) * amount)


def reliability_color(rate: float) -> str:
    red = (198, 40, 40)
    yellow = (245, 196, 83)
    green = (47, 133, 90)

    if rate <= 0.5:
        amount = max(0.0, min(1.0, rate / 0.5))
        rgb = tuple(
            interpolate_channel(red_channel, yellow_channel, amount)
            for red_channel, yellow_channel in zip(red, yellow)
        )
    else:
        amount = max(0.0, min(1.0, (rate - 0.5) / 0.5))
        rgb = tuple(
            interpolate_channel(yellow_channel, green_channel, amount)
            for yellow_channel, green_channel in zip(yellow, green)
        )
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def text_color_for_background(background: str) -> str:
    rgb = tuple(int(background[idx:idx + 2], 16) for idx in (1, 3, 5))
    red, green, blue = [channel / 255.0 for channel in rgb]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < 0.46 else "#111827"


def html_wrapped_label(label: str, width: int, max_lines: int) -> str:
    return "<br>".join(
        html.escape(line)
        for line in fit_wrapped_label(label, width, max_lines).splitlines()
    )


def cell_tooltip(cot: str, scenario: str, cell: dict[str, Any] | None) -> str:
    if not cell:
        return f"CoT row: {cot}\nScene axis value: {scenario}\nNo examples yet"

    total = cell["total"]
    reliable = cell["reliable"]
    unknown = cell["unknown"]
    rate = reliable / total if total else 0.0
    clip_ids = "\n".join(clip_id for clip_id in cell["clip_ids"] if clip_id)
    return (
        f"CoT row: {cot}\n"
        f"Scene axis value: {scenario}\n"
        f"Reliable/total: {reliable}/{total}\n"
        f"Unreliable: {cell['unreliable']}\n"
        f"Unknown reliability: {unknown}\n"
        f"Reliability rate: {rate:.2%}\n"
        f"Clip IDs:\n{clip_ids}"
    )


def summary_tooltip(cot: str, summary: dict[str, Any] | None) -> str:
    if not summary:
        return f"CoT row: {cot}\nNo examples yet"

    total = summary["total"]
    reliable = summary["reliable"]
    unknown = summary["unknown"]
    rate = reliable / total if total else 0.0
    clip_ids = "\n".join(clip_id for clip_id in summary["clip_ids"] if clip_id)
    return (
        f"CoT row: {cot}\n"
        "Summary across benchmark entries for this CoT row\n"
        f"Reliable/total: {reliable}/{total}\n"
        f"Unreliable: {summary['unreliable']}\n"
        f"Unknown reliability: {unknown}\n"
        f"Reliability rate: {rate:.2%}\n"
        f"Clip IDs:\n{clip_ids}"
    )


def write_html_heatmap(
    output_path: Path,
    *,
    cells: dict[tuple[str, str], dict[str, Any]],
    row_summaries: dict[str, dict[str, Any]],
    cot_labels: list[str],
    scenario_labels: list[str],
    title: str,
    wrap_x: int,
    wrap_y: int,
    max_label_lines: int,
) -> None:
    table_rows: list[str] = []

    header_cells = [
        '<th class="corner">CoT action or causal factor / Scene axis value</th>',
        *[
            (
                '<th class="scenario" title="{title}"><div>{label}</div></th>'
            ).format(
                title=html.escape(scenario, quote=True),
                label=html_wrapped_label(scenario, wrap_x, max_label_lines),
            )
            for scenario in scenario_labels
        ],
        (
            '<th class="summary" title="Reliability summary across benchmark '
            'entries for this CoT row"><div>summary</div></th>'
        ),
    ]
    table_rows.append("<tr>{}</tr>".format("".join(header_cells)))

    for cot in cot_labels:
        row_cells = [
            (
                '<th class="cot" title="{title}">{label}</th>'
            ).format(
                title=html.escape(cot, quote=True),
                label=html_wrapped_label(cot, wrap_y, max_label_lines),
            )
        ]
        for scenario in scenario_labels:
            cell = cells.get((cot, scenario))
            if not cell:
                row_cells.append(
                    '<td class="cell empty" title="{tooltip}"></td>'.format(
                        tooltip=html.escape(
                            cell_tooltip(cot, scenario, None),
                            quote=True,
                        )
                    )
                )
                continue

            total = cell["total"]
            reliable = cell["reliable"]
            unknown = cell["unknown"]
            rate = reliable / total if total else 0.0
            background = reliability_color(rate)
            text_color = text_color_for_background(background)
            unknown_html = (
                f'<span class="unknown">?{unknown}</span>' if unknown else ""
            )
            row_cells.append(
                (
                    '<td class="cell filled" style="background:{background};'
                    'color:{text_color}" title="{tooltip}">'
                    '<span class="ratio">{reliable}/{total}</span>'
                    "{unknown}</td>"
                ).format(
                    background=background,
                    text_color=text_color,
                    tooltip=html.escape(
                        cell_tooltip(cot, scenario, cell),
                        quote=True,
                    ),
                    reliable=reliable,
                    total=total,
                    unknown=unknown_html,
                )
            )
        summary = row_summaries.get(cot)
        if summary:
            total = summary["total"]
            reliable = summary["reliable"]
            unknown = summary["unknown"]
            rate = reliable / total if total else 0.0
            background = reliability_color(rate)
            text_color = text_color_for_background(background)
            unknown_html = (
                f'<span class="unknown">?{unknown}</span>' if unknown else ""
            )
            row_cells.append(
                (
                    '<td class="cell summary-cell filled" '
                    'style="background:{background};color:{text_color}" '
                    'title="{tooltip}">'
                    '<span class="ratio">{reliable}/{total}</span>'
                    '<span class="rate">{rate:.1%}</span>'
                    "{unknown}</td>"
                ).format(
                    background=background,
                    text_color=text_color,
                    tooltip=html.escape(
                        summary_tooltip(cot, summary),
                        quote=True,
                    ),
                    reliable=reliable,
                    total=total,
                    rate=rate,
                    unknown=unknown_html,
                )
            )
        else:
            row_cells.append(
                '<td class="cell summary-cell empty" title="{tooltip}"></td>'.format(
                    tooltip=html.escape(summary_tooltip(cot, None), quote=True)
                )
            )
        table_rows.append("<tr>{}</tr>".format("".join(row_cells)))

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --border: #e5e7eb;
      --empty: #d9d9d9;
      --header: #f8fafc;
      --text: #111827;
      --muted: #4b5563;
    }}
    body {{
      margin: 24px;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: white;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 20px;
      font-weight: 650;
    }}
    .meta {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }}
    .swatch {{
      display: inline-block;
      width: 42px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(0, 0, 0, 0.12);
      vertical-align: -2px;
    }}
    .swatch.red {{ background: #c62828; }}
    .swatch.green {{ background: #2f855a; }}
    .swatch.gray {{ background: var(--empty); }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      max-height: calc(100vh - 140px);
    }}
    table {{
      border-collapse: separate;
      border-spacing: 0;
      font-size: 12px;
    }}
    th, td {{
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      box-sizing: border-box;
    }}
    th {{
      background: var(--header);
      font-weight: 600;
      line-height: 1.25;
    }}
    th.corner {{
      position: sticky;
      top: 0;
      left: 0;
      z-index: 4;
      min-width: 360px;
      max-width: 360px;
      padding: 10px;
      text-align: left;
    }}
    th.scenario {{
      position: sticky;
      top: 0;
      z-index: 3;
      min-width: 88px;
      max-width: 88px;
      height: 130px;
      padding: 8px 6px;
      vertical-align: bottom;
      text-align: left;
    }}
    th.scenario div {{
      max-height: 116px;
      overflow: hidden;
    }}
    th.summary {{
      position: sticky;
      top: 0;
      z-index: 3;
      min-width: 104px;
      max-width: 104px;
      height: 130px;
      padding: 8px 6px;
      vertical-align: bottom;
      text-align: left;
    }}
    th.cot {{
      position: sticky;
      left: 0;
      z-index: 2;
      min-width: 360px;
      max-width: 360px;
      padding: 8px 10px;
      text-align: left;
      background: var(--header);
    }}
    td.cell {{
      min-width: 88px;
      width: 88px;
      height: 48px;
      text-align: center;
      vertical-align: middle;
      font-weight: 700;
      line-height: 1.1;
    }}
    td.empty {{
      background: var(--empty);
    }}
    td.summary-cell {{
      min-width: 104px;
      width: 104px;
    }}
    .ratio {{
      display: block;
      font-size: 13px;
    }}
    .rate {{
      display: block;
      margin-top: 3px;
      font-size: 11px;
      font-weight: 650;
      opacity: 0.85;
    }}
    .unknown {{
      display: block;
      margin-top: 3px;
      font-size: 11px;
      font-weight: 650;
      opacity: 0.8;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="meta">{len(cot_labels)} flattened CoT rows x {len(scenario_labels)} scene-axis columns plus one summary column. Cell text is reliable/total; ?N means N examples had unknown reliability.</p>
  <div class="legend">
    <span><span class="swatch red"></span> more unreliable</span>
    <span><span class="swatch green"></span> more reliable</span>
    <span><span class="swatch gray"></span> no examples yet</span>
  </div>
  <div class="table-wrap">
    <table>
      {"".join(table_rows)}
    </table>
  </div>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    entries = load_benchmark(args.benchmark)
    if not entries:
        raise ValueError(f"No benchmark entries found in {args.benchmark}")

    output_path = args.output or default_output_path(args.benchmark)
    if args.output is None and args.format == "png":
        output_path = args.benchmark.with_suffix(".cot_reliability_heatmap.png")
    summary_csv = args.summary_csv or default_summary_path(output_path)

    cells = aggregate(
        entries,
        scenario_axis=args.scenario_axis,
        cot_axis=args.cot_axis,
    )
    row_summaries = aggregate_row_summaries(entries, cot_axis=args.cot_axis)
    cot_labels = order_labels(
        [
            facet["row_label"]
            for entry in entries
            for facet in cot_axis_facets(entry, args.cot_axis)
        ],
        args.order,
    )
    scenario_labels = order_scene_labels(
        [
            scenario
            for entry in entries
            for scenario in scenario_labels_for_entry(entry, args.scenario_axis)
        ],
        args.order,
    )

    title = args.title or (
        f"CoT Reliability Coverage: {args.benchmark.name} "
        f"({args.cot_axis} x {args.scenario_axis})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    output_format = resolved_output_format(output_path, args.format)
    if output_format == "png":
        matrix, annotations = build_matrices(
            cells,
            cot_labels=cot_labels,
            scenario_labels=scenario_labels,
        )
        plot_heatmap(
            output_path,
            matrix=matrix,
            annotations=annotations,
            cot_labels=cot_labels,
            scenario_labels=scenario_labels,
            title=title,
            wrap_x=args.wrap_x,
            wrap_y=args.wrap_y,
            max_label_lines=args.max_label_lines,
            dpi=args.dpi,
        )
    else:
        write_html_heatmap(
            output_path,
            cells=cells,
            row_summaries=row_summaries,
            cot_labels=cot_labels,
            scenario_labels=scenario_labels,
            title=title,
            wrap_x=args.wrap_x,
            wrap_y=args.wrap_y,
            max_label_lines=args.max_label_lines,
        )
    write_summary_csv(summary_csv, cells=cells)

    total_examples = sum(cell["total"] for cell in cells.values())
    reliable_examples = sum(cell["reliable"] for cell in cells.values())
    unknown_examples = sum(cell["unknown"] for cell in cells.values())
    print(f"Loaded {len(entries)} entries")
    print(
        f"Plotted {len(cot_labels)} flattened CoT rows x "
        f"{len(scenario_labels)} scene-axis columns"
    )
    print(
        f"Cell memberships: {reliable_examples}/{total_examples} reliable; "
        f"{unknown_examples} unknown"
    )
    print(f"Wrote heatmap: {output_path}")
    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
