"""Copy benchmark frame folders into one ordered review directory."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from benchmark_analysis import extract_entries, load_json

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = (
    "metadata.json",
    "additional_info.json",
    "trajectory_plot.png",
    "trajectory_plot_geometry.json",
    "camera_front_wide_120fov.jpg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the extracted-frame folder for each benchmark entry into "
            "a single directory for side-by-side review."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=REPO_ROOT / "data" / "benchmark" / "benchmark.json",
        help="Benchmark JSON to export. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark_side_by_side",
        help="Directory to receive the ordered folder copies.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to resolve relative source_scene_file/scene_folder values.",
    )
    parser.add_argument(
        "--order",
        choices=("benchmark", "clip_id"),
        default="benchmark",
        help="Folder order to use. Default preserves the benchmark JSON order.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before exporting.",
    )
    parser.add_argument(
        "--no-fill-missing-trajectory-plot",
        action="store_true",
        help="Do not generate trajectory_plot.png from metadata.json when it is missing.",
    )
    return parser.parse_args()


def resolve_frame_dir(entry: Dict[str, Any], benchmark: Path, repo_root: Path) -> Path:
    """Locate the per-clip frame directory for one benchmark entry.

    Prefers the artifact layout (the directory holding ``source_scene_file``,
    normally ``data/media/scenes/<clip_id>/``) and falls back to the original
    workspace layout (``<scene_folder>/<clip_id>/``).
    """
    clip_id = entry.get("clip_id")
    if not clip_id:
        raise ValueError(f"Benchmark entry is missing clip_id: {entry}")

    candidates = []
    source_scene_file = entry.get("source_scene_file")
    if source_scene_file:
        source_path = Path(source_scene_file)
        if source_path.is_absolute():
            candidates.append(source_path.parent)
        else:
            candidates.append((repo_root / source_path).parent)
            candidates.append((benchmark.parent / source_path).parent)

    scene_folder = entry.get("scene_folder")
    if scene_folder:
        rel_frame_dir = Path(scene_folder) / str(clip_id)
        if rel_frame_dir.is_absolute():
            candidates.append(rel_frame_dir)
        else:
            candidates.append(repo_root / rel_frame_dir)
            candidates.append(benchmark.parent / rel_frame_dir)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    formatted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find frame folder for {clip_id}. Tried:\n  {formatted}"
    )


def safe_folder_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("_")


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def save_trajectory_plot_from_metadata(metadata_path: Path, output_path: Path) -> bool:
    with metadata_path.open() as f:
        metadata = json.load(f)

    trajectory_xy = metadata.get("trajectory_xy_rig_frame") or []
    if not trajectory_xy:
        return False

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(4.0, 6.0), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    px = [-pt["ry"] for pt in trajectory_xy]
    py = [pt["rx"] for pt in trajectory_xy]
    ax.plot(px, py, "o-", color="#4ecdc4", linewidth=2.5, markersize=5, alpha=0.9)

    ego_rect = Rectangle(
        (-0.9, -2.25),
        1.8,
        4.5,
        facecolor="#ffd93d",
        edgecolor="#ffffff",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(ego_rect)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 80)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3, color="#555555")
    ax.tick_params(colors="#ffffff", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.tight_layout()
    fig.savefig(str(output_path), facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return True


def export_entry(
    index: int,
    entry: Dict[str, Any],
    source: Path,
    output: Path,
    fill_missing_trajectory_plot: bool,
) -> Dict[str, Any]:
    dest_name = f"{index:03d}__{safe_folder_name(source)}"
    dest = output / dest_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(str(source), str(dest), copy_function=shutil.copy2)

    generated_files = []
    plot_path = dest / "trajectory_plot.png"
    metadata_path = dest / "metadata.json"
    if fill_missing_trajectory_plot and not plot_path.exists() and metadata_path.exists():
        if save_trajectory_plot_from_metadata(metadata_path, plot_path):
            generated_files.append("trajectory_plot.png")

    missing_expected_files = [
        name for name in EXPECTED_FILES if not (dest / name).exists()
    ]

    return {
        "order": index,
        "clip_id": entry["clip_id"],
        "scene_folder": entry.get("scene_folder"),
        "source": str(source),
        "destination": str(dest.resolve()),
        "generated_files": generated_files,
        "missing_expected_files": missing_expected_files,
        "file_count": count_files(dest),
    }


def write_readme(output: Path, benchmark: Path, order: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Benchmark Frame Folder Export",
        "",
        f"- Benchmark: `{benchmark}`",
        f"- Order: `{order}`",
        f"- Entries: `{len(rows)}`",
        "",
        "Folders are prefixed with a zero-padded row number so normal file browsers",
        "show them in the same order as the benchmark export.",
        "",
        "If a source folder lacked `trajectory_plot.png`, this exporter generated",
        "that plot from `metadata.json` and recorded it in `manifest.json`.",
        "",
        "See `manifest.json` for the source path and destination path of each copy.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    benchmark = args.benchmark.resolve()
    output = args.output.resolve()
    repo_root = args.repo_root.resolve()

    entries = extract_entries(load_json(benchmark))
    if args.order == "clip_id":
        entries = sorted(entries, key=lambda entry: str(entry.get("clip_id", "")))

    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = []
    fill_missing_trajectory_plot = not args.no_fill_missing_trajectory_plot
    for index, entry in enumerate(entries, start=1):
        source = resolve_frame_dir(entry, benchmark, repo_root)
        manifest.append(
            export_entry(index, entry, source, output, fill_missing_trajectory_plot)
        )

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(output, benchmark, args.order, manifest)

    print(f"Exported {len(manifest)} frame folders to {output}")


if __name__ == "__main__":
    main()
