#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Batch CoT analysis runner for tutorial_alpamayo scenes.

Default behavior:
- Scenes: 1..50
- Sampling: every_nth=10
- Input ASLs: <scene_dir>/rollouts/**/*.asl
- Output per ASL: <scene_dir>/aggregate/cot_analysis/*.cot_consistency.json
- Resume:
  - Per-ASL: delegated to `cot_analysis --resume`
  - Per-scene: tracked via `scene_resume.json`

API key:
- You can fill `API_KEY_PLACEHOLDER` later, or pass `--api-key`, or set GEMINI_API_KEY.
- If no key is provided, `cot_analysis` runs in dry-run mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Fill this later if you want to hardcode your key locally.
API_KEY_PLACEHOLDER = "REDACTED-API-KEY"


@dataclass
class SceneRunState:
    scene_dir: Path
    state_path: Path
    completed_asls: set[str]
    failed_asls: dict[str, str]
    errored_entries: dict[str, int] = field(default_factory=dict)  # asl_rel -> count

    @classmethod
    def load(cls, scene_dir: Path, state_path: Path) -> "SceneRunState":
        if state_path.exists():
            data = json.loads(state_path.read_text())
            completed = set(data.get("completed_asls", []))
            failed = data.get("failed_asls", {})
            if not isinstance(failed, dict):
                failed = {}
            errored = data.get("errored_entries", {})
            if not isinstance(errored, dict):
                errored = {}
            return cls(
                scene_dir=scene_dir,
                state_path=state_path,
                completed_asls=completed,
                failed_asls=failed,
                errored_entries=errored,
            )
        return cls(
            scene_dir=scene_dir,
            state_path=state_path,
            completed_asls=set(),
            failed_asls={},
            errored_entries={},
        )

    def mark_completed(self, asl_path: Path, error_count: int = 0) -> None:
        rel = str(asl_path.relative_to(self.scene_dir))
        self.completed_asls.add(rel)
        self.failed_asls.pop(rel, None)
        if error_count > 0:
            self.errored_entries[rel] = error_count
        else:
            self.errored_entries.pop(rel, None)

    def mark_failed(self, asl_path: Path, reason: str) -> None:
        rel = str(asl_path.relative_to(self.scene_dir))
        self.failed_asls[rel] = reason

    def has_errored_entries(self) -> bool:
        """Check if any completed ASL has entries with API errors."""
        return any(count > 0 for count in self.errored_entries.values())

    def save(self, all_scene_asls: list[Path]) -> None:
        all_rel = [str(p.relative_to(self.scene_dir)) for p in all_scene_asls]
        total_errors = sum(self.errored_entries.values())
        data = {
            "scene": self.scene_dir.name,
            "completed_asls": sorted(self.completed_asls),
            "failed_asls": self.failed_asls,
            "errored_entries": self.errored_entries,
            "total_errored_entries": total_errors,
            "total_asls": len(all_rel),
            "is_scene_complete": (
                len(self.completed_asls) == len(all_rel) and total_errors == 0
            ),
        }
        self.state_path.write_text(json.dumps(data, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch CoT analysis for tutorial_alpamayo with scene/asl-level resume"
    )
    parser.add_argument(
        "--tutorial-root",
        type=Path,
        default=Path("tutorial_alpamayo"),
        help="Path to tutorial_alpamayo root directory",
    )
    parser.add_argument(
        "--scene-start",
        type=int,
        default=1,
        help="Start scene index (inclusive)",
    )
    parser.add_argument(
        "--scene-end",
        type=int,
        default=50,
        help="End scene index (inclusive)",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=10,
        help="Analyze every Nth timestep inside each ASL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model passed to cot_analysis",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key. Overrides placeholder and GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls passed to cot_analysis",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="camera_front_wide_120fov",
        help="Camera logical ID",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Text-only mode (do not send images to Gemini)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level passed to cot_analysis",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="cot_analysis",
        help="Python module name for CoT analyzer",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-process entries that have API errors (e.g. 429 RESOURCE_EXHAUSTED). "
             "Un-marks completed scenes that have errors and retries them.",
    )
    return parser.parse_args()


def find_scene_dirs(tutorial_root: Path, scene_start: int, scene_end: int) -> list[Path]:
    scene_dirs: list[Path] = []
    for scene_dir in sorted(tutorial_root.glob("scene_*")):
        if not scene_dir.is_dir():
            continue
        name = scene_dir.name
        # Expected format: scene_<num>_...
        parts = name.split("_", 2)
        if len(parts) < 2:
            continue
        try:
            scene_idx = int(parts[1])
        except ValueError:
            continue
        if scene_start <= scene_idx <= scene_end:
            scene_dirs.append(scene_dir)
    return scene_dirs


def build_output_path(scene_dir: Path, asl_path: Path) -> Path:
    rel = asl_path.relative_to(scene_dir / "rollouts")
    stem = "__".join(rel.with_suffix("").parts)
    return scene_dir / "aggregate" / "cot_analysis" / f"{stem}.cot_consistency.json"


def count_errors_in_output(output_path: Path) -> int:
    """Count entries with API errors in an output JSON file."""
    if not output_path.exists():
        return 0
    try:
        data = json.loads(output_path.read_text())
        results = data.get("results", []) if isinstance(data, dict) else data
        return sum(
            1 for r in results
            if isinstance(r, dict) and "error" in r.get("evaluation", r)
        )
    except (json.JSONDecodeError, KeyError):
        return 0


# Exit code used by cot_analysis module when quota is exhausted
_QUOTA_EXHAUSTED_EXIT_CODE = 3


def run_single_asl(
    args: argparse.Namespace, asl_path: Path, output_path: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        args.module,
        "--asl_glob",
        str(asl_path),
        "--output",
        str(output_path),
        "--every_nth",
        str(args.every_nth),
        "--model",
        args.model,
        "--delay",
        str(args.delay),
        "--camera_id",
        args.camera_id,
        "--log_level",
        args.log_level,
        "--resume",
    ]

    if args.retry_errors:
        cmd.append("--retry_errors")

    if args.no_images:
        cmd.append("--no_images")

    if args.api_key:
        cmd.extend(["--api_key", args.api_key])

    env = os.environ.copy()
    extra_pythonpath = [
        str(Path("src")),
        str(Path("src/tools")),
        str(Path("src/utils")),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        extra_pythonpath + ([existing_pythonpath] if existing_pythonpath else [])
    )

    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def main() -> int:
    args = parse_args()

    if args.scene_end < args.scene_start:
        raise ValueError("scene-end must be >= scene-start")

    tutorial_root = args.tutorial_root
    if not tutorial_root.exists():
        raise FileNotFoundError(f"Tutorial root not found: {tutorial_root}")

    # Resolve API key precedence: CLI > placeholder > env in cot_analysis
    if args.api_key is None and API_KEY_PLACEHOLDER.strip():
        args.api_key = API_KEY_PLACEHOLDER.strip()

    scene_dirs = find_scene_dirs(tutorial_root, args.scene_start, args.scene_end)
    if not scene_dirs:
        print("No matching scene directories found.")
        return 1

    total_scenes = len(scene_dirs)
    total_asls = 0
    succeeded_asls = 0
    failed_asls = 0
    skipped_asls = 0

    print(
        f"Running CoT analysis for scenes {args.scene_start}-{args.scene_end} "
        f"(found {total_scenes} scenes), every_nth={args.every_nth}"
    )
    if args.retry_errors:
        print("Retry-errors mode: will re-process entries with API errors.")
    if not args.api_key:
        print("No API key provided. This will run in dry-run mode until you fill it in.")

    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        rollouts_dir = scene_dir / "rollouts"
        scene_output_dir = scene_dir / "aggregate" / "cot_analysis"
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        state_path = scene_output_dir / "scene_resume.json"

        scene_asls = sorted(rollouts_dir.rglob("*.asl")) if rollouts_dir.exists() else []
        total_asls += len(scene_asls)

        print(f"\n[{scene_idx}/{total_scenes}] Scene: {scene_dir.name} | ASLs: {len(scene_asls)}")

        state = SceneRunState.load(scene_dir=scene_dir, state_path=state_path)

        if not scene_asls:
            print("  No ASL files found, skipping scene.")
            state.save(scene_asls)
            continue

        quota_exhausted = False
        for asl_idx, asl_path in enumerate(scene_asls, start=1):
            rel_asl = str(asl_path.relative_to(scene_dir))
            output_path = build_output_path(scene_dir, asl_path)

            # In retry-errors mode, check if the output has errored entries
            if rel_asl in state.completed_asls:
                if args.retry_errors:
                    error_count = count_errors_in_output(output_path)
                    if error_count > 0:
                        print(
                            f"  [{asl_idx}/{len(scene_asls)}] RETRY {rel_asl} "
                            f"({error_count} errored entries)"
                        )
                    else:
                        skipped_asls += 1
                        print(f"  [{asl_idx}/{len(scene_asls)}] SKIP completed (clean): {rel_asl}")
                        continue
                else:
                    skipped_asls += 1
                    print(f"  [{asl_idx}/{len(scene_asls)}] SKIP completed: {rel_asl}")
                    continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if rel_asl not in state.completed_asls:
                print(f"  [{asl_idx}/{len(scene_asls)}] RUN {rel_asl}")
            print(f"    -> output: {output_path.relative_to(scene_dir)}")

            proc = run_single_asl(args, asl_path=asl_path, output_path=output_path)

            if proc.returncode == _QUOTA_EXHAUSTED_EXIT_CODE:
                # Quota exhausted — mark what we have and stop everything
                error_count = count_errors_in_output(output_path)
                state.mark_completed(asl_path, error_count=error_count)
                state.save(scene_asls)
                print(
                    f"    status: QUOTA EXHAUSTED (exit code {_QUOTA_EXHAUSTED_EXIT_CODE}). "
                    "Stopping all further scenes."
                )
                quota_exhausted = True
                failed_asls += 1
                break
            elif proc.returncode == 0:
                error_count = count_errors_in_output(output_path)
                state.mark_completed(asl_path, error_count=error_count)
                succeeded_asls += 1
                if error_count > 0:
                    print(f"    status: OK (but {error_count} entries still have errors)")
                else:
                    print("    status: OK")
            else:
                failed_asls += 1
                reason = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout) else "unknown error"
                state.mark_failed(asl_path, reason)
                print(f"    status: FAIL ({reason})")

            state.save(scene_asls)

        if quota_exhausted:
            break

    print("\n=== Batch CoT analysis summary ===")
    print(f"Scenes processed: {total_scenes}")
    print(f"Total ASLs found: {total_asls}")
    print(f"Succeeded this run: {succeeded_asls}")
    print(f"Failed this run: {failed_asls}")
    print(f"Skipped this run: {skipped_asls}")
    print("Per-scene resume state: <scene>/aggregate/cot_analysis/scene_resume.json")

    return 0 if failed_asls == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
