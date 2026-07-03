#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Keep the AlpaSim mirror of the shared monitor core in sync with src/.

``src/alpasim_utils`` is the single source of truth for the monitor code that
both tiers execute: the rule-based consistency core and the ``cot_consistency``
judge subpackage. The AlpaSim workspace ships a byte-identical copy under
``alpasim/src/utils/alpasim_utils`` so the simulator tier stays a standalone
uv workspace (its eval scorer and online ``ConsistencyMonitor`` import the
copy). Every ``.py`` file in the root package must therefore match its mirror;
the mirror may additionally hold simulator-only modules, which are ignored.

Edit at the root, then run:

    uv run python tools/sync_monitor_core.py --fix

``tests/test_monitor_core_sync.py`` runs the same comparison under pytest, so
one-sided edits fail the test suite instead of silently forking the online
monitor from the offline judge.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "alpasim_utils"
MIRROR_ROOT = REPO_ROOT / "alpasim" / "src" / "utils" / "alpasim_utils"


def mirrored_files() -> list[Path]:
    """All source files, relative to SOURCE_ROOT."""
    return sorted(
        path.relative_to(SOURCE_ROOT)
        for path in SOURCE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def drifted_files() -> list[Path]:
    """Source files whose mirror is missing or differs byte-wise."""
    drifted = []
    for relative in mirrored_files():
        mirror = MIRROR_ROOT / relative
        if not mirror.is_file() or not filecmp.cmp(
            SOURCE_ROOT / relative, mirror, shallow=False
        ):
            drifted.append(relative)
    return drifted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or restore the alpasim mirror of src/alpasim_utils."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="List drifted files and exit 1 if the mirror is out of sync.",
    )
    mode.add_argument(
        "--fix",
        action="store_true",
        help="Copy drifted files from src/ into the alpasim mirror.",
    )
    args = parser.parse_args()

    drifted = drifted_files()
    if not drifted:
        print(f"mirror in sync ({len(mirrored_files())} files)")
        return 0

    for relative in drifted:
        if args.fix:
            target = MIRROR_ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE_ROOT / relative, target)
            print(f"synced: {relative}")
        else:
            print(f"drift:  {relative}")

    if args.check:
        print(
            "mirror out of sync; run: uv run python tools/sync_monitor_core.py --fix",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
