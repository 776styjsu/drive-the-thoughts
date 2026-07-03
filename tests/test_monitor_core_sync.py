# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from tools.sync_monitor_core import MIRROR_ROOT, drifted_files, mirrored_files


def test_alpasim_mirror_matches_source_of_truth() -> None:
    drifted = drifted_files()
    assert not drifted, (
        "src/alpasim_utils is the source of truth for the shared monitor core; "
        f"these files diverged in the alpasim mirror ({MIRROR_ROOT}): "
        f"{[str(path) for path in drifted]}. "
        "Run: uv run python tools/sync_monitor_core.py --fix"
    )


def test_mirror_manifest_covers_the_monitor_core() -> None:
    names = {path.as_posix() for path in mirrored_files()}
    # Guard against the comparison silently going empty after a repo move.
    assert "consistency.py" in names
    assert "cot_consistency/llm_judge.py" in names
    assert len(names) >= 10
