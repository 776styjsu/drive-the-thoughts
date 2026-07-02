# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""JSON loading and per-clip entry extraction.

The artifact's JSON files come in a few container shapes: a bare list of
entries, an object with the list under ``results``/``entries`` (benchmark and
judge outputs), or a single entry object with a ``clip_id``. These helpers
normalize all of them to ``list[dict]`` so callers never re-implement the
shape sniffing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Keys probed, in order, for the entry list inside an object container.
ENTRY_LIST_KEYS = ("results", "entries", "examples", "data")


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return its contents."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_entries(data: Any) -> list[dict]:
    """Extract per-clip entries from any of the artifact's JSON containers."""
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object or list")

    if data.get("clip_id"):
        return [data]

    for key in ENTRY_LIST_KEYS:
        entries = data.get(key)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]

    raise ValueError(
        "Unrecognized JSON format; expected clip_id, "
        + ", or ".join(ENTRY_LIST_KEYS)
    )


def load_entries(path: str | Path) -> list[dict]:
    """Load a JSON file and extract its per-clip entries in one step."""
    return extract_entries(load_json(path))


def index_by_clip_id(entries: list[dict]) -> dict[str, dict]:
    """Map ``clip_id -> entry``, dropping entries without a usable clip_id."""
    index: dict[str, dict] = {}
    for entry in entries:
        clip_id = str(entry.get("clip_id", "")).strip()
        if clip_id:
            index[clip_id] = entry
    return index
