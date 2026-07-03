# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prompt template registry for the CoT-consistency judge.

A prompt is any callable ``build_prompt(cot_text, traj_features) -> str``.
Three sources are supported, resolved in this order:

1. A path to a ``.py`` file defining ``build_prompt`` (one-off experiments).
2. A registered builder in :data:`EXTERNAL_BUILDERS` — prompts that live in
   ``alpasim_utils.cot_consistency`` so the in-loop runtime monitor can use
   them without importing this package.
3. An auto-discovered ``prompt_<name>.py`` module in this package (``default``
   aliases ``prompt.py``). Adding a new template needs no registry edits: drop
   in the file and pass ``--prompt <name>``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

from alpasim_utils.cot_consistency import build_center_of_lane_prompt

PromptBuilder = Callable[[str, dict], str]

#: Prompt names whose module stem does not follow the prompt_<name>.py pattern.
PROMPT_ALIASES = {
    "default": "prompt",
}

#: Prompt builders implemented outside this package (shared with the runtime).
EXTERNAL_BUILDERS: dict[str, PromptBuilder] = {
    "center_of_lane": build_center_of_lane_prompt,
    "center_of_lane_v5": build_center_of_lane_prompt,
}

_PROMPT_DIR = Path(__file__).resolve().parent
_BUILDER_CACHE: dict[str, PromptBuilder] = {}


def discover_prompt_names() -> list[str]:
    """List selectable prompt names from all three sources."""
    names = set(PROMPT_ALIASES) | set(EXTERNAL_BUILDERS)
    for path in _PROMPT_DIR.glob("prompt_*.py"):
        names.add(path.stem[len("prompt_") :])
    return sorted(names)


def _load_build_prompt(module_path: Path) -> PromptBuilder:
    """Import a prompt module from a file path and return its build_prompt."""
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_prompt"):
        raise AttributeError(
            f"{module_path} has no build_prompt(cot_text, traj_features)"
        )
    return module.build_prompt


def resolve_prompt_builder(name: str) -> PromptBuilder:
    """Resolve a ``--prompt`` value to a ``build_prompt`` callable."""
    if name in _BUILDER_CACHE:
        return _BUILDER_CACHE[name]

    path_candidate = Path(name)
    if path_candidate.suffix == ".py":
        if not path_candidate.exists():
            raise FileNotFoundError(f"Prompt file not found: {name}")
        builder = _load_build_prompt(path_candidate)
    elif name in EXTERNAL_BUILDERS:
        builder = EXTERNAL_BUILDERS[name]
    else:
        stem = PROMPT_ALIASES.get(name, f"prompt_{name}")
        module_path = _PROMPT_DIR / f"{stem}.py"
        if not module_path.exists():
            raise ValueError(
                f"Unknown prompt '{name}'. Available: "
                f"{', '.join(discover_prompt_names())}, or a path to a .py file "
                f"defining build_prompt()."
            )
        builder = _load_build_prompt(module_path)

    _BUILDER_CACHE[name] = builder
    return builder
