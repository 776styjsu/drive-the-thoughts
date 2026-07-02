# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import builtins

import pytest

from alpasim_utils.cot_consistency.llm_judge import build_client


def test_build_client_explains_missing_openai_sdk(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'", name="openai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="uv run --extra llm"):
        build_client("EMPTY", "http://localhost:8000/v1")
