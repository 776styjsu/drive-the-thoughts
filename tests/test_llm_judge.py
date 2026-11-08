# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import builtins

import pytest

from alpasim_utils.cot_consistency.llm_judge import build_client, resolve_provider


def test_build_client_explains_missing_openai_sdk(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'", name="openai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="uv run --extra llm"):
        build_client("EMPTY", "http://localhost:8000/v1")


def test_deepseek_provider_requests_thinking_mode(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    provider = resolve_provider("deepseek", load_dotenv=False)

    assert provider["api_key"] == "test-key"
    assert provider["base_url"] == "https://api.deepseek.com"
    # Thinking mode ignores temperature, so none is sent.
    assert provider["temperature"] is None
    assert provider["extra_params"]["reasoning_effort"] == "high"
    assert provider["extra_params"]["extra_body"] == {"thinking": {"type": "enabled"}}


def test_deepseek_openrouter_provider_pins_deepseek_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    provider = resolve_provider("deepseek_openrouter", load_dotenv=False)

    assert provider["api_key"] == "test-key"
    assert provider["base_url"] == "https://openrouter.ai/api/v1"
    assert provider["model"] == "deepseek/deepseek-v4.1-flash-1017"
    assert provider["temperature"] is None
    extra_body = provider["extra_params"]["extra_body"]
    assert extra_body["reasoning"] == {"effort": "high"}
    # Quantized third-party hosts must never serve the judge.
    assert extra_body["provider"] == {"order": ["deepseek"], "allow_fallbacks": False}


def test_claude_opus_openrouter_provider_pins_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    provider = resolve_provider("claude_opus_openrouter", load_dotenv=False)

    assert provider["model"] == "anthropic/claude-opus-5.5"
    # Opus 5.5 rejects sampling parameters, so none is sent.
    assert provider["temperature"] is None
    extra_body = provider["extra_params"]["extra_body"]
    assert extra_body["reasoning"] == {"effort": "high"}
    assert extra_body["provider"] == {"only": ["anthropic"], "allow_fallbacks": False}


def test_jev_openrouter_provider_pins_a_version(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    provider = resolve_provider("jev_openrouter", load_dotenv=False)

    # A versioned ID, not the ~typesafe/jev-latest alias, which moves.
    assert provider["model"] == "typesafe/jev-1.14-20261014"
    assert provider["base_url"] == "https://openrouter.ai/api/v1"
    assert provider["temperature"] is None
    assert provider["extra_params"] == {}
