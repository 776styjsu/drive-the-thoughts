# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Minimal OpenAI-compatible LLM client for CoT-consistency judging.

This is the trimmed, dependency-light core shared by the in-loop runtime
``ConsistencyMonitor`` and any other caller that cannot depend on the
``alpasim-tools`` package (``alpasim-tools`` depends on ``alpasim_runtime``, so
the runtime cannot import it back). It mirrors the provider table and the
JSON-parsing behaviour of ``cot_analysis/__main__.py`` but drops the per-call
timing instrumentation. ``openai`` is imported lazily so importing this module
never requires the SDK.

The heavier offline CLI (``cot_analysis/__main__.py``) keeps its own richer
``call_llm``; this module is intentionally a separate, leaner implementation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fixed seed for deterministic decoding where the backend honours it.
DEFAULT_SEED = 42

# institutional GenAI gateway (OpenAI-compatible gateway running Kimi K2.5)
DEFAULT_BASE_URL = "https://genai-gateway.example.edu/api"
DEFAULT_MODEL = "Kimi K2.5"
QWEN3_4B_FP8_MODEL = "Qwen/Qwen3-4B-FP8"
QWEN35_4B_FP8_MODEL = "RedHatAI/Qwen3.5-4B-FP8-dynamic"
QWEN3_LOCAL_BASE_URL = "http://localhost:8000/v1"

# Selectable model backends. Mirrors cot_analysis/__main__.py PROVIDERS so the
# online monitor and the offline CLI judge identically when pointed at the same
# backend. Each provider resolves its own API key / base URL from the
# environment (loaded from .env).
PROVIDERS: dict[str, dict] = {
    "gateway": {
        "label": "Institutional gateway (Kimi K2.5)",
        "model": DEFAULT_MODEL,
        "api_key_env": "GENAI_GATEWAY_KEY",
        "base_url_env": "GENAI_GATEWAY_BASE_URL",
        "base_url": DEFAULT_BASE_URL,
        "temperature": 0,
        "supports_images": True,
        "extra_params": {},
    },
    "openai": {
        "label": "OpenAI GPT-5.5 (high reasoning)",
        "model": "gpt-5.5",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": None,  # use the OpenAI SDK default endpoint
        # GPT-5.5 is a reasoning model: it rejects temperature != 1 (default),
        # so we omit temperature and rely on the fixed seed for determinism.
        "temperature": None,
        "supports_images": True,
        "extra_params": {"reasoning_effort": "high"},
    },
    "qwen3_4b_fp8": {
        "label": "Local Qwen3-4B-FP8 via vLLM (non-thinking)",
        "model": QWEN3_4B_FP8_MODEL,
        "api_key_env": "QWEN3_API_KEY",
        "base_url_env": "QWEN3_BASE_URL",
        "base_url": QWEN3_LOCAL_BASE_URL,
        "default_api_key": "EMPTY",
        "temperature": 0.0,
        "supports_images": False,
        "extra_params": {
            "max_tokens": 1024,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    },
    "qwen35_4b_fp8": {
        "label": "Local Qwen3.5-4B-FP8 via vLLM (non-thinking)",
        "model": QWEN35_4B_FP8_MODEL,
        "api_key_env": "QWEN35_API_KEY",
        "base_url_env": "QWEN35_BASE_URL",
        "base_url": QWEN3_LOCAL_BASE_URL,
        "default_api_key": "EMPTY",
        "temperature": 0.0,
        "supports_images": False,
        "extra_params": {
            "max_tokens": 1024,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        },
    },
}

# Some OpenAI-compatible gateways reject the response_format parameter. We
# attempt JSON mode first and disable it for the rest of the run if rejected.
_USE_JSON_MODE = True


# =============================================================================
# .env loading (no external dependency) — mirrors cot_analysis/__main__.py
# =============================================================================


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file (existing vars take precedence)."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def find_and_load_dotenv() -> None:
    """Search for a .env file from the cwd and module dir upward, then load it."""
    seen: set[Path] = set()
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        current = base
        while True:
            candidate = current / ".env"
            if candidate not in seen and candidate.exists():
                _load_dotenv(candidate)
                return
            seen.add(candidate)
            if current.parent == current:
                break
            current = current.parent


# =============================================================================
# Provider / client resolution
# =============================================================================


def resolve_provider(
    provider: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    load_dotenv: bool = True,
) -> dict:
    """Resolve a provider name into concrete client/request settings.

    Reads the API key and base URL from the provider's environment variables
    (optionally loading a ``.env`` file first), applying explicit overrides when
    given. Returns a dict with keys: ``model``, ``api_key``, ``base_url``,
    ``temperature``, ``extra_params``, ``supports_images``, ``label``.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown judge provider '{provider}'. Available: {sorted(PROVIDERS)}"
        )
    if load_dotenv:
        find_and_load_dotenv()

    spec = PROVIDERS[provider]
    resolved_api_key = (
        api_key or os.environ.get(spec["api_key_env"]) or spec.get("default_api_key")
    )
    resolved_base_url = (
        base_url or os.environ.get(spec["base_url_env"]) or spec["base_url"]
    )
    return {
        "label": spec["label"],
        "model": model or spec["model"],
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "temperature": spec.get("temperature", 0),
        "extra_params": dict(spec.get("extra_params") or {}),
        "supports_images": spec.get("supports_images", True),
    }


def build_client(api_key: str, base_url: str | None) -> Any:
    """Create an OpenAI client pointed at the (possibly self-hosted) endpoint."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


# =============================================================================
# Inference + parsing
# =============================================================================


def call_llm(
    client: Any,
    model_name: str,
    prompt: str,
    *,
    seed: int = DEFAULT_SEED,
    temperature: float | None = 0,
    extra_params: dict | None = None,
) -> str:
    """Send the prompt to the model and return the response text.

    Decoding is deterministic via a fixed seed where supported. ``temperature``
    is sent only when not None (reasoning models reject non-default values).
    ``extra_params`` carries provider-specific request fields. The call streams
    and accumulates the content delta, which the institutional Open WebUI gateway requires
    and other OpenAI-compatible servers (vLLM, OpenAI) support.
    """
    global _USE_JSON_MODE

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for autonomous vehicle reasoning "
                "systems. Respond with ONLY a single JSON object, no markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    def _create(use_json: bool) -> str:
        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "seed": seed,
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if extra_params:
            kwargs.update(extra_params)
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        parts: list[str] = []
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                parts.append(delta.content)
        return "".join(parts)

    try:
        return _create(_USE_JSON_MODE)
    except Exception as exc:
        # The gateway may reject response_format; retry once without it and
        # disable JSON mode for the remainder of the run.
        if _USE_JSON_MODE:
            try:
                result = _create(False)
                _USE_JSON_MODE = False
                logger.warning(
                    "Disabling JSON response_format (server rejected it): %s", exc
                )
                return result
            except Exception as exc2:  # pragma: no cover - surfaced to caller
                return json.dumps({"error": str(exc2)})
        return json.dumps({"error": str(exc)})


def parse_response(response_text: str) -> dict:
    """Parse a structured JSON response from the model."""
    if not response_text:
        return {"parse_error": True, "raw_response": response_text}
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                return {"parse_error": True, "raw_response": response_text}
        if not isinstance(parsed, dict):
            return {"parse_error": True, "raw_response": response_text}
        return parsed
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_response": response_text}


def score_from_evaluation(
    parsed: dict, dimension: str = "cot_output_alignment"
) -> float | None:
    """Extract the 1-5 alignment score from a parsed judge response.

    Returns None when the response carries no usable numeric score (parse
    error, missing dimension, or a non-numeric score).
    """
    if not isinstance(parsed, dict) or parsed.get("parse_error") or parsed.get("error"):
        return None
    dim_data = parsed.get(dimension)
    if not isinstance(dim_data, dict) or "score" not in dim_data:
        return None
    try:
        return float(dim_data["score"])
    except (TypeError, ValueError):
        return None
