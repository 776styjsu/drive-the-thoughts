#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Run the CoT judge through OpenRouter's Batch API.

Builds the same judge requests as ``cot_analysis`` for each variant, submits
them as one OpenRouter batch (billed at half the synchronous price), waits for
it to finish, and writes one ``cot_analysis``-format result file per variant
(``<label>.<variant>.run_NNN.json``), so the evaluation tools read them
unchanged.

The provider must point at OpenRouter. Its ``extra_body.provider`` becomes the
batch-level routing; the other request fields go into every request body.
Streaming options are dropped because batch requests cannot stream, so the
results carry token usage but no per-request latency.

The batch ID is saved before polling starts. If the waiting process stops,
rerun the same command with ``--batch-id`` to collect the results.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from alpasim_utils.cot_consistency import DEFAULT_SEED, parse_response, resolve_provider
from alpasim_utils.cot_consistency.llm_judge import JUDGE_SYSTEM_PROMPT
from cot_analysis.__main__ import (
    _judge_run_config,
    _resolve_variant_settings,
    build_parser,
)
from cot_analysis.benchmark_source import (
    extract_entries_from_benchmark,
    filter_unreliable_benchmark_cots,
)
from cot_analysis.pipeline import (
    aggregate_results,
    build_output_data,
    prepare_judge_input,
    result_is_complete,
)

logger = logging.getLogger(__name__)

ENDPOINT = "/v1/chat/completions"
FINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
# A new batch can 404 for a short while after creation; rate limits, server
# errors, and dropped connections are also worth riding out while polling.
RETRYABLE_STATUS_CODES = {404, 429, 500, 502, 503, 504}
MAX_CONSECUTIVE_POLL_FAILURES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--benchmark-source-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--provider",
        default="claude_opus_openrouter",
        help="Judge provider config to send. Default: %(default)s",
    )
    parser.add_argument(
        "--label",
        default="opus",
        help="Output file prefix. Default: %(default)s",
    )
    parser.add_argument(
        "--variants",
        default="llm,f_llm_map_graph",
        help="Comma-separated variants. Default: %(default)s",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
        help="NNN in the output names. Default: %(default)s",
    )
    parser.add_argument("--include-unreliable-cot", action="store_true")
    parser.add_argument(
        "--limit-entries",
        type=int,
        default=None,
        help="Judge only the first N entries per variant (smoke tests).",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Collect this already-submitted batch instead of submitting.",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_variant(args: argparse.Namespace, variant: str) -> dict:
    """Load entries and judge inputs exactly as ``cot_analysis`` would."""
    cot_argv = [
        "--benchmark_json",
        str(args.benchmark_json),
        "--benchmark_source_root",
        str(args.benchmark_source_root),
        "--provider",
        args.provider,
        "--seed",
        str(args.seed),
        "--variant",
        variant,
    ]
    if not args.include_unreliable_cot:
        cot_argv.append("--skip-unreliable-cot")
    cot_args = build_parser().parse_args(cot_argv)
    (
        cot_args.variant,
        cot_args.prompt,
        cot_args.trajectory_frame,
        cot_args.lane_reference,
    ) = _resolve_variant_settings(cot_args)

    provider = resolve_provider(args.provider)
    use_images = not cot_args.no_images and provider["supports_images"]
    if use_images:
        raise SystemExit("The batch runner sends text-only requests.")

    entries = extract_entries_from_benchmark(
        cot_args.benchmark_json,
        source_root=cot_args.benchmark_source_root,
        trajectory_frame=cot_args.trajectory_frame,
        lane_reference=cot_args.lane_reference,
    )
    original_entry_count = len(entries)
    skipped = []
    if cot_args.skip_unreliable_cot:
        entries, skipped = filter_unreliable_benchmark_cots(entries)
    if args.limit_entries is not None:
        entries = entries[: args.limit_entries]
    clip_ids = [entry["clip_id"] for entry in entries]
    if len(set(clip_ids)) != len(clip_ids):
        raise SystemExit(f"{variant}: duplicate clip IDs; results are matched by clip ID.")

    judge_inputs = {}
    for entry in entries:
        if entry.get("error"):
            continue
        try:
            judge_inputs[entry["clip_id"]] = prepare_judge_input(
                entry,
                prompt_name=cot_args.prompt,
                trajectory_frame=cot_args.trajectory_frame,
                lane_reference=cot_args.lane_reference,
                use_images=False,
            )
        except Exception as exc:
            entry["error"] = str(exc)

    return {
        "variant": variant,
        "cot_args": cot_args,
        "provider": provider,
        "run_config": _judge_run_config(cot_args, provider, use_images),
        "entries": entries,
        "original_entry_count": original_entry_count,
        "skipped": skipped,
        "judge_inputs": judge_inputs,
    }


def _split_provider_params(provider: dict, seed: int) -> tuple[dict, dict]:
    """Return (batch-level routing, per-request body fields except messages)."""
    params = copy.deepcopy(provider["extra_params"])
    params.pop("stream_options", None)
    extra_body = params.pop("extra_body", {})
    routing = extra_body.pop("provider", None)
    if routing is None:
        raise SystemExit("The provider config must pin extra_body.provider.")
    body = {"model": provider["model"], "seed": seed, **params, **extra_body}
    if provider["temperature"] is not None:
        body["temperature"] = provider["temperature"]
    # call_llm requests JSON mode first; batch requests cannot fall back.
    body["response_format"] = {"type": "json_object"}
    return routing, body


def _client(provider: dict) -> httpx.Client:
    if not provider["api_key"]:
        raise SystemExit(f"No API key for {provider['label']}.")
    if "openrouter.ai" not in (provider["base_url"] or ""):
        raise SystemExit("The batch runner needs an OpenRouter base_url.")
    return httpx.Client(
        base_url=provider["base_url"].rstrip("/"),
        headers={"Authorization": f"Bearer {provider['api_key']}"},
        timeout=120.0,
    )


def _batch_object(response: httpx.Response) -> dict:
    if response.is_error:
        # The body says why, e.g. which credit check a 402 failed.
        raise httpx.HTTPStatusError(
            f"{response.status_code} from {response.request.url}: {response.text[:500]}",
            request=response.request,
            response=response,
        )
    payload = response.json()
    if isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def _submit(client: httpx.Client, routing: dict, body_template: dict, prepared: list) -> tuple[dict, dict]:
    requests = []
    custom_ids = {}
    for spec in prepared:
        for idx, entry in enumerate(spec["entries"]):
            judge_input = spec["judge_inputs"].get(entry["clip_id"])
            if judge_input is None:
                continue
            custom_id = f"{spec['variant']}-{idx:04d}"
            custom_ids[custom_id] = {
                "variant": spec["variant"],
                "clip_id": entry["clip_id"],
                "fingerprint": judge_input.fingerprint,
            }
            body = dict(body_template)
            body["messages"] = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_input.prompt},
            ]
            requests.append({"custom_id": custom_id, "body": body})
    # OpenRouter rejects a batch whose "requests" key precedes the routing keys.
    payload = {
        "endpoint": ENDPOINT,
        "model": body_template["model"],
        "provider": routing,
        "completion_window": "24h",
        "requests": requests,
    }
    batch = _batch_object(client.post("/batches", json=payload))
    logger.info("Submitted batch %s with %d requests", batch["id"], len(requests))
    return batch, custom_ids


def _wait(client: httpx.Client, batch_id: str, poll_seconds: float) -> dict:
    failures = 0
    while True:
        try:
            batch = _batch_object(client.get(f"/batches/{batch_id}"))
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            failures += 1
            if (code is not None and code not in RETRYABLE_STATUS_CODES) or (
                failures >= MAX_CONSECUTIVE_POLL_FAILURES
            ):
                raise
            logger.warning("Polling batch %s failed (%s); retrying", batch_id, exc)
            time.sleep(poll_seconds)
            continue
        failures = 0
        status = batch.get("status")
        logger.info(
            "Batch %s: %s %s", batch_id, status, json.dumps(batch.get("request_counts"))
        )
        if status in FINAL_STATUSES:
            return batch
        time.sleep(poll_seconds)


def _result_for(spec: dict, entry: dict, item: dict | None, batch_id: str) -> dict:
    """One ``process_entry``-shaped result from a batch result item."""
    judge_input = spec["judge_inputs"][entry["clip_id"]]
    cot_args = spec["cot_args"]
    attempt: dict = {"json_mode": True, "status": "error"}
    body = None
    if item is None:
        raw_response = json.dumps({"error": "missing from batch results"})
    else:
        response = item.get("response") or {}
        body = response.get("body") if response.get("status_code") == 200 else None
        if body is None:
            error = item.get("error") or response
            raw_response = json.dumps({"error": error}, default=str)
            attempt["error"] = error
        else:
            choice = (body.get("choices") or [{}])[0]
            raw_response = (choice.get("message") or {}).get("content") or ""
            attempt["finish_reason"] = choice.get("finish_reason")
            attempt["usage"] = body.get("usage")
            attempt["status"] = "ok" if raw_response else "error"
    return {
        "clip_id": entry["clip_id"],
        "chain_of_thought": judge_input.cot_text,
        "trajectory_features": judge_input.trajectory_features,
        "evaluation": parse_response(raw_response),
        "judge_prompt": judge_input.prompt,
        "judge_input_fingerprint": judge_input.fingerprint,
        "judge_raw_response": raw_response,
        "judge_image_attached": False,
        "timestamp_us": entry.get("timestamp_us"),
        "variant": cot_args.variant,
        "prompt": cot_args.prompt,
        "trajectory_frame": cot_args.trajectory_frame,
        "lane_reference": cot_args.lane_reference,
        "benchmark": entry.get("benchmark"),
        "source_report": entry.get("source_report"),
        "llm_timing": {
            "model": (body or {}).get("model", spec["provider"]["model"]),
            "stream": False,
            "batch_id": batch_id,
            "generation_id": (body or {}).get("id"),
            "served_by": (body or {}).get("provider"),
            "attempts": [attempt],
        },
    }


def _write_variant(args, spec, batch, items_by_custom_id, custom_ids, batch_meta) -> int:
    by_clip = {
        meta["clip_id"]: items_by_custom_id.get(cid)
        for cid, meta in custom_ids.items()
        if meta["variant"] == spec["variant"]
    }
    results = []
    for entry in spec["entries"]:
        clip_id = entry["clip_id"]
        if entry.get("error"):
            results.append(
                {
                    "clip_id": clip_id,
                    "error": entry["error"],
                    "benchmark": entry.get("benchmark"),
                    "source_report": entry.get("source_report"),
                }
            )
            continue
        results.append(_result_for(spec, entry, by_clip.get(clip_id), batch["id"]))

    output = build_output_data(
        results,
        aggregate_results(results),
        entries_to_analyze=len(spec["entries"]),
        original_entry_count=spec["original_entry_count"],
        skipped_unreliable_cot_entries=spec["skipped"],
    )
    output["run_config"] = {**spec["run_config"], "batch": batch_meta}
    path = args.output_dir / f"{args.label}.{spec['variant']}.run_{args.run_index:03d}.json"
    with path.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    incomplete = sum(1 for r in results if not result_is_complete(r))
    logger.info(
        "%s: %d/%d complete -> %s",
        spec["variant"],
        len(results) - incomplete,
        len(results),
        path,
    )
    return incomplete


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    prepared = [_prepare_variant(args, variant) for variant in variants]
    provider = prepared[0]["provider"]
    routing, body_template = _split_provider_params(provider, args.seed)
    # Named by variant too, so variants can go in separate batches.
    state_path = (
        args.output_dir
        / f"{args.label}.batch.{'+'.join(variants)}.run_{args.run_index:03d}.json"
    )

    with _client(provider) as client:
        if args.batch_id is None:
            if state_path.exists():
                previous = json.loads(state_path.read_text())["batch_id"]
                raise SystemExit(
                    f"{state_path} already records batch {previous}; pass "
                    f"--batch-id {previous} to collect it, or remove the file to resubmit."
                )
            batch, custom_ids = _submit(client, routing, body_template, prepared)
            state = {
                "batch_id": batch["id"],
                "submitted_utc": _utc_now_iso(),
                "endpoint": ENDPOINT,
                "routing": routing,
                "request_params": body_template,
                "custom_ids": custom_ids,
            }
            state_path.write_text(json.dumps(state, indent=2) + "\n")
        else:
            state = json.loads(state_path.read_text())
            if state["batch_id"] != args.batch_id:
                raise SystemExit(f"{state_path} records batch {state['batch_id']}.")
            custom_ids = state["custom_ids"]
        batch = _wait(client, state["batch_id"], args.poll_seconds)

    # Collection rebuilds the judge inputs, so they must match what was sent.
    for spec in prepared:
        for cid, meta in custom_ids.items():
            judge_input = spec["judge_inputs"].get(meta["clip_id"])
            if meta["variant"] == spec["variant"] and (
                judge_input is None or judge_input.fingerprint != meta["fingerprint"]
            ):
                raise SystemExit(f"Judge input changed since submission: {cid}")

    state.update(
        {
            "status": batch.get("status"),
            "finished_utc": _utc_now_iso(),
            "request_counts": batch.get("request_counts"),
            "usage": batch.get("usage"),
        }
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    logger.info("Batch %s ended %s; usage %s", batch["id"], batch.get("status"), json.dumps(batch.get("usage")))

    items = {item["custom_id"]: item for item in (batch.get("results") or [])}
    unknown = sorted(set(items) - set(custom_ids))
    if unknown:
        raise SystemExit(f"Batch returned results for unsent request IDs: {unknown[:5]}")
    batch_meta = {
        "api": "openrouter",
        "batch_id": batch["id"],
        "endpoint": ENDPOINT,
        "routing": routing,
        "request_params": body_template,
        "status": batch.get("status"),
    }
    incomplete = sum(
        _write_variant(args, spec, batch, items, custom_ids, batch_meta)
        for spec in prepared
    )
    if incomplete:
        logger.error("%d judgment(s) incomplete; see llm_timing.attempts", incomplete)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
