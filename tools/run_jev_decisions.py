#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Run the CoT judge on TypeSafe Jev through OpenRouter's decisions API.

Jev is a decisions model: it generates no text and answers a Score question
with a probability for each rubric level. OpenRouter serves it only at
``/api/alpha/decisions``, not at chat/completions, so ``cot_analysis`` cannot
call it. This tool builds the same judge prompt as ``cot_analysis`` for each
variant and rearranges it into a Jev request without rewording it:

- ``state`` holds the evidence: the CoT and the INPUT section's trajectory
  text, verbatim.
- ``instructions`` holds the alignment question (with the state paths named),
  the coordinate-frame notes, and, for F-LLM, the signal-selection guidance.
- ``criteria`` holds the five rubric levels, lowest score first, each with the
  prompt's worked examples for that score. Jev judges every level on its own
  and never sees its number, so the "Score N:" prefixes are dropped.

The output-format section and the request for a justification are dropped,
since Jev cannot write either. The verdict is the most probable level (ties go
to the lower score), mapped back to the 1-5 scale; the full distribution and
Jev's expected score are kept alongside it. Results are written in the
``cot_analysis`` format (``<label>.<variant>.run_NNN.json``), so the
evaluation tools read them unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from alpasim_utils.cot_consistency import DEFAULT_SEED
from cot_analysis.pipeline import aggregate_results, build_output_data, result_is_complete
from tools.run_openrouter_batch import _prepare_variant, _utc_now_iso

logger = logging.getLogger(__name__)

DIMENSION = "cot_output_alignment"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 6

_SECTION = re.compile(r"^=== (.+?) ===$", re.MULTILINE)
_RUBRIC_LINE = re.compile(r"^- \*\*Score ([1-5]): (.+?)\*\* (.+)$")
_EXAMPLE_FIELD = re.compile(r"^\s+- (CoT|Trajectory|Justification|Score): (.*)$")
QUESTION = "Does the executed trajectory match the ACTION stated in the CoT? Judge only the action."
JEV_QUESTION = (
    "Does the executed trajectory (`trajectory`) match the ACTION stated in the "
    "CoT (`chain_of_thought`)? Judge only the action."
)
# The prompt's opening lines, minus the role framing a chat model needs.
JEV_CONTEXT = (
    "`chain_of_thought` is the Chain-of-Thought (CoT) reasoning produced by a "
    "driving model; `trajectory` is the raw numerical trajectory it actually "
    "produced."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--benchmark-source-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--provider",
        default="jev_openrouter",
        help="Judge provider config to send. Default: %(default)s",
    )
    parser.add_argument(
        "--label",
        default="jev",
        help="Output file prefix. Default: %(default)s",
    )
    parser.add_argument(
        "--variants",
        default="llm,f_llm_map_graph",
        help="Comma-separated variants. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Recorded in run_config only; Jev takes no seed.",
    )
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
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the requests next to the outputs instead of sending them.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _sections(prompt: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a judge prompt into its preamble and ``=== TITLE ===`` sections."""
    matches = list(_SECTION.finditer(prompt))
    preamble = prompt[: matches[0].start()] if matches else prompt
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        sections.append((match.group(1), prompt[match.end() : end].strip()))
    return preamble, sections


def _rubric_and_examples(criteria_text: str) -> tuple[str, str | None, dict, dict]:
    """Parse the EVALUATION CRITERIA section.

    Returns (question, signal guidance or None, {score: level text},
    {score: [example strings]}).
    """
    head, _, rest = criteria_text.partition("\nRubric:\n")
    rubric_text, _, examples_text = rest.partition("\nExample:\n")
    if not rest or not examples_text:
        raise ValueError("Prompt has no Rubric/Example blocks to convert.")

    # head: score instruction, "**CoT-Output Alignment**", question, guidance.
    paragraphs = [p.strip() for p in head.strip().split("\n\n") if p.strip()]
    title_and_question = [p for p in paragraphs if p.startswith("**CoT-Output Alignment**")]
    if len(title_and_question) != 1:
        raise ValueError("Prompt has no CoT-Output Alignment question.")
    question = title_and_question[0].split("\n", 1)[1].strip()
    after = paragraphs[paragraphs.index(title_and_question[0]) + 1 :]
    guidance = "\n\n".join(after) or None

    levels = {}
    for line in rubric_text.strip().splitlines():
        match = _RUBRIC_LINE.match(line.strip())
        if match is None:
            raise ValueError(f"Unparsed rubric line: {line!r}")
        levels[int(match.group(1))] = f"{match.group(2)} {match.group(3)}"
    if sorted(levels) != [1, 2, 3, 4, 5]:
        raise ValueError(f"Rubric scores {sorted(levels)} are not 1-5.")

    examples: dict[int, list[str]] = {score: [] for score in levels}
    for block in re.split(r"^\d+\)$", examples_text, flags=re.MULTILINE)[1:]:
        fields = {}
        for line in block.strip("\n").splitlines():
            match = _EXAMPLE_FIELD.match(line)
            if match:
                fields[match.group(1)] = match.group(2).strip()
        if set(fields) != {"CoT", "Trajectory", "Justification", "Score"}:
            raise ValueError(f"Unparsed example block: {block!r}")
        examples[int(fields["Score"])].append(
            f"CoT: {fields['CoT']} Trajectory: {fields['Trajectory']} "
            f"Justification: {fields['Justification']}"
        )
    return question, guidance, levels, examples


def jev_request(prompt: str, cot_text: str, model: str) -> dict:
    """Rearrange a ``cot_analysis`` judge prompt into a Jev Score request."""
    _, sections = _sections(prompt)
    by_title = dict(sections)
    cot_block = f'**Chain-of-Thought Reasoning:**\n"{cot_text}"'
    input_text = by_title.get("INPUT", "")
    if not input_text.startswith(cot_block):
        raise ValueError("INPUT section does not open with the CoT.")
    trajectory = input_text[len(cot_block) :].strip()
    frames = [f"{title}\n{body}" for title, body in sections if title.startswith("COORDINATE FRAME")]

    question, guidance, levels, examples = _rubric_and_examples(
        by_title.get("EVALUATION CRITERIA", "")
    )
    if question != QUESTION:
        raise ValueError(f"Unexpected alignment question: {question!r}")
    instructions = {
        "question": JEV_QUESTION,
        "context": JEV_CONTEXT,
        "coordinate_frames": "\n\n".join(frames),
    }
    if guidance is not None:
        instructions["signal_selection"] = guidance
    return {
        "model": model,
        "state": {"chain_of_thought": cot_text, "trajectory": trajectory},
        "questions": {
            DIMENSION: {
                "type": "score",
                "instructions": instructions,
                "criteria": [
                    {"what": levels[score], "examples": examples[score]}
                    for score in sorted(levels)
                ],
            }
        },
    }


def evaluation_from_answer(answer: dict) -> dict:
    """Map a Jev Score answer (levels 0-4) to a 1-5 judge evaluation."""
    if answer.get("type") != "score":
        raise ValueError(f"Expected a score answer, got {answer.get('type')!r}.")
    probabilities = {int(k): float(v) for k, v in answer["probabilities"].items()}
    if sorted(probabilities) != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected levels 0-4, got {sorted(probabilities)}.")
    # max() keeps the first maximum, so ties go to the lower score.
    level = max(sorted(probabilities), key=lambda k: probabilities[k])
    tied = sum(1 for p in probabilities.values() if p == probabilities[level]) > 1
    return {
        DIMENSION: {
            "score": level + 1,
            "decision_rule": "argmax_level",
            "argmax_tie": tied,
            "score_probabilities": {str(k + 1): p for k, p in sorted(probabilities.items())},
            "expected_score": float(answer["score"]) + 1,
            "confidence": answer.get("confidence"),
        }
    }


def _decisions_url(base_url: str) -> str:
    # The provider's base URL is OpenRouter's /api/v1; decisions live at /api/alpha.
    if "openrouter.ai" not in base_url or not base_url.rstrip("/").endswith("/v1"):
        raise SystemExit(f"Expected an OpenRouter /api/v1 base URL, got {base_url}.")
    return base_url.rstrip("/")[: -len("/v1")] + "/alpha/decisions"


def _call(client: httpx.Client, url: str, request: dict) -> tuple[dict | None, dict]:
    """POST one decision request with retries; return (response body, timing)."""
    timing: dict = {
        "model": request["model"],
        "stream": False,
        "endpoint": url,
        "started_at_utc": _utc_now_iso(),
        "attempts": [],
    }
    total_start = time.perf_counter()
    body = None
    for attempt_no in range(MAX_ATTEMPTS):
        attempt: dict = {"started_at_utc": _utc_now_iso(), "status": "error"}
        start = time.perf_counter()
        retry_after = None
        try:
            response = client.post(url, json=request)
            attempt["http_status"] = response.status_code
            if response.is_success:
                body = response.json()
                attempt["status"] = "ok"
                attempt["usage"] = body.get("usage")
            else:
                attempt["error"] = response.text[:500]
                retry_after = response.headers.get("retry-after")
        except (httpx.TransportError, ValueError) as exc:
            attempt["error_type"] = type(exc).__name__
            attempt["error"] = str(exc)
        attempt["total_elapsed_s"] = time.perf_counter() - start
        timing["attempts"].append(attempt)
        retryable = attempt.get("http_status") in RETRYABLE_STATUS_CODES or (
            "error_type" in attempt
        )
        if body is not None or not retryable:
            break
        try:
            delay = float(retry_after) if retry_after else 2.0**attempt_no
        except ValueError:
            delay = 2.0**attempt_no
        time.sleep(min(delay, 60.0))
    timing["status"] = "ok" if body is not None else "error"
    timing["retry_count"] = len(timing["attempts"]) - 1
    timing["total_elapsed_s"] = time.perf_counter() - total_start
    timing["ended_at_utc"] = _utc_now_iso()
    if body is not None:
        timing["model"] = body.get("model", request["model"])
        timing["generation_id"] = body.get("id")
        timing["served_by"] = body.get("provider")
    return body, timing


def _judge(client, url, spec, entry) -> dict:
    judge_input = spec["judge_inputs"][entry["clip_id"]]
    cot_args = spec["cot_args"]
    request = jev_request(judge_input.prompt, judge_input.cot_text, spec["provider"]["model"])
    body, timing = _call(client, url, request)
    try:
        if body is None:
            raise ValueError(timing["attempts"][-1].get("error", "request failed"))
        evaluation = evaluation_from_answer(body["answers"][DIMENSION])
    except (KeyError, TypeError, ValueError) as exc:
        evaluation = {"error": str(exc)}
    return {
        "clip_id": entry["clip_id"],
        "chain_of_thought": judge_input.cot_text,
        "trajectory_features": judge_input.trajectory_features,
        "evaluation": evaluation,
        # The chat prompt the request was built from, for cross-judge checks.
        "judge_prompt": judge_input.prompt,
        "judge_request": request,
        "judge_request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "judge_input_fingerprint": judge_input.fingerprint,
        "judge_raw_response": json.dumps(body) if body is not None else None,
        "judge_image_attached": False,
        "timestamp_us": entry.get("timestamp_us"),
        "variant": cot_args.variant,
        "prompt": cot_args.prompt,
        "trajectory_frame": cot_args.trajectory_frame,
        "lane_reference": cot_args.lane_reference,
        "benchmark": entry.get("benchmark"),
        "source_report": entry.get("source_report"),
        "llm_timing": timing,
    }


def _run_variant(args, spec, client, url) -> int:
    def judge_or_error(entry):
        if entry.get("error"):
            return {
                "clip_id": entry["clip_id"],
                "error": entry["error"],
                "benchmark": entry.get("benchmark"),
                "source_report": entry.get("source_report"),
            }
        return _judge(client, url, spec, entry)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(judge_or_error, spec["entries"]))

    output = build_output_data(
        results,
        aggregate_results(results),
        entries_to_analyze=len(spec["entries"]),
        original_entry_count=spec["original_entry_count"],
        skipped_unreliable_cot_entries=spec["skipped"],
    )
    output["run_config"] = {
        **spec["run_config"],
        "decisions": {
            "api": "openrouter",
            "endpoint": url,
            "request_builder": "tools/run_jev_decisions.py:jev_request",
            "decision_rule": "argmax level, ties to the lower score; score = level + 1",
        },
    }
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

    if args.dry_run:
        for spec in prepared:
            requests = [
                jev_request(ji.prompt, ji.cot_text, provider["model"])
                for ji in spec["judge_inputs"].values()
            ]
            path = args.output_dir / f"{args.label}.{spec['variant']}.requests.json"
            path.write_text(json.dumps(requests, indent=2, ensure_ascii=False) + "\n")
            logger.info("%s: wrote %d requests -> %s", spec["variant"], len(requests), path)
        return 0

    if not provider["api_key"]:
        raise SystemExit(f"No API key for {provider['label']}.")
    url = _decisions_url(provider["base_url"])
    headers = {"Authorization": f"Bearer {provider['api_key']}"}
    with httpx.Client(headers=headers, timeout=120.0) as client:
        incomplete = sum(_run_variant(args, spec, client, url) for spec in prepared)
    if incomplete:
        logger.error("%d judgment(s) incomplete; see llm_timing.attempts", incomplete)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
