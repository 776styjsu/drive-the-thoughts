"""Time the rule-based monitor per pair on the 100 reliable benchmark pairs.

Per query (in memory, after inputs are loaded): trajectory parsing into
meta-actions (build_additional_info), then CoT parsing and matching
(match_cot_to_trajectory) to a verdict. File I/O is excluded, matching the
judge-only wall clock used for the LLM rows. The paper's row
(data/results/latency/rule_latency.json) came from one core of a standard CPU node:

    uv run python tools/time_rule_monitor.py --output rule_latency.json
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from alpasim_utils.consistency import match_cot_to_trajectory
from alpasim_utils.trajectory_additional_info import build_additional_info
from benchmark_analysis import cot_is_reliable, extract_entries, load_json
from cot_analysis.benchmark_source import resolve_benchmark_source_path
from cot_analysis.consistency_check import _flatten_cot


def pct(values, q):
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, default=Path("data/benchmark/benchmark.json"))
    p.add_argument("--released", type=Path, default=Path("data/results/rule/benchmark.rule_consistency.json"))
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    entries = [e for e in extract_entries(load_json(args.benchmark)) if cot_is_reliable(e)]
    released = {r["clip_id"]: r for r in load_json(args.released)["results"]}

    pairs = []
    for e in entries:
        path = resolve_benchmark_source_path(e["source_scene_file"], args.benchmark, Path("."))
        metadata = json.loads(path.read_text())
        pairs.append((str(e["clip_id"]), metadata, _flatten_cot(metadata.get("chain_of_thought")) or ""))

    # Check the timed path reproduces the released (Table 1) verdicts.
    mismatches = []
    for clip_id, metadata, cot in pairs:
        meta_actions = build_additional_info(metadata)["meta_actions"]
        label = match_cot_to_trajectory(cot, meta_actions).label
        if label != released[clip_id]["report"]["label"]:
            mismatches.append((clip_id, label, released[clip_id]["report"]["label"]))

    def one(metadata, cot):
        t0 = time.perf_counter_ns()
        meta_actions = build_additional_info(metadata)["meta_actions"]
        t1 = time.perf_counter_ns()
        report = match_cot_to_trajectory(cot, meta_actions)
        _ = report.label
        t2 = time.perf_counter_ns()
        return t1 - t0, t2 - t0

    for _ in range(args.warmup):
        for _, metadata, cot in pairs:
            one(metadata, cot)

    evid_ms, e2e_ms = [], []
    for _ in range(args.rounds):
        for _, metadata, cot in pairs:
            ev, tot = one(metadata, cot)
            evid_ms.append(ev / 1e6)
            e2e_ms.append(tot / 1e6)

    cpu_model = next(
        (l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
        platform.processor(),
    )
    med = statistics.median(e2e_ms)
    result = {
        "cpu_model": cpu_model,
        "python": platform.python_version(),
        "pairs": len(pairs),
        "rounds": args.rounds,
        "samples": len(e2e_ms),
        "verdict_mismatches_vs_released": mismatches,
        "e2e_ms": {"median": med, "p95": pct(e2e_ms, 0.95), "mean": statistics.fmean(e2e_ms),
                   "min": min(e2e_ms), "max": max(e2e_ms)},
        "evidence_ms": {"median": statistics.median(evid_ms), "p95": pct(evid_ms, 0.95)},
        "rate_hz": 1000.0 / med,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
