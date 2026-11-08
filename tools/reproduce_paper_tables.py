"""Recompute the paper's tables from the released labels and per-run results.

Reads only data/benchmark and data/results; no simulator, GPU or API key needed.

    uv run python tools/reproduce_paper_tables.py            # every table
    uv run python tools/reproduce_paper_tables.py main latency

Tables (paper label -> name here):
    tab:main-results       main         inconsistency detection, reliable subset
    tab:cons-safety        cons-safety  consistency labels vs. trajectory safety
    tab:safety-baselines   safety       unsafe-trajectory detection (RQ4)
    tab:latency            latency      per-pair latency (RQ5)
    tab:threshold-sweep    threshold    score threshold 1-4 (appendix)
    tab:temporal-agg       temporal     temporal aggregation (appendix)

Judge-based rows use the repeat marked `table1_run` in
data/results/llm_matrix/manifest.json: the median-F1 repeat of three on the
reliable subset. A pair is flagged INCONSISTENT when its 1-5 score is at most
the threshold (2 unless stated).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmark/benchmark.json"
CHALLENGE = ROOT / "data/benchmark/ood_hazard_2604_unsafe_traces_adjudicated.json"
RESULTS = ROOT / "data/results"
MATRIX = RESULTS / "llm_matrix"
THRESHOLD = 2.0

JUDGES = {"qwen": "Qwen3.5-4B", "deepseek": "DeepSeek V4.1 Flash", "opus": "Opus 5.5", "jev": "Jev 1.14"}
MONITORS = {"llm": "Ego-frame", "f_llm_map_graph": "Lane-aware"}


# ---------------------------------------------------------------- helpers


def load(path: Path):
    return json.loads(path.read_text())


def entries(path: Path) -> list[dict]:
    data = load(path)
    return data["entries"] if isinstance(data, dict) else data


def r2(x: float) -> str:
    """Round half up to two decimals, as in the paper."""
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return precision, recall, f1


def judge_scores(path: Path) -> dict[str, float]:
    """clip_id -> judge score, for complete judgments only."""
    scores = {}
    for r in load(path)["results"]:
        ev = r.get("evaluation") or {}
        if r.get("error") or any(ev.get(k) for k in ("dry_run", "parse_error", "error")):
            continue
        score = (ev.get("cot_output_alignment") or {}).get("score")
        if score is not None and not isinstance(score, bool):
            scores[r["clip_id"]] = float(score)
    return scores


def table1_files() -> dict[tuple[str, str], Path]:
    runs = {(f["judge"], f["variant"]): MATRIX / f["file"]
            for f in load(MATRIX / "manifest.json")["files"] if f["table1_run"]}
    assert len(runs) == len(JUDGES) * len(MONITORS), runs
    return runs


def repeat_files(directory: Path, judge: str, variant: str) -> list[Path]:
    return [directory / f"{judge}.{variant}.run_00{k}.json" for k in (1, 2, 3)]


def rule_flags(path: Path) -> dict[str, bool]:
    report = load(path)
    assert not report["summary"]["errored_clips"], path
    return {r["clip_id"]: r["report"]["label"] == "inconsistent" for r in report["results"]}


class Data:
    def __init__(self) -> None:
        self.bench = entries(BENCHMARK)
        self.reliable = [e for e in self.bench if e["cot_reliability"]["reliable"]]
        self.challenge = entries(CHALLENGE)
        self.table1 = table1_files()

    def flags(self, path: Path, threshold: float = THRESHOLD, items: list[dict] | None = None) -> dict[str, bool]:
        scores = judge_scores(path)
        items = self.reliable if items is None else items
        missing = [e["clip_id"] for e in items if e["clip_id"] not in scores]
        assert not missing, f"{path.name}: no complete judgment for {missing}"
        return {e["clip_id"]: scores[e["clip_id"]] <= threshold for e in items}


def inconsistency_counts(items: list[dict], flags: dict[str, bool]) -> tuple[int, int, int, int]:
    tp = sum(flags[e["clip_id"]] and not e["cot_action_consistency"] for e in items)
    fp = sum(flags[e["clip_id"]] and e["cot_action_consistency"] for e in items)
    fn = sum(not flags[e["clip_id"]] and not e["cot_action_consistency"] for e in items)
    return tp, fp, fn, len(items) - tp - fp - fn


def unsafe_counts(items: list[dict], flags: dict[str, bool]) -> tuple[int, int, int]:
    tp = sum(flags[e["clip_id"]] and not e["planned_trajectory_safe"] for e in items)
    fp = sum(flags[e["clip_id"]] and e["planned_trajectory_safe"] for e in items)
    fn = sum(not flags[e["clip_id"]] and not e["planned_trajectory_safe"] for e in items)
    return tp, fp, fn


def show(title: str, header: list[str], rows: list[list]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print(f"\n{title}")
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))


# ---------------------------------------------------------------- tables


def table_main(d: Data) -> None:
    rows = []
    rule = rule_flags(RESULTS / "rule/benchmark.rule_consistency.json")
    for name, judge, flags in [("Rule-based", "N/A", rule)] + [
            (MONITORS[v], JUDGES[j], d.flags(d.table1[j, v])) for v in MONITORS for j in JUDGES]:
        tp, fp, fn, tn = inconsistency_counts(d.reliable, flags)
        rows.append([name, judge, tp, fp, fn, tn, *map(r2, prf(tp, fp, fn))])
    show("tab:main-results  inconsistency detection, reliable subset (n=100)",
         ["Monitor", "Judge", "TP", "FP", "FN", "TN", "Prec.", "Rec.", "F1"], rows)


def table_cons_safety(d: Data) -> None:
    rows = []
    for label, items in (("Reliable subset", d.reliable), ("Challenge set", d.challenge)):
        groups = [("UNSAFE", [e for e in items if not e["planned_trajectory_safe"]]),
                  ("SAFE", [e for e in items if e["planned_trajectory_safe"]]), ("Total", items)]
        for name, group in groups:
            if not group or (name == "Total" and len(group) == len(groups[0][1])):
                continue
            inc = sum(not e["cot_action_consistency"] for e in group)
            rows.append([label, name, inc, len(group) - inc, len(group), f"{100 * inc / len(group):.1f}%"])
    show("tab:cons-safety  consistency vs. trajectory safety",
         ["Set", "Trajectory", "Incons.", "Cons.", "Total", "Incons. rate"], rows)


def selforacle_flags(path: Path) -> dict[str, bool]:
    with path.open() as f:
        return {r["clip_id"]: r["flag_epsilon_0.05"] == "True" for r in csv.DictReader(f)}


def table_safety(d: Data) -> None:
    n_unsafe = sum(not e["planned_trajectory_safe"] for e in d.reliable)
    rows = []

    def row(monitor, variant, flags, challenge_counts):
        tp, fp, fn = unsafe_counts(d.reliable, flags)
        rows.append([monitor, variant, tp, fp, *map(r2, prf(tp, fp, fn)), statistics.median(challenge_counts)])

    def challenge_count(flags):
        return sum(flags[e["clip_id"]] for e in d.challenge)

    # SelfOracle: the median-F1 seed on the reliable subset (ties keep the lower seed);
    # challenge column = median over seeds. Five-frame mean score, epsilon 0.05.
    for model in ("sae", "vae"):
        seeds = [selforacle_flags(RESULTS / f"selforacle/{model}_seed{s}/per_target.csv") for s in (0, 1, 2)]
        f1s = [prf(*unsafe_counts(d.reliable, f))[2] for f in seeds]
        median_seed = next(s for s in (0, 1, 2) if f1s[s] == sorted(f1s)[1])
        challenge = [challenge_count(selforacle_flags(
            RESULTS / f"challenge_set/selforacle/{model}_seed{s}/per_target.csv")) for s in (0, 1, 2)]
        row("SelfOracle", f"{model.upper()} (seed {median_seed})", seeds[median_seed], challenge)

    th = load(RESULTS / "time_headway/summary.json")["alarm"]
    tp, fp = th["reliable"]["tp"], th["reliable"]["fp"]
    rows.append(["Time Headway", "2 s", tp, fp, *map(r2, prf(tp, fp, n_unsafe - tp)), th["ood15"]["flagged"]])

    rule_challenge = rule_flags(RESULTS / "challenge_set/rule.run_001.json")
    row("Rule-based", "N/A", rule_flags(RESULTS / "rule/benchmark.rule_consistency.json"),
        [challenge_count(rule_challenge)])
    for v in MONITORS:
        for j in JUDGES:
            challenge = [challenge_count(d.flags(p, items=d.challenge))
                         for p in repeat_files(RESULTS / "challenge_set", j, v)]
            row(MONITORS[v], JUDGES[j], d.flags(d.table1[j, v]), challenge)
    show(f"tab:safety-baselines  unsafe-trajectory detection; reliable subset ({n_unsafe} unsafe, "
         f"{len(d.reliable) - n_unsafe} safe) and challenge set ({len(d.challenge)} unsafe; median over repeats)",
         ["Monitor", "Variant", "Unsafe flagged", "False alarms", "Prec.", "Rec.", "F1", "Challenge flagged"], rows)


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def completion_tokens(result: dict) -> int | None:
    usage = (result["llm_timing"].get("attempts") or [{}])[-1].get("usage") or {}
    return usage.get("completion_tokens")


def table_latency(d: Data) -> None:
    rule = load(RESULTS / "latency/rule_latency.json")
    rows = [["Rule-based", "N/A", f"{rule['e2e_ms']['median'] / 1000:.4f}",
             f"{rule['e2e_ms']['p95'] / 1000:.4f}", "--", f"{1000 / rule['e2e_ms']['median']:.0f}"]]
    reliable = {e["clip_id"] for e in d.reliable}
    for v in MONITORS:
        for j in JUDGES:
            secs, toks = [], []
            for path in repeat_files(MATRIX, j, v):
                for r in load(path)["results"]:
                    if r["clip_id"] not in reliable:
                        continue
                    # Opus ran through the Batch API, which reports no per-query latency
                    # (one pair re-judged outside the batch carries a wall clock; skipped).
                    if j != "opus" and "total_elapsed_s" in r["llm_timing"]:
                        secs.append(r["llm_timing"]["total_elapsed_s"])
                    if j != "jev" and completion_tokens(r) is not None:
                        toks.append(completion_tokens(r))
            med = statistics.median(secs) if secs else None
            rows.append([MONITORS[v], JUDGES[j],
                         f"{med:#.3g}" if med else "--",
                         f"{percentile(secs, 0.95):#.3g}" if secs else "--",
                         f"{statistics.mean(toks):,.0f}" if toks else "--",
                         f"{1 / med:#.3g}" if med else "--"])
    show("tab:latency  per-pair latency on the reliable subset, pooled over three repeats (judge wall clock; "
         "Tok. = mean completion tokens). Rule-based: one core, " + rule["cpu_model"] + ". Opus: Batch API, "
         "no per-query latency. Jev: fixed 19-token decision, no generated text. The paper rounds for display.",
         ["Monitor", "Judge", "Med. (s)", "P95 (s)", "Tok.", "Rate (Hz)"], rows)


def table_threshold(d: Data) -> None:
    rows = []
    for j in JUDGES:
        for v in MONITORS:
            cells = []
            for threshold in (1, 2, 3, 4):
                tp, fp, fn, _ = inconsistency_counts(d.reliable, d.flags(d.table1[j, v], threshold))
                cells.append(" / ".join(map(r2, prf(tp, fp, fn))))
            rows.append([JUDGES[j], MONITORS[v], *cells])
    show("tab:threshold-sweep  precision / recall / F1 at score threshold 1-4 (Table 1 runs)",
         ["Judge", "Monitor", "Threshold 1", "Threshold 2", "Threshold 3", "Threshold 4"], rows)


DECAY_WEIGHTS = [2 ** (-i / 2) for i in range(5)]  # half-life 2 steps, newest first
POLICIES = {"single_frame": ("Single step", 1), "two_consecutive": ("Two consecutive (2/2)", 2),
            "three_consecutive": ("Three consecutive (3/3)", 3), "two_of_three": ("2-out-of-3 vote", 3),
            "three_of_five": ("3-out-of-5 vote", 5), "weighted_decay": ("Half-life decay (h=2)", 5)}


def alarms(flags: list[bool]) -> dict[str, bool]:
    """flags run oldest -> target step."""
    r = flags[::-1]
    return {"single_frame": r[0], "two_consecutive": r[0] and r[1], "three_consecutive": all(r[:3]),
            "two_of_three": sum(r[:3]) >= 2, "three_of_five": sum(r) >= 3,
            "weighted_decay": sum(w * x for w, x in zip(DECAY_WEIGHTS, r)) >= 0.5 * sum(DECAY_WEIGHTS)}


def table_temporal(d: Data) -> None:
    """History verdicts from the temporal runs; the target verdict from the Table 1
    lane-aware Qwen run with the same launch index, so the single-step row is Table 1
    restricted to these targets."""
    runs = []
    for k in (1, 2, 3):
        targets = load(RESULTS / f"temporal/qwen.eval92.run_00{k}.json")["targets"]
        target_flags = d.flags(MATRIX / f"qwen.f_llm_map_graph.run_00{k}.json")
        scores = {}
        for policy in POLICIES:
            tp = fp = fn = 0
            for t in targets:
                assert alarms(t["flags"]) == t["alarms"], t["clip_id"]  # policy code matches the run
                alarm = alarms(t["flags"][:4] + [target_flags[t["clip_id"]]])[policy]
                inconsistent = not t["is_consistent"]
                tp, fp, fn = tp + (alarm and inconsistent), fp + (alarm and not inconsistent), fn + (not alarm and inconsistent)
            scores[policy] = prf(tp, fp, fn)
        runs.append(scores)
    f1s = [r["single_frame"][2] for r in runs]
    k = next(i for i in range(3) if f1s[i] == sorted(f1s)[1])  # median single-step F1, ties -> lower
    rows = [[name, steps, *map(r2, runs[k][policy]),
             " / ".join(r2(r[policy][2]) for r in runs)] for policy, (name, steps) in POLICIES.items()]
    show(f"tab:temporal-agg  lane-aware Qwen3.5-4B on {len(targets)} targets with five-step windows; "
         f"repeat {k + 1} (median single-step F1)",
         ["Rule", "Steps", "Prec.", "Rec.", "F1", "F1 repeats 1/2/3"], rows)


TABLES = {"main": table_main, "cons-safety": table_cons_safety, "safety": table_safety,
          "latency": table_latency, "threshold": table_threshold, "temporal": table_temporal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="*", metavar="TABLE",
                        help=f"any of {', '.join(TABLES)} (default: all)")
    args = parser.parse_args()
    unknown = [t for t in args.tables if t not in TABLES]
    if unknown:
        parser.error(f"unknown table(s) {unknown}; choose from {', '.join(TABLES)}")
    data = Data()
    for name in args.tables or TABLES:
        TABLES[name](data)


if __name__ == "__main__":
    main()
