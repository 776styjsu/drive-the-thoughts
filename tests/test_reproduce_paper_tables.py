"""The released results reproduce the paper's reported counts."""

import importlib.util
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce_paper_tables", ROOT / "tools/reproduce_paper_tables.py")
rpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpt)


@pytest.fixture(scope="module")
def data():
    return rpt.Data()


# tab:main-results (TP, FP, FN, TN) on the 100 reliable pairs.
MAIN = {
    ("qwen", "llm"): (22, 28, 4, 46), ("deepseek", "llm"): (19, 16, 7, 58),
    ("opus", "llm"): (18, 15, 8, 59), ("jev", "llm"): (21, 28, 5, 46),
    ("qwen", "f_llm_map_graph"): (23, 16, 3, 58), ("deepseek", "f_llm_map_graph"): (21, 7, 5, 67),
    ("opus", "f_llm_map_graph"): (21, 4, 5, 70), ("jev", "f_llm_map_graph"): (23, 22, 3, 52),
}

# tab:safety-baselines: (unsafe flagged of 9, false alarms of 91, challenge median of 15).
SAFETY = {
    ("qwen", "llm"): (6, 44, 9), ("deepseek", "llm"): (5, 30, 8),
    ("opus", "llm"): (5, 28, 7), ("jev", "llm"): (6, 43, 8),
    ("qwen", "f_llm_map_graph"): (7, 32, 8), ("deepseek", "f_llm_map_graph"): (7, 21, 7),
    ("opus", "f_llm_map_graph"): (7, 18, 7), ("jev", "f_llm_map_graph"): (7, 38, 10),
}


def test_subsets(data):
    assert len(data.bench) == 150 and len(data.reliable) == 100 and len(data.challenge) == 15
    assert sum(not e["cot_action_consistency"] for e in data.challenge) == 10


@pytest.mark.parametrize("key", MAIN)
def test_main_results(data, key):
    assert rpt.inconsistency_counts(data.reliable, data.flags(data.table1[key])) == MAIN[key]


def test_rule_main_results(data):
    flags = rpt.rule_flags(rpt.RESULTS / "rule/benchmark.rule_consistency.json")
    assert rpt.inconsistency_counts(data.reliable, flags) == (9, 13, 17, 61)


@pytest.mark.parametrize("key", SAFETY)
def test_safety(data, key):
    tp, fp, _ = rpt.unsafe_counts(data.reliable, data.flags(data.table1[key]))
    challenge = [sum(data.flags(p, items=data.challenge).values())
                 for p in rpt.repeat_files(rpt.RESULTS / "challenge_set", *key)]
    assert (tp, fp, statistics.median(challenge)) == SAFETY[key]
