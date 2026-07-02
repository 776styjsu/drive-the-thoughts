# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest

from benchmark_analysis import (
    BinaryConfusion,
    classification_metrics,
    clean_cot,
    consistency_from_value,
    cot_is_reliable,
    cot_is_unreliable,
    cot_reliability_flag,
    extract_entries,
    ground_truth_is_consistent,
    index_by_clip_id,
    judgment_for,
    llm_judgment,
    rule_judgment,
)


def test_extract_entries_supports_all_container_shapes() -> None:
    entries = [{"clip_id": "a"}, {"clip_id": "b"}]
    assert extract_entries(entries) == entries
    assert extract_entries({"results": entries}) == entries
    assert extract_entries({"entries": entries}) == entries
    assert extract_entries({"clip_id": "solo"}) == [{"clip_id": "solo"}]
    assert extract_entries([{"clip_id": "a"}, "not-a-dict"]) == [{"clip_id": "a"}]
    with pytest.raises(ValueError):
        extract_entries({"unexpected": 1})


def test_index_by_clip_id_drops_blank_ids() -> None:
    entries = [{"clip_id": "a"}, {"clip_id": "  "}, {"no_id": True}]
    assert list(index_by_clip_id(entries)) == ["a"]


def test_cot_reliability_flag_supports_both_schemas() -> None:
    assert cot_reliability_flag({"cot_reliability": {"reliable": True}}) is True
    assert cot_reliability_flag({"cot_reliability": {"reliable": False}}) is False
    assert cot_reliability_flag({"cot_reliable": "unreliable"}) is False
    assert cot_reliability_flag({"cot_reliable": "yes"}) is True
    assert cot_reliability_flag({}) is None
    assert cot_reliability_flag(None) is None
    # The nested schema wins over a flat key when both are present.
    assert (
        cot_reliability_flag(
            {"cot_reliability": {"reliable": False}, "cot_reliable": True}
        )
        is False
    )


def test_strict_reliability_helpers_treat_unknown_conservatively() -> None:
    assert not cot_is_reliable({})
    assert not cot_is_unreliable({})
    assert cot_is_reliable({"cot_reliable": True})
    assert cot_is_unreliable({"cot_reliable": False})


def test_ground_truth_prefers_explicit_consistency_keys() -> None:
    assert ground_truth_is_consistent({"cot_action_consistency": True}) is True
    assert ground_truth_is_consistent({"label": "inconsistent"}) is False
    with pytest.raises(ValueError):
        ground_truth_is_consistent({"clip_id": "x"})


def test_consistency_from_value_reads_verdict_strings() -> None:
    assert consistency_from_value("Consistent") is True
    assert consistency_from_value("invalid_parse") is False
    assert consistency_from_value("???") is None
    assert consistency_from_value(0) is False
    assert consistency_from_value(1) is True


def test_llm_judgment_uses_verdict_then_score() -> None:
    verdict_entry = {
        "evaluation": {"cot_output_alignment": {"verdict": "inconsistent"}}
    }
    assert llm_judgment(verdict_entry).is_consistent is False

    graded_entry = {"evaluation": {"cot_output_alignment": {"score": 5}}}
    judgment = llm_judgment(graded_entry)
    assert judgment.is_consistent is True
    assert judgment.score == 5.0
    assert llm_judgment({"evaluation": {"cot_output_alignment": {"score": 2}}}).is_consistent is False

    with pytest.raises(ValueError):
        llm_judgment({"clip_id": "x", "evaluation": {}})


def test_rule_judgment_reads_report_label_and_flags_invalid_parse() -> None:
    consistent = rule_judgment({"report": {"label": "consistent", "score": 1.0}})
    assert consistent.is_consistent is True
    assert consistent.valid_parse is True

    invalid = rule_judgment({"report": {"label": "invalid_parse", "score": 0.0}})
    assert invalid.is_consistent is False
    assert invalid.valid_parse is False

    with pytest.raises(ValueError):
        rule_judgment({"report": {}})


def test_judgment_for_dispatches_by_consistency_type() -> None:
    entry = {"evaluation": {"cot_output_alignment": {"score": 4}}}
    assert judgment_for(entry, "cot_output_alignment").is_consistent is True
    rule_entry = {"report": {"label": "contradictory"}}
    assert judgment_for(rule_entry, "rule_based").is_consistent is False
    with pytest.raises(ValueError):
        judgment_for(entry, "nope")


def test_binary_confusion_counts_and_f1() -> None:
    confusion = BinaryConfusion()
    # 2 true positives, 1 false positive, 3 true negatives, 1 false negative.
    for actual, predicted in [
        (False, False),
        (False, False),
        (True, False),
        (True, True),
        (True, True),
        (True, True),
        (False, True),
    ]:
        confusion.add(actual_is_consistent=actual, predicted_is_consistent=predicted)

    assert confusion.true_positive == 2
    assert confusion.false_positive == 1
    assert confusion.true_negative == 3
    assert confusion.false_negative == 1
    assert confusion.total == 7
    assert confusion.accuracy == pytest.approx(5 / 7)
    assert confusion.f1 == pytest.approx(2 / 3)

    metrics = classification_metrics(confusion)
    assert metrics["positive_class"] == "inconsistent"
    assert metrics["inconsistent"]["precision"] == pytest.approx(2 / 3)
    assert metrics["inconsistent"]["recall"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 4) / 2)


def test_clean_cot_renders_stringified_lists() -> None:
    assert clean_cot(["a", "b"]) == "a | b"
    assert clean_cot("['a', 'b']") == "a | b"
    assert clean_cot("plain text") == "plain text"
    assert clean_cot(None) == ""
