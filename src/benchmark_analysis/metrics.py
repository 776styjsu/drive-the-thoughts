# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Binary classification metrics for consistency monitoring.

"Inconsistent" is the positive class throughout: the monitors exist to detect
inconsistent reasoning, so precision/recall/F1 default to that framing while
the consistent-class numbers are also reported.
"""

from __future__ import annotations

from dataclasses import dataclass


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


@dataclass
class BinaryConfusion:
    """Confusion counts with inconsistent as the positive class."""

    true_positive: int = 0  # actual inconsistent, predicted inconsistent
    false_positive: int = 0  # actual consistent,   predicted inconsistent
    true_negative: int = 0  # actual consistent,   predicted consistent
    false_negative: int = 0  # actual inconsistent, predicted consistent

    def add(self, *, actual_is_consistent: bool, predicted_is_consistent: bool) -> None:
        if actual_is_consistent:
            if predicted_is_consistent:
                self.true_negative += 1
            else:
                self.false_positive += 1
        else:
            if predicted_is_consistent:
                self.false_negative += 1
            else:
                self.true_positive += 1

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def correct(self) -> int:
        return self.true_positive + self.true_negative

    @property
    def accuracy(self) -> float:
        return _safe_div(self.correct, self.total)

    @property
    def f1(self) -> float:
        """F1 for the inconsistent (positive) class."""
        return precision_recall_f1(
            self.true_positive, self.false_positive, self.false_negative
        )["f1"]

    def to_nested_dict(self) -> dict:
        """The actual-by-predicted dict shape used in the JSON reports."""
        return {
            "actual_consistent": {
                "predicted_consistent": self.true_negative,
                "predicted_inconsistent": self.false_positive,
            },
            "actual_inconsistent": {
                "predicted_consistent": self.false_negative,
                "predicted_inconsistent": self.true_positive,
            },
        }


def classification_metrics(confusion: BinaryConfusion) -> dict:
    """Precision/recall/F1 per class, balanced accuracy, and Cohen's kappa."""
    tp = confusion.true_positive
    fp = confusion.false_positive
    tn = confusion.true_negative
    fn = confusion.false_negative
    total = confusion.total

    recall_inconsistent = _safe_div(tp, tp + fn)
    recall_consistent = _safe_div(tn, tn + fp)
    balanced_accuracy = (recall_inconsistent + recall_consistent) / 2

    observed_agreement = _safe_div(tp + tn, total)
    expected_agreement = _safe_div(
        (tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), total * total
    )
    kappa = _safe_div(observed_agreement - expected_agreement, 1 - expected_agreement)

    return {
        "positive_class": "inconsistent",
        "inconsistent": precision_recall_f1(tp, fp, fn),
        "consistent": precision_recall_f1(tn, fn, fp),
        "balanced_accuracy": balanced_accuracy,
        "cohens_kappa": kappa,
    }
