"""Tests for fraud.metrics evaluation functions."""

import numpy as np
import pytest

from fraud import metrics as M


def test_pr_auc_perfect_ranking():
    """PR-AUC of a perfect ranking is 1.0."""
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9, 1.0])

    auc = M.pr_auc(y, score)
    assert auc == 1.0


def test_recall_at_fpr_zero():
    """Recall at max_fpr=0 returns the recall achievable with zero false positives."""
    # With perfect ranking where positives have higher scores than negatives
    y = np.array([0, 0, 1, 1])
    score = np.array([0.0, 0.1, 0.9, 1.0])

    recall, threshold = M.recall_at_fpr(y, score, max_fpr=0.0)
    # At FPR=0, we can achieve recall of 1.0 by flagging both positives
    assert recall == 1.0


def test_precision_recall_f1():
    """Precision, recall, and F1 are calculated correctly on a 6-row example."""
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = 0.5

    result = M.precision_recall_f1(y, score, threshold)

    # score >= 0.5: indices 3, 4, 5 are positive
    # tp = 3 (all three are positives), fp = 0, fn = 0
    assert result["tp"] == 3
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_best_f1_threshold():
    """Best F1 threshold separates a perfectly separable case."""
    # Perfect separability: negatives have low scores, positives have high scores
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    threshold = M.best_f1_threshold(y, score)

    # The optimal threshold should achieve perfect F1 (1.0)
    # Verify that using this threshold gives tp=4, fp=0
    result = M.precision_recall_f1(y, score, threshold)
    assert result["tp"] == 4
    assert result["fp"] == 0


def test_expected_cost_curve_minimizes():
    """With missed_fraud_cost=100 and review_cost=1, optimal threshold flags exactly the frauds."""
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    cost = M.CostModel(missed_fraud_cost=100, review_cost=1)
    result = M.expected_cost_curve(y, score, cost, n_points=10)

    # Optimal threshold should flag both frauds and no negatives
    assert result["missed"] == 0
    assert result["flagged"] == 2


def test_brier_perfect_probabilities():
    """Brier score of perfect probabilities is 0."""
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.0, 0.0, 1.0, 1.0])

    score = M.brier(y, prob)
    assert score == 0.0


def test_reliability_bins():
    """Reliability returns bins whose n sums to len(y)."""
    y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.15, 0.25, 0.7, 0.8, 0.3, 0.75, 0.85, 0.9])

    bins = M.reliability(y, prob, bins=5)

    # Sum of n across all bins should equal len(y)
    total_n = sum(b["n"] for b in bins)
    assert total_n == len(y)


def test_full_report_has_required_keys():
    """full_report returns all required keys."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    report = M.full_report(y, score)

    # Check for key metrics
    required_keys = [
        "pr_auc", "roc_auc", "recall_at_fpr_0.1pct", "recall_at_fpr_1pct",
        "threshold", "precision", "recall", "f1", "flagged", "tp", "fp", "fn",
        "cost_threshold", "cost_per_10k", "cost_missed", "cost_flagged"
    ]

    for key in required_keys:
        assert key in report
