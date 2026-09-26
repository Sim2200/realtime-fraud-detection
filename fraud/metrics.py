"""Evaluation for a heavily imbalanced binary problem.

Accuracy is meaningless at a 0.17% positive rate (predicting "never fraud" scores
99.8%). The primary metric is PR-AUC (average precision), which only looks at the
positive class. The others answer operational questions:

  recall_at_fpr     of the frauds, how many do we catch if we may only flag
                    0.1% (or 1%) of legitimate transactions?
  cost threshold    with a cost per missed fraud and a cost per manual review,
                    which score threshold minimises expected cost?
  calibration       do predicted probabilities mean what they say? (Brier score,
                    reliability curve)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def recall_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> tuple[float, float]:
    """Highest recall (TPR) achievable while FPR <= max_fpr, and the threshold that gives it."""
    fpr, tpr, thr = roc_curve(y, score)
    ok = fpr <= max_fpr
    i = int(np.argmax(np.where(ok, tpr, -1.0)))
    return float(tpr[i]), float(thr[i])


def precision_recall_f1(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn}


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


@dataclass(frozen=True)
class CostModel:
    """Money lost per missed fraud vs money spent per manual review of a flagged transaction."""

    missed_fraud_cost: float = 200.0   # roughly the mean fraudulent amount in this data is ~122; add chargeback fees
    review_cost: float = 5.0           # analyst time per flagged transaction


def expected_cost_curve(y: np.ndarray, score: np.ndarray, cost: CostModel, n_points: int = 400,
                        thresholds: np.ndarray | None = None) -> dict:
    """Expected cost per 10,000 transactions at each threshold, and the minimising threshold.

    Thresholds default to score quantiles plus a log-spaced grid, because calibrated
    fraud probabilities sit near zero for almost every row and quantiles alone would
    never sample the region where the decision actually changes.
    """
    if thresholds is None:
        thresholds = np.unique(np.concatenate([np.quantile(score, np.linspace(0, 1, n_points)),
                                               np.logspace(-6, 0, n_points)]))
    rows = []
    for t in thresholds:
        pred = score >= t
        fn = int((~pred & (y == 1)).sum())
        flagged = int(pred.sum())
        total = fn * cost.missed_fraud_cost + flagged * cost.review_cost
        rows.append((float(t), total / len(y) * 10_000, fn, flagged))
    best = min(rows, key=lambda r: r[1])
    return {"threshold": best[0], "cost_per_10k": round(best[1], 2), "missed": best[2], "flagged": best[3],
            "curve": [{"threshold": r[0], "cost_per_10k": round(r[1], 2)} for r in rows]}


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(brier_score_loss(y, prob))


def reliability(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict]:
    """Mean predicted vs observed fraud rate per probability bin (quantile bins, so the rare class shows)."""
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, bins + 1)))
    out = []
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            out.append({"bin": b, "n": int(m.sum()), "mean_pred": float(prob[m].mean()), "observed": float(y[m].mean())})
    return out


def full_report(y: np.ndarray, score: np.ndarray, threshold: float | None = None,
                cost: CostModel = CostModel()) -> dict:
    """Everything the README tables need, from one score vector."""
    threshold = best_f1_threshold(y, score) if threshold is None else threshold
    r01, _ = recall_at_fpr(y, score, 0.001)
    r1, _ = recall_at_fpr(y, score, 0.01)
    c = expected_cost_curve(y, score, cost)
    return {
        "pr_auc": round(pr_auc(y, score), 4),
        "roc_auc": round(roc_auc(y, score), 4),
        "recall_at_fpr_0.1pct": round(r01, 4),
        "recall_at_fpr_1pct": round(r1, 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in precision_recall_f1(y, score, threshold).items()},
        "cost_threshold": round(c["threshold"], 6), "cost_per_10k": c["cost_per_10k"],
        "cost_missed": c["missed"], "cost_flagged": c["flagged"],
    }
