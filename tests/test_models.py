"""Tests for fraud.models training functions."""

import numpy as np
import pytest

from fraud import models


def test_train_models():
    """Train each model with each strategy; verify score shape and PR-AUC."""
    np.random.seed(42)

    # Create synthetic separable dataset: 2000 rows x 5 features, 3% positives
    n_rows = 2000
    x = np.random.randn(n_rows, 5).astype(np.float32)
    y = np.zeros(n_rows, dtype=np.int64)

    # Make 3% positives, with higher mean scores for positives
    n_pos = max(1, int(n_rows * 0.03))
    pos_idx = np.random.choice(n_rows, n_pos, replace=False)
    x[pos_idx] += 2.0  # Shift positives to be more separable
    y[pos_idx] = 1

    # Split: train 70%, val 10%, test 20%
    i_train = int(n_rows * 0.7)
    i_val = int(n_rows * 0.8)

    x_train, y_train = x[:i_train], y[:i_train]
    x_val, y_val = x[i_train:i_val], y[i_train:i_val]
    x_test, y_test = x[i_val:], y[i_val:]

    for name in ("logreg", "xgboost", "lightgbm"):
        for strategy in ("none", "weights", "smote"):
            trained = models.train(
                name, strategy, x_train, y_train, x_val, y_val,
                seed=0, n_jobs=1
            )

            # Verify score shape and range
            scores = trained.score(x_test)
            assert scores.shape == (len(x_test),)
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)

            # Verify PR-AUC on separable data is high
            from fraud import metrics as M
            pr_auc = M.pr_auc(y_test, scores)
            assert pr_auc > 0.8, f"{name} {strategy}: PR-AUC {pr_auc:.3f} < 0.8"


def test_resample_none():
    """resample(..., 'none', 0) returns inputs unchanged."""
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.array([0, 0, 1], dtype=np.int64)

    x_out, y_out = models.resample(x, y, "none", 0)

    np.testing.assert_array_equal(x, x_out)
    np.testing.assert_array_equal(y, y_out)


def test_resample_smote_increases_positives():
    """resample(x, y, 'smote', 0) increases the positive count."""
    np.random.seed(42)

    # Create imbalanced data: 1000 samples, 3% positives (~30 positives)
    # SMOTE needs enough samples for k_neighbors to work
    x = np.random.randn(1000, 5).astype(np.float32)
    y = np.zeros(1000, dtype=np.int64)
    n_pos = 30
    y[:n_pos] = 1

    x_out, y_out = models.resample(x, y, "smote", 0)

    # SMOTE should increase positive count to ~10% of majority
    n_pos_out = (y_out == 1).sum()
    n_neg_out = (y_out == 0).sum()

    assert n_pos_out > n_pos
    # With sampling_strategy=0.1, positives should be ~10% of negatives
    assert n_pos_out / n_neg_out > 0.05
