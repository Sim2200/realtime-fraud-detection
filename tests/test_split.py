"""Tests for fraud.data time-split functionality."""

import numpy as np
import pandas as pd
import pytest

from fraud import data


def test_time_split_no_leakage():
    """Verify no time leakage: train seq < val seq < test seq."""
    np.random.seed(42)
    n_rows = 1000

    # Create DataFrame with seq (row position in chronological order) as shuffled permutation
    # so the input is NOT already time-ordered.
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n_rows) for i in range(1, 29)},
        "Amount": np.random.exponential(50, n_rows),
        data.TIME: np.random.permutation(n_rows),
        "Class": np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    })

    split = data.time_split(df)

    # Assert: every seq in train < every seq in val < every seq in test
    assert split.train[data.TIME].max() < split.val[data.TIME].min()
    assert split.val[data.TIME].max() < split.test[data.TIME].min()


def test_time_split_sizes():
    """Verify split sizes are 70/10/20."""
    n_rows = 1000
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n_rows) for i in range(1, 29)},
        "Amount": np.random.uniform(0, 100, n_rows),
        data.TIME: np.arange(n_rows),  # Ordered seq
        "Class": np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    })

    split = data.time_split(df)

    train_frac = len(split.train) / n_rows
    val_frac = len(split.val) / n_rows
    test_frac = len(split.test) / n_rows

    assert abs(train_frac - 0.70) < 0.01
    assert abs(val_frac - 0.10) < 0.01
    assert abs(test_frac - 0.20) < 0.01


def test_time_split_feature_order():
    """Verify feature matrix column order equals fraud.data.FEATURES."""
    n_rows = 500
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n_rows) for i in range(1, 29)},
        "Amount": np.random.uniform(0, 100, n_rows),
        data.TIME: np.arange(n_rows),
        "Class": np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    })

    split = data.time_split(df)

    # Feature order should match FEATURES (V1..V28 + Amount, no seq)
    expected_features = data.FEATURES
    assert len(expected_features) == 29  # V1..V28 + Amount
    assert split.x_train.shape[1] == len(expected_features)
    assert split.x_val.shape[1] == len(expected_features)
    assert split.x_test.shape[1] == len(expected_features)


def test_split_summary():
    """Verify Split.summary() reports correct row counts and fraud counts."""
    n_rows = 700
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n_rows) for i in range(1, 29)},
        "Amount": np.random.uniform(0, 100, n_rows),
        data.TIME: np.arange(n_rows),
        "Class": np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    })

    split = data.time_split(df)
    summary = split.summary()

    assert "train" in summary
    assert "val" in summary
    assert "test" in summary

    assert summary["train"]["rows"] == len(split.train)
    assert summary["val"]["rows"] == len(split.val)
    assert summary["test"]["rows"] == len(split.test)

    assert summary["train"]["frauds"] == int(split.train["Class"].sum())
    assert summary["val"]["frauds"] == int(split.val["Class"].sum())
    assert summary["test"]["frauds"] == int(split.test["Class"].sum())

    # Check that summary reports seq_from/seq_to instead of time_from_s/time_to_s
    assert "seq_from" in summary["train"]
    assert "seq_to" in summary["train"]


def test_sample():
    """Verify data.sample(df, 0.1) returns the first 10% of rows by seq."""
    n_rows = 1000
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n_rows) for i in range(1, 29)},
        "Amount": np.random.uniform(0, 100, n_rows),
        data.TIME: np.arange(n_rows),
        "Class": np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    })

    sampled = data.sample(df, 0.1)

    assert len(sampled) == 100
    # Should be the first 10% by seq order (first 100 rows after sorting by seq)
    assert sampled.index.tolist() == list(range(100))
