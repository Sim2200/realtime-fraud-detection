"""Load the credit-card fraud dataset and split it by time.

Dataset: OpenML id 1597 ("creditcard"): 284,807 card transactions over two days in
September 2013 by European cardholders, 492 of them fraud (0.172%). Features V1-V28
are PCA components of the original (confidential) fields; Amount is the transaction
amount. It is public and anonymized.

OpenML's copy omits the original `Time` column (seconds since the first transaction),
but its rows are in the original chronological order: row 0 is the first transaction
of the Kaggle file (V1 = -1.3598, Amount = 149.62). So the row position is used as
the time axis (column `seq`) and is never used as a model feature.

Why a time-based split: fraud patterns and the feature distribution drift over time,
and in production the model always scores transactions that happen *after* the ones
it was trained on. A random split would let the model see the future and inflate
every metric. So: the first 70% of transactions in time order are train, the next
10% validation (thresholds, calibration, early stopping), the last 20% test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/creditcard.parquet")
OPENML_ID = 1597
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TIME = "seq"  # row position in the original chronological order
TARGET = "Class"


def download(path: Path = DATA_PATH) -> pd.DataFrame:
    """Fetch from OpenML (no login) and cache as Parquet."""
    from sklearn.datasets import fetch_openml

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(path.parent / "openml_cache")).frame
    frame[TARGET] = frame[TARGET].astype(int)
    frame = frame.reset_index(drop=True)
    frame[TIME] = np.arange(len(frame), dtype=np.int64)
    frame.to_parquet(path)
    return frame


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return download(path)
    df = pd.read_parquet(path)
    df[TARGET] = df[TARGET].astype(int)
    if TIME not in df:
        df[TIME] = np.arange(len(df), dtype=np.int64)
    return df.sort_values(TIME, kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    """Feature matrices and labels for the three time windows, plus the raw frames."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict:
        def part(df: pd.DataFrame) -> dict:
            return {"rows": int(len(df)), "frauds": int(df[TARGET].sum()),
                    "fraud_rate": round(float(df[TARGET].mean()), 6),
                    "seq_from": int(df[TIME].min()), "seq_to": int(df[TIME].max())}
        return {"train": part(self.train), "val": part(self.val), "test": part(self.test)}


def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1,
               features: list[str] = FEATURES) -> Split:
    """Split by position in time order (the `seq` column). No shuffling anywhere."""
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    n = len(df)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    train, val, test = df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]
    as_x = lambda d: d[features].to_numpy(dtype=np.float32)  # noqa: E731
    as_y = lambda d: d[TARGET].to_numpy(dtype=np.int64)  # noqa: E731
    return Split(as_x(train), as_y(train), as_x(val), as_y(val), as_x(test), as_y(test), train, val, test)


def sample(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    """A time-contiguous prefix sample (for CI), keeping all frauds in that prefix."""
    n = int(len(df) * frac)
    return df.sort_values(TIME, kind="stable").iloc[:n].reset_index(drop=True)
