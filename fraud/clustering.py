"""K-means segments over the autoencoder embedding.

Fit on the training window, then report the fraud rate per cluster on train and
test. A cluster whose fraud rate is many times the base rate is a "fraud-dense
segment"; the cluster id is then tried as an extra feature for the best
supervised model to see whether it adds anything the trees cannot already learn.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fit_kmeans(z_train: np.ndarray, k: int = 12, seed: int = 0) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z_train)


def cluster_report(km: KMeans, z: np.ndarray, y: np.ndarray) -> list[dict]:
    labels = km.predict(z)
    base = float(y.mean())
    rows = []
    for c in range(km.n_clusters):
        m = labels == c
        rate = float(y[m].mean()) if m.sum() else 0.0
        rows.append({"cluster": c, "rows": int(m.sum()), "frauds": int(y[m].sum()), "fraud_rate": round(rate, 5),
                     "lift_vs_base": round(rate / base, 1) if base else None})
    return sorted(rows, key=lambda r: -r["fraud_rate"])


def one_hot_cluster(km: KMeans, z: np.ndarray) -> np.ndarray:
    labels = km.predict(z)
    out = np.zeros((len(z), km.n_clusters), dtype=np.float32)
    out[np.arange(len(z)), labels] = 1.0
    return out
