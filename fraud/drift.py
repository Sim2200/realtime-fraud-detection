"""Population Stability Index between the training window and later windows.

PSI = sum over bins of (p_later - p_train) * ln(p_later / p_train), with bins set
on the training distribution (deciles). Common reading: < 0.1 stable, 0.1-0.25
moderate shift, > 0.25 significant shift. Computed per feature and on the model
score, so both input drift and output drift are visible.
"""

from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, edges)[0] / len(reference)
    p_cur = np.histogram(current, edges)[0] / len(current)
    p_ref, p_cur = np.clip(p_ref, eps, None), np.clip(p_cur, eps, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def label(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "significant"


def feature_drift(x_ref: np.ndarray, x_cur: np.ndarray, names: list[str]) -> list[dict]:
    rows = [{"feature": n, "psi": round(psi(x_ref[:, i], x_cur[:, i]), 4)} for i, n in enumerate(names)]
    for r in rows:
        r["status"] = label(r["psi"])
    return sorted(rows, key=lambda r: -r["psi"])
