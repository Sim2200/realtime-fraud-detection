"""Supervised baselines and the three ways of handling class imbalance.

Models: logistic regression, Random Forest, XGBoost, LightGBM.
Imbalance strategies, each applied to the training window only:
  none      train as-is (0.17% positives)
  weights   class_weight / scale_pos_weight so a fraud counts ~ (neg/pos) times more
  smote     SMOTE oversampling of the minority class to 10% of the majority, on the
            training set only, so validation/test stay at the real rate
Early stopping and threshold choice use the validation window, never test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODELS = ("logreg", "random_forest", "xgboost", "lightgbm")
STRATEGIES = ("none", "weights", "smote")


@dataclass
class Trained:
    name: str
    strategy: str
    model: object
    fit_seconds: float
    n_train: int
    extra: dict = field(default_factory=dict)

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


def make_model(name: str, strategy: str, pos_weight: float, seed: int, n_jobs: int = 4):
    w = pos_weight if strategy == "weights" else 1.0
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight={0: 1.0, 1: w} if strategy == "weights" else None,
                                 random_state=seed)
        return make_pipeline(StandardScaler(), clf)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2, n_jobs=n_jobs,
                                      class_weight={0: 1.0, 1: w} if strategy == "weights" else None,
                                      random_state=seed)
    if name == "xgboost":
        return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=1, scale_pos_weight=w, eval_metric="aucpr", n_jobs=n_jobs,
                             random_state=seed, tree_method="hist", early_stopping_rounds=50)
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, scale_pos_weight=w, n_jobs=n_jobs, random_state=seed, verbose=-1)
    raise ValueError(name)


def resample(x: np.ndarray, y: np.ndarray, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if strategy != "smote":
        return x, y
    sm = SMOTE(sampling_strategy=0.1, random_state=seed, k_neighbors=5)
    return sm.fit_resample(x, y)


def train(name: str, strategy: str, x_train: np.ndarray, y_train: np.ndarray,
          x_val: np.ndarray, y_val: np.ndarray, seed: int = 0, n_jobs: int = 4) -> Trained:
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    x_fit, y_fit = resample(x_train, y_train, strategy, seed)
    model = make_model(name, strategy, pos_weight, seed, n_jobs)
    t0 = time.perf_counter()
    extra: dict = {}
    if name == "xgboost":
        model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
        extra["best_iteration"] = int(model.best_iteration)
    elif name == "lightgbm":
        import lightgbm as lgb
        model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        extra["best_iteration"] = int(model.best_iteration_)
    else:
        model.fit(x_fit, y_fit)
    return Trained(name, strategy, model, round(time.perf_counter() - t0, 2), int(len(y_fit)), extra)
