"""Differentially private training with Opacus (DP-SGD) on a small MLP.

What DP-SGD protects: the trained weights (and therefore the model's scores)
change by a bounded amount if any single transaction is added to or removed from
the training set. Epsilon is the privacy budget: smaller is stronger. The
mechanism is per-example gradient clipping plus calibrated Gaussian noise, with
the budget accounted by Opacus's RDP accountant at delta = 1 / n_train.

What it does NOT protect: the test data, the features themselves (they are already
PCA-anonymised by the data owner), or anything the model is later queried with.
It also says nothing about attacks on the raw data store. The README states this.

The same MLP is trained without DP (epsilon = inf) so the utility cost is measured
against an identical architecture, not against the tree models.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from opacus import PrivacyEngine
from sklearn.preprocessing import StandardScaler
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPClassifier:
    """Plain or DP-SGD MLP with class-weighted BCE. epsilon=None means no privacy (ordinary SGD)."""

    def __init__(self, epsilon: float | None = None, epochs: int = 8, batch_size: int = 1024, lr: float = 1e-3,
                 max_grad_norm: float = 1.0, seed: int = 0, pos_weight: float | None = None) -> None:
        self.epsilon, self.epochs, self.batch_size, self.lr = epsilon, epochs, batch_size, lr
        self.max_grad_norm, self.seed, self.pos_weight = max_grad_norm, seed, pos_weight
        self.scaler = StandardScaler()
        self.model: MLP | None = None
        self.fit_seconds = 0.0
        self.epsilon_spent: float | None = None
        self.noise_multiplier: float | None = None
        self.delta: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        torch.manual_seed(self.seed)
        t0 = time.perf_counter()
        xs = torch.tensor(self.scaler.fit_transform(x), dtype=torch.float32)
        ys = torch.tensor(y, dtype=torch.float32)
        self.model = MLP(xs.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=self.batch_size, shuffle=True)
        pw = self.pos_weight if self.pos_weight is not None else float((ys == 0).sum() / max(1.0, float(ys.sum())))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
        if self.epsilon is not None:
            self.delta = 1.0 / len(xs)
            engine = PrivacyEngine(accountant="rdp")
            self.model, opt, loader = engine.make_private_with_epsilon(
                module=self.model, optimizer=opt, data_loader=loader, epochs=self.epochs,
                target_epsilon=self.epsilon, target_delta=self.delta, max_grad_norm=self.max_grad_norm)
            self.noise_multiplier = float(opt.noise_multiplier)
        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        if self.epsilon is not None:
            self.epsilon_spent = float(engine.get_epsilon(self.delta))
        self.fit_seconds = round(time.perf_counter() - t0, 2)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        xs = torch.tensor(self.scaler.transform(x), dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.model(xs)).numpy()

    def plain_module(self) -> nn.Module:
        """The underlying nn.Module (Opacus wraps it in GradSampleModule)."""
        m = self.model
        return m._module if hasattr(m, "_module") else m
