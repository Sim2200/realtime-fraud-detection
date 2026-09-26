"""Unsupervised anomaly detection, trained on legitimate transactions only.

Both detectors never see a fraud label. That is the realistic setting when a new
fraud pattern appears before anyone has labelled it.

  IsolationForest   random partitioning; anomalies isolate in few splits
  Autoencoder       a small PyTorch MLP trained to reconstruct legitimate rows;
                    the reconstruction error is the anomaly score
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn


def isolation_forest(x_train_legit: np.ndarray, seed: int = 0) -> tuple[IsolationForest, float]:
    t0 = time.perf_counter()
    model = IsolationForest(n_estimators=300, max_samples=4096, contamination="auto", random_state=seed, n_jobs=4)
    model.fit(x_train_legit)
    return model, round(time.perf_counter() - t0, 2)


def isolation_scores(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    return -model.score_samples(x)  # higher = more anomalous


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int, bottleneck: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                                     nn.Linear(16, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(),
                                     nn.Linear(32, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoEncoderDetector:
    """Standardise features, train the AE on legitimate rows, score by reconstruction error."""

    def __init__(self, seed: int = 0, epochs: int = 20, batch_size: int = 512, lr: float = 1e-3) -> None:
        self.seed, self.epochs, self.batch_size, self.lr = seed, epochs, batch_size, lr
        self.scaler = StandardScaler()
        self.model: AutoEncoder | None = None
        self.fit_seconds = 0.0
        self.history: list[float] = []

    def fit(self, x_legit: np.ndarray, x_val_legit: np.ndarray | None = None) -> "AutoEncoderDetector":
        torch.manual_seed(self.seed)
        t0 = time.perf_counter()
        xs = torch.tensor(self.scaler.fit_transform(x_legit), dtype=torch.float32)
        self.model = AutoEncoder(xs.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs), batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            self.model.train()
            total = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = ((self.model(batch) - batch) ** 2).mean()
                loss.backward()
                opt.step()
                total += float(loss) * len(batch)
            self.history.append(total / len(xs))
        self.fit_seconds = round(time.perf_counter() - t0, 2)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        xs = torch.tensor(self.scaler.transform(x), dtype=torch.float32)
        with torch.no_grad():
            err = ((self.model(xs) - xs) ** 2).mean(dim=1)
        return err.numpy()

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Bottleneck representation, used for clustering."""
        assert self.model is not None
        self.model.eval()
        xs = torch.tensor(self.scaler.transform(x), dtype=torch.float32)
        with torch.no_grad():
            return self.model.encoder(xs).numpy()
