"""Fraud detection on the OpenML credit-card dataset.

Two rules for the OpenMP runtimes on macOS, learned the hard way:
1. XGBoost and LightGBM must be imported before PyTorch (the reverse order segfaults).
2. PyTorch must run single-threaded once the tree libraries are loaded (its
   multi-threaded training loop deadlocks otherwise). The autoencoder and the MLPs
   here are small, so this costs little.
"""

import lightgbm  # noqa: F401  (must precede torch, see above)
import xgboost  # noqa: F401
import torch

torch.set_num_threads(1)
