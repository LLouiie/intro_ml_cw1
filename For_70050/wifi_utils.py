from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class WiFiDataset:
    features: np.ndarray  # shape (n_samples, n_features)
    labels: np.ndarray    # shape (n_samples,)


def load_wifi_dataset(path: str) -> WiFiDataset:
    """
    Load WiFi dataset where rows are tab- or space-separated values with the last
    column as integer class label and the rest as continuous features.
    """
    # Robust loading: try tab-delimited first, fallback to space-delimited floats
    try:
        data = np.loadtxt(path, delimiter="\t")
    except ValueError:
        data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + label)")

    features = data[:, :-1].astype(np.float64)
    labels = data[:, -1].astype(int)
    return WiFiDataset(features=features, labels=labels)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int | None = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * (1.0 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_indices(n_samples: int, k: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


