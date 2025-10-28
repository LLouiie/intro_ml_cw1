from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class WiFiDataset:
    features: np.ndarray  # shape (n_samples, n_features)
    labels: np.ndarray    # shape (n_samples,)


def load_wifi_dataset(path: str) -> WiFiDataset:
    """
    Load WiFi dataset where rows are tab- or space-separated values.
    The last column is an integer label (room number), and all others are continuous features.
    """
    try:
        data = np.loadtxt(path, delimiter="\t")
    except ValueError:
        data = np.loadtxt(path)

    # Basic integrity check: must have at least 2 columns (features + label)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + label)")

    # Check for missing values
    if np.isnan(data).any():
        raise ValueError("Dataset contains NaN values.")
    
    # Split features (all columns except last) and labels (last column)
    features = data[:, :-1].astype(np.float64)
    labels = data[:, -1].astype(int)

    return WiFiDataset(features=features, labels=labels)


def k_fold_indices(n_samples: int, k: int = 10, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate k-fold train/test index splits.
    Returns a list of (train_indices, test_indices) tuples.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if n_samples < k:
        raise ValueError("Number of samples must be >= k")
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)    # randomly shuffle indices
    
    folds = np.array_split(indices, k)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        # current fold is test, rest are training
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))
    return splits


