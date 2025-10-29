from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class WiFiDataset:
    """
    Container for a WiFi localization dataset.

    Attributes:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Label array of shape (n_samples,).
    """
    features: np.ndarray  # shape (n_samples, n_features)
    labels: np.ndarray    # shape (n_samples,)


def load_wifi_dataset(path: str) -> WiFiDataset:
    """
    Load a WiFi dataset from text files.

    Each row should represent one sample:
    - All columns except the last are continuous features.
    - The last column is an integer label representing the room number.

    Args:
        path (str): Path to the dataset file.

    Returns:
        WiFiDataset: 
            A dataclass containing:
            - features (np.ndarray): Continuous features of shape (n_samples, n_features).
            - labels (np.ndarray): Integer class labels of shape (n_samples,).

    Raises:
        ValueError: If the dataset has fewer than 2 columns (needs at least one feature and one label),
            or if it contains NaN values.
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
    Generate deterministic k-fold train/test splits for cross-validation.

    Randomly shuffles all sample indices using the given seed, then partitions them into k folds.
    Each fold is used once as the test set while the remaining folds form the training set.

    Args:
        n_samples (int): Total number of samples in the dataset.
        k (int, optional): Number of folds for cross-validation. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: 
            A list of length k, where each element is a tuple:
            (train_indices, test_indices), both 1D NumPy arrays of integers.

    Raises:
        ValueError: If k < 2, or if n_samples < k.
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


