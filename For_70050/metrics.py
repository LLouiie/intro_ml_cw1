from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from wifi_utils import k_fold_indices


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    m = len(classes)
    cm = np.zeros((m, m), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[int(t)], class_to_idx[int(p)]] += 1
    return cm, list(map(int, classes.tolist()))


def cross_val_scores(model_factory, X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> List[float]:
    scores: List[float] = []
    for train_idx, val_idx in k_fold_indices(len(y), k=k, seed=seed):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        scores.append(accuracy(y[val_idx], y_pred))
    return scores


