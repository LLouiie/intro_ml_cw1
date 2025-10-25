from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    # Avoid log(0) by masking
    mask = probs > 0
    return float(-np.sum(probs[mask] * np.log2(probs[mask])))


def best_threshold_for_feature(X_col: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Returns (best_gain, threshold)
    Threshold splits: left <= t, right > t
    """
    # Sort by feature values
    order = np.argsort(X_col)
    X_sorted = X_col[order]
    y_sorted = y[order]

    # Candidate thresholds are midpoints between consecutive unique values
    unique_mask = np.r_[True, np.diff(X_sorted) != 0]
    if unique_mask.sum() <= 1:
        return 0.0, float(X_sorted[0])

    # Prefix class counts for efficient entropy updates
    classes = np.unique(y_sorted)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    left_counts = np.zeros(k, dtype=np.int64)
    total_counts = np.bincount([class_to_idx[c] for c in y_sorted], minlength=k)

    n = len(y_sorted)
    best_gain = 0.0
    best_thr = X_sorted[0]
    base_H = entropy(y_sorted)

    # Iterate split points where value changes
    for i in range(n - 1):
        cls_idx = class_to_idx[y_sorted[i]]
        left_counts[cls_idx] += 1
        if X_sorted[i] == X_sorted[i + 1]:
            continue
        right_counts = total_counts - left_counts
        left_n = i + 1
        right_n = n - left_n
        # Weighted entropy
        def H_from_counts(counts: np.ndarray) -> float:
            total = counts.sum()
            if total == 0:
                return 0.0
            probs = counts / total
            mask = probs > 0
            return float(-np.sum(probs[mask] * np.log2(probs[mask])))

        H_left = H_from_counts(left_counts)
        H_right = H_from_counts(right_counts)
        gain = base_H - (left_n / n) * H_left - (right_n / n) * H_right
        if gain > best_gain:
            best_gain = gain
            best_thr = (X_sorted[i] + X_sorted[i + 1]) / 2.0

    return float(best_gain), float(best_thr)


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    prediction: Optional[int] = None
    # for visualization
    num_samples: Optional[int] = None
    impurity: Optional[float] = None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = max_depth
        self.min_samples_split = max(2, int(min_samples_split))
        self.root: Optional[Node] = None
        # Track how often each feature is used for splitting
        self.split_counts_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.split_counts_ = np.zeros(X.shape[1], dtype=int)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model is not fitted")
        preds = [self._predict_one(self.root, row) for row in X]
        return np.asarray(preds, dtype=int)

    # Internal helpers
    def _most_common_label(self, y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node()
        num_samples, num_features = X.shape
        node.num_samples = int(num_samples)
        node.impurity = float(entropy(y))

        # Stopping criteria
        if depth >= self.max_depth or num_samples < self.min_samples_split or node.impurity == 0.0:
            node.prediction = self._most_common_label(y)
            return node

        # Find best split across all features
        best_gain = 0.0
        best_feat = None
        best_thr = None
        for j in range(num_features):
            gain, thr = best_threshold_for_feature(X[:, j], y)
            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_thr = thr

        if best_feat is None or best_gain <= 0.0:
            node.prediction = self._most_common_label(y)
            return node

        # Split data
        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.prediction = self._most_common_label(y)
            return node

        node.feature = int(best_feat)
        node.threshold = float(best_thr)
        if self.split_counts_ is not None:
            self.split_counts_[best_feat] += 1
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _predict_one(self, node: Node, x: np.ndarray) -> int:
        while node.prediction is None:
            assert node.feature is not None and node.threshold is not None
            if x[node.feature] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return int(node.prediction)


