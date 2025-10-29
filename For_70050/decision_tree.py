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
    hello
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
    
   


class DecisionTreeClassifier:
    def __init__(self):
        self.root: Optional[Node] = None      
        self.max_depth: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.root, self.max_depth = self._build_tree(X, y, depth=0)
        return self
    
    def get_depth(self) -> int:
        """Get the maximum depth of the tree."""
        return self.max_depth
    
    def _compute_depth(self, node: Optional[Node]) -> int:
        """Recursively compute maximum depth of tree."""
        if node is None:
            return 0
        if node.prediction is not None:
            return 0  # Leaf node contributes 0 to depth
        left_depth = self._compute_depth(node.left) if node.left else 0
        right_depth = self._compute_depth(node.right) if node.right else 0
        return 1 + max(left_depth, right_depth)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model is not fitted")
        preds = [self._predict_one(self.root, row) for row in X]
        return np.asarray(preds, dtype=int)

    # Internal helpers
    def _most_common_label(self, y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Tuple[Node, int]:
        """
        Returns (node, actual_depth) as per PDF algorithm specification.
        """
        node = Node()
        num_features = X.shape[1]

        # Base case: all samples have the same label (leaf node)
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            node.prediction = int(unique_labels[0])
            return node, depth

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
            return node, depth

        # Split data
        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.prediction = self._most_common_label(y)
            return node, depth

        # Create node with split feature and threshold
        node.feature = int(best_feat)
        node.threshold = float(best_thr)

        # Left branch (recursive): lbranch, l_depth ← DECISION TREE LEARNING (ldataset, depth+1)
        node.left, l_depth = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        
        # Right branch (recursive): rbranch, r_depth ← DECISION TREE LEARNING (rdataset, depth+1)
        node.right, r_depth = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Return (node, max(l_depth, r_depth)) as per PDF
        return node, max(l_depth, r_depth)

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

    def prune(self, X_val: np.ndarray, y_val: np.ndarray) -> "DecisionTreeClassifier":
        """
        Prune the decision tree based on validation error.
        
        For each node directly connected to two leaves, evaluate if replacing it with
        a single leaf reduces or maintains validation error.
        Decision: prune if validation error on pruned tree <= error on original tree.
        Iterates until no improvements can be made.
        
        Args:
            X_val: Validation features (used to evaluate each pruning decision)
            y_val: Validation labels (used to evaluate each pruning decision)
            
        Returns:
            self (allows chaining)
        """
        if self.root is None:
            raise RuntimeError("Model is not fitted")
        
        # Iterate pruning until no improvements can be made
        changed = True
        while changed:
            changed = self._prune_one_pass(self.root, X_val, y_val)
        
        # Update depth after pruning
        self.max_depth = self._compute_depth(self.root)
        
        return self
    
    def _prune_one_pass(self, node: Node, X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """
        Perform one pass of bottom-up pruning.
        Returns True if any pruning occurred in this pass.
        """
        if node.prediction is not None:
            # Leaf node, nothing to prune
            return False
        
        # First, recursively prune children (bottom-up)
        pruned_left = self._prune_one_pass(node.left, X_val, y_val) if node.left else False
        pruned_right = self._prune_one_pass(node.right, X_val, y_val) if node.right else False
        
        # Now check if current node can be pruned
        # Node can be pruned if it's directly connected to two leaves
        if self._is_leaf(node.left) and self._is_leaf(node.right):
            # Calculate majority class from the two leaf predictions
            majority_pred = self._get_majority_from_leaves(node.left, node.right)
            
            # Store current state
            original_pred = node.prediction
            original_feat = node.feature
            original_thr = node.threshold
            original_left = node.left
            original_right = node.right
            
            # Calculate error with original node first
            original_error = self._evaluate_error(X_val, y_val)
            
            # Temporarily replace node with leaf
            node.prediction = majority_pred
            node.feature = None
            node.threshold = None
            node.left = None
            node.right = None
            
            # Evaluate error with pruned node
            pruned_error = self._evaluate_error(X_val, y_val)
            
            # Restore original node
            node.prediction = original_pred
            node.feature = original_feat
            node.threshold = original_thr
            node.left = original_left
            node.right = original_right
            
            # Prune if error does not increase (reduces or stays same)
            if pruned_error <= original_error:
                # Actually prune the node
                node.prediction = majority_pred
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                return True
            
        return pruned_left or pruned_right
    
    def _get_majority_from_leaves(self, left: Node, right: Node) -> int:
        """
        Get majority class from two leaf nodes.
        If predictions are different, prefer left (arbitrary choice).
        """
        if left.prediction == right.prediction:
            return left.prediction
        # If different, return left prediction (could use counts if available)
        return left.prediction
    
    def _is_leaf(self, node: Optional[Node]) -> bool:
        """Check if node is a leaf."""
        if node is None:
            return False
        return node.prediction is not None
    
    def _evaluate_error(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Evaluate classification error on validation set.
        Returns 1 - accuracy (error rate).
        """
        if self.root is None:
            return 1.0
        
        y_pred = self.predict(X_val)
        # Return error rate (1 - accuracy)
        return float(1.0 - np.mean(y_val == y_pred))


