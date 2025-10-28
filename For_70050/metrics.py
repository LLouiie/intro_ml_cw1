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


def recall_precision_f1(cm: np.ndarray, classes: List[int]) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Calculate recall, precision, and F1 for each class from confusion matrix.
    
    Args:
        cm: Confusion matrix (n_classes x n_classes)
        classes: List of class labels
    
    Returns:
        recall_dict, precision_dict, f1_dict: Dictionaries mapping class -> metric
    """
    n_classes = len(classes)
    recall_dict = {}
    precision_dict = {}
    f1_dict = {}
    
    for i, cls in enumerate(classes):
        # True positives: cm[i, i]
        # False negatives: sum of row i excluding diagonal
        # False positives: sum of column i excluding diagonal
        
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_dict[cls] = float(recall)
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision_dict[cls] = float(precision)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_dict[cls] = float(f1)
    
    return recall_dict, precision_dict, f1_dict


def cross_val_scores(model_factory, X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> List[float]:
    scores: List[float] = []
    for train_idx, val_idx in k_fold_indices(len(y), k=k, seed=seed):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        scores.append(accuracy(y[val_idx], y_pred))
    return scores


