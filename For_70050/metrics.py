from typing import Dict, List, Tuple

import numpy as np

from wifi_utils import k_fold_indices


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the classification accuracy between true and predicted labels.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the confusion matrix for multi-class classification.
    """
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


def precision_recall_f1(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class precision, recall, and F1 scores from confusion matrix.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp, dtype=float),
                       where=(precision + recall) != 0)
    return precision, recall, f1


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Evaluate model performance for one test set.
    Returns accuracy, confusion matrix, precision, recall, and f1 arrays.
    """
    acc = accuracy(y_true, y_pred)
    cm, classes = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(cm)
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classes": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cross_val_evaluate(model_factory, X: np.ndarray, y: np.ndarray, k: int = 10, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation and aggregate metrics across folds.
    Returns average confusion matrix, mean accuracy, and per-class averaged metrics.
    """
    folds = k_fold_indices(len(y), k=k, seed=seed)
    cms, accuracies, precisions, recalls, f1s = [], [], [], [], []

    for train_idx, test_idx in folds:
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        res = evaluate(y[test_idx], y_pred)
        cms.append(res["confusion_matrix"])
        accuracies.append(res["accuracy"])
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f1s.append(res["f1"])

    mean_cm = np.sum(cms, axis=0)  
    mean_acc = float(np.mean(accuracies))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_f1 = np.mean(f1s, axis=0)

    return {
        "mean_confusion_matrix": mean_cm,
        "mean_accuracy": mean_acc,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
    }
