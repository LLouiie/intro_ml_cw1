from typing import Dict, List, Tuple
import numpy as np
from wifi_utils import k_fold_indices


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the classification accuracy between true and predicted labels.

    Args:
        y_true (np.ndarray): Ground-truth labels, shape (n_samples,).
        y_pred (np.ndarray): Predicted labels, shape (n_samples,).

    Returns:
        float: Classification accuracy, i.e. the proportion of correct predictions.
    """
    # Ensure arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")

    # Handle empty input
    if y_true.size == 0:
        return 0.0

    # Mean of boolean equality gives the accuracy
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the confusion matrix for a multi-class classification problem.

    Args:
        y_true (np.ndarray): Ground-truth labels, shape (n_samples,).
        y_pred (np.ndarray): Predicted labels, shape (n_samples,).

    Returns:
        Tuple[np.ndarray, List[int]]:
            - np.ndarray: Confusion matrix of shape (n_classes, n_classes),
              where entry (i, j) counts samples with true label i and predicted label j.
            - List[int]: Sorted list of unique class labels corresponding to matrix indices.
    """
    # Determine all unique class labels that appear in true or predicted arrays
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_to_idx = {c: i for i, c in enumerate(classes)}  # map class label to matrix index
    m = len(classes)

    # Initialize an empty confusion matrix
    cm = np.zeros((m, m), dtype=int)

    # For each sample, increment the (true_label, predicted_label) cell
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[int(t)], class_to_idx[int(p)]] += 1

    # Return matrix and label order
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
    Compute per-class precision, recall, and F1 score from a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix of shape (n_classes, n_classes).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - precision (np.ndarray): Per-class precision scores.
            - recall (np.ndarray): Per-class recall scores.
            - f1 (np.ndarray): Per-class F1 scores.
    """
    # Temporarily ignore division warnings when dividing by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        # True positives are diagonal elements
        tp = np.diag(cm)
        # False positives: predicted as class j but true label ≠ j
        fp = np.sum(cm, axis=0) - tp
        # False negatives: true label = i but predicted ≠ i
        fn = np.sum(cm, axis=1) - tp

        # Compute precision = TP / (TP + FP)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        # Compute recall = TP / (TP + FN)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        # Compute F1 = 2 * precision * recall / (precision + recall)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp, dtype=float),
                       where=(precision + recall) != 0)
    return precision, recall, f1


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Evaluate classification performance on a single test set.

    Args:
        y_true (np.ndarray): Ground-truth labels, shape (n_samples,).
        y_pred (np.ndarray): Predicted labels, shape (n_samples,).

    Returns:
        Dict[str, np.ndarray]: Dictionary containing evaluation results:
            - "accuracy" (float): Overall accuracy score.
            - "confusion_matrix" (np.ndarray): Confusion matrix.
            - "classes" (List[int]): Class label order used in the matrix.
            - "precision" (np.ndarray): Per-class precision values.
            - "recall" (np.ndarray): Per-class recall values.
            - "f1" (np.ndarray): Per-class F1 values.
    """
    # Compute all evaluation metrics
    acc = accuracy(y_true, y_pred)
    cm, classes = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(cm)

    # Return a dictionary with all results
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
    Perform k-fold cross-validation and aggregate classification metrics.

    Each fold trains the model on (k-1)/k of the data and evaluates it
    on the remaining 1/k held-out fold. The metrics are averaged across folds.

    Args:
        model_factory (Callable[[], Any]): A callable (e.g. class or lambda)
            that returns a new, unfitted model instance supporting `fit()` and `predict()`.
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Ground-truth labels, shape (n_samples,).
        k (int, optional): Number of folds for cross-validation. Default is 10.
        seed (int, optional): Random seed for fold shuffling. Default is 42.

    Returns:
        Dict[str, np.ndarray]: Dictionary of aggregated results:
            - "mean_confusion_matrix" (np.ndarray): Average of confusion matrices across folds.
            - "mean_accuracy" (float): Mean accuracy across folds.
            - "mean_precision" (np.ndarray): Mean per-class precision.
            - "mean_recall" (np.ndarray): Mean per-class recall.
            - "mean_f1" (np.ndarray): Mean per-class F1 score.
    """
    # Generate k folds using helper function
    folds = k_fold_indices(len(y), k=k, seed=seed)

    # Containers to store metrics from all folds
    cms, accuracies, precisions, recalls, f1s = [], [], [], [], []

    # Iterate through all folds
    for train_idx, test_idx in folds:
        # Create a fresh model instance
        model = model_factory()
        # Train model on (k-1) folds
        model.fit(X[train_idx], y[train_idx])
        # Predict on held-out fold
        y_pred = model.predict(X[test_idx])

        # Evaluate metrics on this fold
        res = evaluate(y[test_idx], y_pred)

        # Collect metrics for later averaging
        cms.append(res["confusion_matrix"])
        accuracies.append(res["accuracy"])
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f1s.append(res["f1"])

    # Average confusion matrices
    mean_cm = np.mean(cms, axis=0)
    # Average scalar and vector metrics
    mean_acc = float(np.mean(accuracies))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_f1 = np.mean(f1s, axis=0)

    # Return dictionary of aggregated metrics
    return {
        "mean_confusion_matrix": mean_cm,
        "mean_accuracy": mean_acc,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
    }
