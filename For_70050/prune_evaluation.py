from pathlib import Path
from typing import List, Tuple
import numpy as np

from decision_tree import DecisionTreeClassifier
from metrics import accuracy, confusion_matrix, recall_precision_f1
from visualize import plot_confusion_matrix, plot_tree
from wifi_utils import k_fold_indices


def nested_cv_evaluate_pruned(
    X: np.ndarray,
    y: np.ndarray,
    outer_k: int = 10,
    inner_k: int = 9,
    seed: int = 42
) -> Tuple[List[float], List[np.ndarray], List[List[int]], List[int], List[int]]:
    """
    Perform nested cross-validation for evaluating pruned decision trees.

    The outer loop defines test folds, and the inner loop trains and prunes models
    using validation folds. Each inner model is evaluated on the corresponding outer test fold.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label array of shape (n_samples,).
        outer_k (int, optional): Number of outer folds for testing. Default is 10.
        inner_k (int, optional): Number of inner folds for training/pruning. Default is 9.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[List[float], List[np.ndarray], List[List[int]], List[int], List[int]]:
            - List[float]: Test accuracies across all outer and inner folds.
            - List[np.ndarray]: Confusion matrices for each evaluation.
            - List[List[int]]: Corresponding class label lists for each matrix.
            - List[int]: Tree depths before pruning.
            - List[int]: Tree depths after pruning.
    """
    n_samples = len(y)
    outer_splits = k_fold_indices(n_samples, k=outer_k, seed=seed)

    # Containers for all evaluation results
    all_test_accuracies = []
    all_confusion_matrices = []
    all_classes = []
    all_depths_before = []
    all_depths_after = []

    # Outer loop: defines test folds
    for outer_fold_idx, (outer_train_val_idx, test_idx) in enumerate(outer_splits):
        print(f"Outer fold {outer_fold_idx + 1}/{outer_k}")

        # Outer test data
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Inner loop: for training and pruning using validation folds
        inner_splits = k_fold_indices(len(outer_train_val_idx), k=inner_k, seed=seed + outer_fold_idx)

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
            # Map inner indices back to original dataset indices
            actual_train_idx = outer_train_val_idx[inner_train_idx]
            actual_val_idx = outer_train_val_idx[inner_val_idx]

            X_train = X[actual_train_idx]
            y_train = y[actual_train_idx]
            X_val = X[actual_val_idx]
            y_val = y[actual_val_idx]

            # Train and prune model
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            depth_before = model.get_depth()

            # Perform pruning using validation set
            model.prune(X_val, y_val)
            depth_after = model.get_depth()

            # Record depth information
            all_depths_before.append(depth_before)
            all_depths_after.append(depth_after)

            # Evaluate pruned model on the outer test fold
            y_pred = model.predict(X_test)
            test_acc = accuracy(y_test, y_pred)
            cm, classes = confusion_matrix(y_test, y_pred)

            # Store evaluation results
            all_test_accuracies.append(test_acc)
            all_confusion_matrices.append(cm)
            all_classes.append(classes)

    return all_test_accuracies, all_confusion_matrices, all_classes, all_depths_before, all_depths_after


def aggregate_confusion_matrices(all_cms: List[np.ndarray], all_classes: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
    """
    Aggregate multiple confusion matrices into a single global matrix.

    Handles class index alignment across different folds, as class sets may differ
    between individual models due to stratified or unbalanced splits.

    Args:
        all_cms (List[np.ndarray]): List of confusion matrices from each fold.
        all_classes (List[List[int]]): List of class label lists corresponding to each confusion matrix.

    Returns:
        Tuple[np.ndarray, List[int]]:
            - np.ndarray: Aggregated confusion matrix across all folds.
            - List[int]: Sorted list of unique class labels.
    """
    # Collect all unique class labels across folds
    all_class_values = set()
    for classes in all_classes:
        all_class_values.update(classes)
    unique_classes = sorted(list(all_class_values))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    n_classes = len(unique_classes)

    # Initialize empty aggregated confusion matrix
    aggregated_cm = np.zeros((n_classes, n_classes), dtype=int)

    # Map local class indices to global indices and accumulate counts
    for cm, classes in zip(all_cms, all_classes):
        local_to_global = {local_cls: class_to_idx[local_cls] for local_cls in classes}
        for local_true_idx, true_cls in enumerate(classes):
            for local_pred_idx, pred_cls in enumerate(classes):
                global_true_idx = local_to_global[true_cls]
                global_pred_idx = local_to_global[pred_cls]
                aggregated_cm[global_true_idx, global_pred_idx] += cm[local_true_idx, local_pred_idx]

    return aggregated_cm, unique_classes


def run_prune_evaluation(clean_dataset, noisy_dataset, outdir: Path):
    """
    Run nested cross-validation for pruned trees on both clean and noisy datasets.

    The function performs the following steps:
    1. Runs nested CV pruning evaluation for both datasets.
    2. Prints overall statistics (accuracy, depth changes, confusion matrix).
    3. Saves confusion matrix plots (both count and normalized).
    4. Visualizes final pruned trees for both datasets.

    Args:
        clean_dataset (WiFiDataset): Dataset containing clean data (features & labels).
        noisy_dataset (WiFiDataset): Dataset containing noisy data.
        outdir (Path): Directory where generated plots will be saved.

    Returns:
        None. The function prints results and saves figures.
    """
    datasets = {"clean": clean_dataset, "noisy": noisy_dataset}

    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*70}")
        print(f"Running nested CV for pruned trees on {dataset_name} dataset...")
        print(f"{'='*70}")

        # Run nested cross-validation evaluation
        all_accs, all_cms, all_classes, depths_before, depths_after = nested_cv_evaluate_pruned(
            dataset.features, dataset.labels, outer_k=10, inner_k=9, seed=42
        )

        # Print summary statistics
        print(f"\nResults for {dataset_name} dataset:")
        print(f"Total test results: {len(all_accs)}")
        print(f"Mean accuracy: {np.mean(all_accs):.4f}")
        print(f"Std accuracy: {np.std(all_accs):.4f}")
        print(f"Min: {np.min(all_accs):.4f}, Max: {np.max(all_accs):.4f}")

        # Aggregate confusion matrices from all folds
        agg_cm, classes = aggregate_confusion_matrices(all_cms, all_classes)
        print(f"\nAggregated confusion matrix:")
        print(agg_cm)

        # Compute recall, precision, and F1 metrics
        recall_dict, precision_dict, f1_dict = recall_precision_f1(agg_cm, classes)

        print(f"\nPer-class metrics:")
        for cls in sorted(classes):
            print(f"  Class {cls}: Recall={recall_dict[cls]:.4f}, "
                  f"Precision={precision_dict[cls]:.4f}, F1={f1_dict[cls]:.4f}")

        # Save confusion matrix plots
        plot_confusion_matrix(agg_cm, classes, outdir / f"cm_{dataset_name}_pruned_counts.png", normalize=False)
        plot_confusion_matrix(agg_cm, classes, outdir / f"cm_{dataset_name}_pruned_normalized.png", normalize=True)

        # Analyze and display tree depth reduction before/after pruning
        print(f"\nDepth Analysis for {dataset_name}:")
        print(f"  Before pruning: mean={np.mean(depths_before):.2f}, "
              f"min={np.min(depths_before)}, max={np.max(depths_before)}")
        print(f"  After pruning: mean={np.mean(depths_after):.2f}, "
              f"min={np.min(depths_after)}, max={np.max(depths_after)}")
        print(f"  Avg reduction: {np.mean(depths_before) - np.mean(depths_after):.2f}")

    # Visualize example pruned trees for both datasets
    print(f"\n{'='*70}")
    print("Visualizing pruned trees for both datasets...")

    for ds_name, ds in [("clean", clean_dataset), ("noisy", noisy_dataset)]:
        # Randomly split dataset into training and validation sets
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(ds.labels))
        split_idx = int(0.8 * len(ds.labels))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        # Train and prune a single tree
        model = DecisionTreeClassifier()
        model.fit(ds.features[train_idx], ds.labels[train_idx])
        model.prune(ds.features[val_idx], ds.labels[val_idx])

        # Save visualization
        out_path = outdir / f"tree_{ds_name}_pruned.png"
        plot_tree(model.root, out_path)
        print(f"Saved: {out_path.name}")
