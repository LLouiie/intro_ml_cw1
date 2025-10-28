import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

from decision_tree import DecisionTreeClassifier
from metrics import accuracy, confusion_matrix, cross_val_scores, recall_precision_f1
from visualize import (
    plot_confusion_matrix,
    plot_pca_scatter_with_regions,
    plot_tree,
)
from wifi_utils import WiFiDataset, load_wifi_dataset, k_fold_indices


def nested_cv_evaluate_pruned(
    X: np.ndarray,
    y: np.ndarray,
    outer_k: int = 10,
    inner_k: int = 9,
    seed: int = 42
) -> Tuple[List[float], List[np.ndarray], List[List[int]], List[int], List[int]]:
    """Perform nested cross-validation for pruned trees."""
    n_samples = len(y)
    outer_splits = k_fold_indices(n_samples, k=outer_k, seed=seed)
    
    all_test_accuracies = []
    all_confusion_matrices = []
    all_classes = []
    all_depths_before = []
    all_depths_after = []
    
    for outer_fold_idx, (outer_train_val_idx, test_idx) in enumerate(outer_splits):
        print(f"Outer fold {outer_fold_idx + 1}/{outer_k}")
        
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        inner_splits = k_fold_indices(len(outer_train_val_idx), k=inner_k, seed=seed + outer_fold_idx)
        
        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
            # Map back to original indices
            actual_train_idx = outer_train_val_idx[inner_train_idx]
            actual_val_idx = outer_train_val_idx[inner_val_idx]
            
            X_train = X[actual_train_idx]
            y_train = y[actual_train_idx]
            X_val = X[actual_val_idx]
            y_val = y[actual_val_idx]
            
            # Train and prune
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            depth_before = model.get_depth()
            model.prune(X_val, y_val)
            depth_after = model.get_depth()
            
            all_depths_before.append(depth_before)
            all_depths_after.append(depth_after)
            
            # Evaluate
            y_pred = model.predict(X_test)
            test_acc = accuracy(y_test, y_pred)
            cm, classes = confusion_matrix(y_test, y_pred)
            
            all_test_accuracies.append(test_acc)
            all_confusion_matrices.append(cm)
            all_classes.append(classes)
    
    return all_test_accuracies, all_confusion_matrices, all_classes, all_depths_before, all_depths_after


def aggregate_confusion_matrices(all_cms: List[np.ndarray], all_classes: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
    """Aggregate all confusion matrices."""
    all_class_values = set()
    for classes in all_classes:
        all_class_values.update(classes)
    unique_classes = sorted(list(all_class_values))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    n_classes = len(unique_classes)
    
    aggregated_cm = np.zeros((n_classes, n_classes), dtype=int)
    for cm, classes in zip(all_cms, all_classes):
        local_to_global = {local_cls: class_to_idx[local_cls] for local_cls in classes}
        for local_true_idx, true_cls in enumerate(classes):
            for local_pred_idx, pred_cls in enumerate(classes):
                global_true_idx = local_to_global[true_cls]
                global_pred_idx = local_to_global[pred_cls]
                aggregated_cm[global_true_idx, global_pred_idx] += cm[local_true_idx, local_pred_idx]
    
    return aggregated_cm, unique_classes


def main() -> None:
    parser = argparse.ArgumentParser(description="WiFi Decision Tree Coursework")
    parser.add_argument("--clean", type=str, default="wifi_db/clean_dataset.txt")
    parser.add_argument("--noisy", type=str, default="wifi_db/noisy_dataset.txt")
    parser.add_argument("--dataset", choices=["clean", "noisy"], default="clean", help="Which dataset to use for this run")
    parser.add_argument("--cv", action="store_true", help="Enable k-fold cross-validation instead of training on full dataset")
    parser.add_argument("--prune-cv", action="store_true", help="Run nested CV for pruned trees")
    parser.add_argument("--k", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.prune_cv:
        # Nested CV for pruned trees - evaluate BOTH datasets
        datasets = {
            "clean": load_wifi_dataset(args.clean),
            "noisy": load_wifi_dataset(args.noisy)
        }
        
        for dataset_name, dataset_data in datasets.items():
            print(f"\n{'='*70}")
            print(f"Running nested CV for pruned trees on {dataset_name} dataset...")
            print(f"{'='*70}")
            
            all_accs, all_cms, all_classes, depths_before, depths_after = nested_cv_evaluate_pruned(
                dataset_data.features, dataset_data.labels, outer_k=10, inner_k=9, seed=42
            )
            
            print(f"\nResults for {dataset_name} dataset:")
            print(f"Total test results: {len(all_accs)}")
            print(f"Mean accuracy: {np.mean(all_accs):.4f}")
            print(f"Std accuracy: {np.std(all_accs):.4f}")
            print(f"Min: {np.min(all_accs):.4f}, Max: {np.max(all_accs):.4f}")
            
            # Aggregate CM
            agg_cm, classes = aggregate_confusion_matrices(all_cms, all_classes)
            print(f"\nAggregated confusion matrix:")
            print(agg_cm)
            
            # Calculate recall, precision, F1
            recall_dict, precision_dict, f1_dict = recall_precision_f1(agg_cm, classes)
            
            print(f"\nPer-class metrics:")
            for cls in sorted(classes):
                print(f"  Class {cls}: Recall={recall_dict[cls]:.4f}, Precision={precision_dict[cls]:.4f}, F1={f1_dict[cls]:.4f}")
            
            # Save CM for both datasets
            plot_confusion_matrix(agg_cm, classes, outdir / f"cm_{dataset_name}_pruned_counts.png", normalize=False)
            plot_confusion_matrix(agg_cm, classes, outdir / f"cm_{dataset_name}_pruned_normalized.png", normalize=True)
            
            # Depth analysis
            print(f"\nDepth Analysis for {dataset_name}:")
            print(f"  Before pruning: mean={np.mean(depths_before):.2f}, min={np.min(depths_before)}, max={np.max(depths_before)}")
            print(f"  After pruning: mean={np.mean(depths_after):.2f}, min={np.min(depths_after)}, max={np.max(depths_after)}")
            print(f"  Avg reduction: {np.mean(depths_before) - np.mean(depths_after):.2f}")
        
        # Visualize pruned tree for clean dataset only
        print(f"\n{'='*70}")
        print("Visualizing pruned tree for clean dataset...")
        clean_data = datasets["clean"]
        indices = np.random.default_rng(42).permutation(len(clean_data.labels))
        split_idx = int(0.8 * len(clean_data.labels))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        
        model = DecisionTreeClassifier()
        model.fit(clean_data.features[train_idx], clean_data.labels[train_idx])
        model.prune(clean_data.features[val_idx], clean_data.labels[val_idx])
        
        plot_tree(model.root, outdir / "tree_clean_pruned.png")
        print("Saved: tree_clean_pruned.png")
        
        return
    
    # Select a single dataset for this run (if not prune_cv)
    selected_name = "clean" if args.dataset == "clean" else "noisy"
    selected_path = args.clean if selected_name == "clean" else args.noisy
    data: WiFiDataset = load_wifi_dataset(selected_path)

    if args.cv:
        # k-fold cross-validation on the selected dataset
        scores = cross_val_scores(lambda: DecisionTreeClassifier(), data.features, data.labels, k=args.k)
        print(f"Dataset: {selected_name}")
        print(f"{args.k}-fold CV accuracies: {[round(float(s), 4) for s in scores]}")
        print(f"Mean accuracy: {float(np.mean(scores)):.4f}  (std: {float(np.std(scores)):.4f})")
        return

    # Train on the full selected dataset and produce visualizations
    model = DecisionTreeClassifier()
    model.fit(data.features, data.labels)
    
    print(f"\nDepth before pruning: {model.get_depth()}")

    # Split data for pruning demo (80% train, 20% validation)
    n = len(data.labels)
    split_idx = int(0.8 * n)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train = data.features[train_idx]
    y_train = data.labels[train_idx]
    X_val = data.features[val_idx]
    y_val = data.labels[val_idx]
    
    # Train and prune
    pruned_model = DecisionTreeClassifier()
    pruned_model.fit(X_train, y_train)
    depth_before = pruned_model.get_depth()
    print(f"Depth after fit on train: {depth_before}")
    
    pruned_model.prune(X_val, y_val)
    depth_after = pruned_model.get_depth()
    print(f"Depth after pruning: {depth_after}")
    print(f"Depth reduction: {depth_before - depth_after}")
    
    # Evaluate
    y_pred_train = pruned_model.predict(X_train)
    y_pred_val = pruned_model.predict(X_val)
    
    acc_train = accuracy(y_train, y_pred_train)
    acc_val = accuracy(y_val, y_pred_val)
    
    print(f"\nAccuracy on training set: {acc_train:.4f}")
    print(f"Accuracy on validation set: {acc_val:.4f}")
    
    # Evaluate on full dataset
    y_pred = pruned_model.predict(data.features)
    acc = accuracy(data.labels, y_pred)
    cm, classes = confusion_matrix(data.labels, y_pred)

    print(f"\nDataset: {selected_name}")
    print(f"Accuracy on full dataset: {acc:.4f}")
    print("Classes:", classes)
    print("Confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(linewidth=120)
    print(cm)

    # Visualizations
    plot_confusion_matrix(cm, classes, outdir / f"cm_{selected_name}_pruned_counts.png", normalize=False)
    plot_confusion_matrix(cm, classes, outdir / f"cm_{selected_name}_pruned_normalized.png", normalize=True)

    # PCA scatter with decision regions
    plot_pca_scatter_with_regions(
        data.features,
        data.labels,
        predict_fn=pruned_model.predict,
        out_path=outdir / f"pca_regions_{selected_name}_pruned.png",
        h=0.8,
    )

    # Plot the pruned tree
    if getattr(pruned_model, "root", None) is not None:
        plot_tree(pruned_model.root, outdir / f"tree_{selected_name}_pruned.png")


if __name__ == "__main__":
    main()


