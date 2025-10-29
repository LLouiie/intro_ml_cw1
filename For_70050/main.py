import argparse
from pathlib import Path
import numpy as np

from decision_tree import DecisionTreeClassifier
from metrics import (
    accuracy,
    confusion_matrix,
    cross_val_evaluate,
)
from visualize import (
    plot_confusion_matrix,
    plot_pca_scatter_with_regions,
    plot_tree,
)
from wifi_utils import WiFiDataset, load_wifi_dataset
from prune_evaluation import run_prune_evaluation


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

    # Select a single dataset for this run
    selected_name = "clean" if args.dataset == "clean" else "noisy"
    selected_path = args.clean if selected_name == "clean" else args.noisy
    data: WiFiDataset = load_wifi_dataset(selected_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ADDED: Handle --prune-cv mode for nested CV evaluation
    if args.prune_cv:
        clean_dataset = load_wifi_dataset(args.clean)
        noisy_dataset = load_wifi_dataset(args.noisy)
        run_prune_evaluation(clean_dataset, noisy_dataset, outdir)
        return
    
    # ORIGINAL CODE - UNCHANGED BELOW
    if args.cv:
        results = cross_val_evaluate(
            model_factory=lambda: DecisionTreeClassifier(),
            X=data.features,
            y=data.labels,
            k=args.k,
            seed=42,
        )

        # Use all dataset labels to keep class order consistent across plots
        classes = list(map(int, np.unique(data.labels).astype(int)))

        mean_acc = results["mean_accuracy"]
        mean_cm = results["mean_confusion_matrix"]    # Combined confusion matrix (sum of all folds)
        per_prec = results["mean_precision"]          # Mean per-class precision across folds
        per_rec = results["mean_recall"]              # Mean per-class recall across folds
        per_f1 = results["mean_f1"]                   # Mean per-class F1 across folds

        print(f"Dataset: {selected_name}")
        print(f"{args.k}-fold Mean Accuracy: {mean_acc:.4f}")
        np.set_printoptions(linewidth=120, suppress=True)
        print("Accumulated Confusion Matrix over all test folds (rows=true, cols=pred):")
        print(mean_cm)
        print("Per-class Precision:", np.round(per_prec, 4))
        print("Per-class Recall   :", np.round(per_rec, 4))
        print("Per-class F1       :", np.round(per_f1, 4))

        # Plot the overall confusion matrix from cross-validation (counts & normalized)
        plot_confusion_matrix(mean_cm, classes, outdir / f"cv_cm_{selected_name}_counts.png", normalize=False)
        plot_confusion_matrix(mean_cm, classes, outdir / f"cv_cm_{selected_name}_normalized.png", normalize=True)

        # Also train a representative model on full data to save a tree image for CV runs
        cv_model = DecisionTreeClassifier()
        cv_model.fit(data.features, data.labels)
        if getattr(cv_model, "root", None) is not None:
            plot_tree(cv_model.root, outdir / f"tree_{selected_name}_cv.png")
        return

    # Train on the full selected dataset and produce visualizations (ORIGINAL OUTPUT)
    model = DecisionTreeClassifier()
    model.fit(data.features, data.labels)

    y_pred = model.predict(data.features)
    acc = accuracy(data.labels, y_pred)
    cm, classes = confusion_matrix(data.labels, y_pred)

    print(f"Dataset: {selected_name}")
    print(f"Accuracy (train on full dataset): {acc:.4f}")
    print("Classes:", classes)
    print("Confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(linewidth=120)
    print(cm)

    # Visualizations for the selected dataset (ORIGINAL FILES - YOUR TEAMMATE'S OUTPUT)
    plot_confusion_matrix(cm, classes, outdir / f"cm_{selected_name}_counts.png", normalize=False)
    plot_confusion_matrix(cm, classes, outdir / f"cm_{selected_name}_normalized.png", normalize=True)

    # PCA scatter with decision regions (approximate back-projection)
    plot_pca_scatter_with_regions(
        data.features,
        data.labels,
        predict_fn=model.predict,
        out_path=outdir / f"pca_regions_{selected_name}.png",
        h=0.8,
    )

    # Plot the trained decision tree (for the full-data model)
    if getattr(model, "root", None) is not None:
        # Keep filename consistent with coursework figure naming
        plot_tree(model.root, outdir / "tree.png")
    
    # ADDED: Also generate pruned version - all visualizations
    print(f"\n--- Demonstrating pruning on {selected_name} dataset ---")
    n = len(data.labels)
    split_idx = int(0.8 * n)
    indices = np.random.default_rng(42).permutation(n)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    pruned_model = DecisionTreeClassifier()
    pruned_model.fit(data.features[train_idx], data.labels[train_idx])
    pruned_model.prune(data.features[val_idx], data.labels[val_idx])
    
    # Evaluate pruned model on full dataset
    y_pred_pruned = pruned_model.predict(data.features)
    acc_pruned = accuracy(data.labels, y_pred_pruned)
    cm_pruned, classes_pruned = confusion_matrix(data.labels, y_pred_pruned)
    
    print(f"Pruned accuracy on full dataset: {acc_pruned:.4f}")
    
    # Generate all pruned visualizations
    plot_confusion_matrix(cm_pruned, classes_pruned, outdir / f"cm_{selected_name}_pruned_counts.png", normalize=False)
    plot_confusion_matrix(cm_pruned, classes_pruned, outdir / f"cm_{selected_name}_pruned_normalized.png", normalize=True)
    
    plot_pca_scatter_with_regions(
        data.features,
        data.labels,
        predict_fn=pruned_model.predict,
        out_path=outdir / f"pca_regions_{selected_name}_pruned.png",
        h=0.8,
    )
    
    plot_tree(pruned_model.root, outdir / f"tree_{selected_name}_pruned.png")
    print(f"Saved all pruned visualizations for {selected_name} dataset")


if __name__ == "__main__":
    main()
