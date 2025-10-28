import argparse
from pathlib import Path
import numpy as np

from decision_tree import DecisionTreeClassifier
from metrics import accuracy, confusion_matrix, cross_val_scores
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

    # ADDED: Handle --prune-cv mode for nested CV evaluation
    if args.prune_cv:
        clean_dataset = load_wifi_dataset(args.clean)
        noisy_dataset = load_wifi_dataset(args.noisy)
        run_prune_evaluation(clean_dataset, noisy_dataset, outdir)
        return
    
    # ORIGINAL CODE - UNCHANGED BELOW
    if args.cv:
        # k-fold cross-validation on the selected dataset
        scores = cross_val_scores(lambda: DecisionTreeClassifier(), data.features, data.labels, k=args.k)
        print(f"Dataset: {selected_name}")
        print(f"{args.k}-fold CV accuracies: {[round(float(s), 4) for s in scores]}")
        print(f"Mean accuracy: {float(np.mean(scores)):.4f}  (std: {float(np.std(scores)):.4f})")
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
