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
    plot_tree,
)
from wifi_utils import WiFiDataset, load_wifi_dataset
from prune_evaluation import run_prune_evaluation


def main() -> None:
    """
    Entry point for the WiFi Decision Tree Coursework.

    This script supports three main modes:
    1. Train and evaluate a decision tree on a single dataset (clean or noisy).
    2. Run k-fold cross-validation to assess model performance.
    3. Run nested cross-validation for pruning evaluation.

    Command-line arguments allow switching datasets, enabling CV, and controlling output paths.
    """
    parser = argparse.ArgumentParser(description="WiFi Decision Tree Coursework")
   
    # Dataset paths
    parser.add_argument("--clean", type=str, default="wifi_db/clean_dataset.txt")
    parser.add_argument("--noisy", type=str, default="wifi_db/noisy_dataset.txt")
   
    # Dataset selection
    parser.add_argument("--dataset", choices=["clean", "noisy"], default="clean", help="Which dataset to use for this run")
    
    # Evaluation modes
    parser.add_argument("--cv", action="store_true", help="Enable k-fold cross-validation instead of training on full dataset")
    parser.add_argument("--prune-cv", action="store_true", help="Run nested CV for pruned trees")
    
    # Cross-validation parameters
    parser.add_argument("--k", type=int, default=10, help="Number of folds for cross-validation")
    
    # Output directory for figures
    parser.add_argument("--outdir", type=str, default="figures")

    args = parser.parse_args()

    # Enforce: pruning only via CV. Treat --prune as alias to --prune-cv.
    if getattr(args, "prune", False):
        args.prune_cv = True

    # Select a single dataset for this run
    selected_name = "clean" if args.dataset == "clean" else "noisy"
    selected_path = args.clean if selected_name == "clean" else args.noisy
    data: WiFiDataset = load_wifi_dataset(selected_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Nested cross-validation for pruning evaluation
    if args.prune_cv:
        clean_dataset = load_wifi_dataset(args.clean)
        noisy_dataset = load_wifi_dataset(args.noisy)
        run_prune_evaluation(clean_dataset, noisy_dataset, outdir)
        return
    
    # Standard k-fold cross-validation
    if args.cv:
        results = cross_val_evaluate(
            model_factory=lambda: DecisionTreeClassifier(),
            X=data.features,
            y=data.labels,
            k=args.k,
            seed=42,
        )

        # Ensure class order consistency across plots
        classes = list(map(int, np.unique(data.labels).astype(int)))

        mean_acc = results["mean_accuracy"]
        mean_cm = results["mean_confusion_matrix"]    # Mean confusion matrix across folds
        per_prec = results["mean_precision"]          # Mean per-class precision across folds
        per_rec = results["mean_recall"]              # Mean per-class recall across folds
        per_f1 = results["mean_f1"]                   # Mean per-class F1 across folds


        # Print summary statistics
        print(f"Dataset: {selected_name}")
        print(f"{args.k}-fold Mean Accuracy: {mean_acc:.4f}")
        np.set_printoptions(linewidth=120, suppress=True)
        print("Mean Confusion Matrix across folds (rows=true, cols=pred):")
        print(mean_cm)
        print("Per-class Precision:", np.round(per_prec, 4))
        print("Per-class Recall   :", np.round(per_rec, 4))
        print("Per-class F1       :", np.round(per_f1, 4))

        # Plot the overall confusion matrix from cross-validation (counts & normalized)
        plot_confusion_matrix(mean_cm, classes, outdir / f"cv_cm_{selected_name}.png", normalize=False)
        plot_confusion_matrix(mean_cm, classes, outdir / f"cv_cm_{selected_name}_normalized.png", normalize=True)


        # Train a representative model on full data for visualization
        cv_model = DecisionTreeClassifier()
        cv_model.fit(data.features, data.labels)
        if getattr(cv_model, "root", None) is not None:
            plot_tree(cv_model.root, outdir / f"tree_{selected_name}_cv.png")
        return

    # Train on full dataset with no CV
    model = DecisionTreeClassifier()
    model.fit(data.features, data.labels)

    y_pred = model.predict(data.features)
    acc = accuracy(data.labels, y_pred)
    cm, classes = confusion_matrix(data.labels, y_pred)

    # Print metrics
    print(f"Dataset: {selected_name}")
    print(f"Accuracy (train on full dataset): {acc:.4f}")
    print("Classes:", classes)
    print("Confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(linewidth=120)
    print(cm)

    # Visualizations for the selected dataset (ORIGINAL FILES - YOUR TEAMMATE'S OUTPUT)
    plot_confusion_matrix(cm, classes, outdir / f"cm_{selected_name}_counts.png", normalize=False)
    plot_confusion_matrix(cm, classes, outdir / f"cm_{selected_name}_normalized.png", normalize=True)

    # Plot the trained decision tree (for the full-data model)
    if getattr(model, "root", None) is not None:
        # Keep filename consistent with coursework figure naming
        plot_tree(model.root, outdir / "tree.png")
 

if __name__ == "__main__":
    main()
