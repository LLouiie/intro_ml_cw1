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


def main() -> None:
    parser = argparse.ArgumentParser(description="WiFi Decision Tree Coursework")
    parser.add_argument("--clean", type=str, default="wifi_db/clean_dataset.txt")
    parser.add_argument("--noisy", type=str, default="wifi_db/noisy_dataset.txt")
    parser.add_argument("--dataset", choices=["clean", "noisy"], default="clean", help="Which dataset to use for this run")
    parser.add_argument("--cv", action="store_true", help="Enable k-fold cross-validation instead of training on full dataset")
    parser.add_argument("--k", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    # Select a single dataset for this run
    selected_name = "clean" if args.dataset == "clean" else "noisy"
    selected_path = args.clean if selected_name == "clean" else args.noisy
    data: WiFiDataset = load_wifi_dataset(selected_path)

    outdir = Path(args.outdir)

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

    y_pred = model.predict(data.features)
    acc = accuracy(data.labels, y_pred)
    cm, classes = confusion_matrix(data.labels, y_pred)

    print(f"Dataset: {selected_name}")
    print(f"Accuracy (train on full dataset): {acc:.4f}")
    print("Classes:", classes)
    print("Confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(linewidth=120)
    print(cm)

    # Visualizations for the selected dataset
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


if __name__ == "__main__":
    main()


