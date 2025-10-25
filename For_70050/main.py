from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from decision_tree import DecisionTreeClassifier
from metrics import accuracy, confusion_matrix, cross_val_scores
from wifi_utils import WiFiDataset, load_wifi_dataset


def tune_depth(X: np.ndarray, y: np.ndarray, depths: List[int], k: int = 5) -> int:
    cv_means = []
    for d in depths:
        scores = cross_val_scores(lambda: DecisionTreeClassifier(max_depth=d), X, y, k=k)
        cv_means.append(np.mean(scores))
    best_idx = int(np.argmax(cv_means))
    return int(depths[best_idx])


def plot_depth_curve(depths: List[int], scores: List[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(depths, scores, marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("CV accuracy")
    plt.title("Decision Tree Depth Tuning")
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="WiFi Decision Tree Coursework")
    parser.add_argument("--clean", type=str, default="wifi_db/clean_dataset.txt")
    parser.add_argument("--noisy", type=str, default="wifi_db/noisy_dataset.txt")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--depth_min", type=int, default=1)
    parser.add_argument("--depth_max", type=int, default=15)
    parser.add_argument("--plot", type=str, default="figures/depth_cv.png")
    args = parser.parse_args()

    clean: WiFiDataset = load_wifi_dataset(args.clean)
    noisy: WiFiDataset = load_wifi_dataset(args.noisy)

    depths = list(range(args.depth_min, args.depth_max + 1))
    # Compute CV curve for plotting
    cv_curve = [
        np.mean(cross_val_scores(lambda d=d: DecisionTreeClassifier(max_depth=d), clean.features, clean.labels, k=args.k))
        for d in depths
    ]
    plot_depth_curve(depths, cv_curve, Path(args.plot))

    # Pick best depth and train-final on clean
    best_depth = int(depths[int(np.argmax(cv_curve))])
    model = DecisionTreeClassifier(max_depth=best_depth)
    model.fit(clean.features, clean.labels)

    # Evaluate on clean (train perf) and noisy (generalization perf)
    y_pred_clean = model.predict(clean.features)
    y_pred_noisy = model.predict(noisy.features)
    acc_clean = accuracy(clean.labels, y_pred_clean)
    acc_noisy = accuracy(noisy.labels, y_pred_noisy)
    cm_noisy, classes = confusion_matrix(noisy.labels, y_pred_noisy)

    print(f"Best depth: {best_depth}")
    print(f"Accuracy (clean/train): {acc_clean:.4f}")
    print(f"Accuracy (noisy/test): {acc_noisy:.4f}")
    print("Classes:", classes)
    print("Confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(linewidth=120)
    print(cm_noisy)


if __name__ == "__main__":
    main()


