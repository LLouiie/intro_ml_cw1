from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, classes: list[int], out_path: Path, normalize: bool = False) -> None:
    if normalize:
        with np.errstate(all="ignore"):
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        data = cm_norm
        fmt = ".2f"
    else:
        data = cm
        fmt = "d"

    plt.figure(figsize=(5.2, 4.5))
    im = plt.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = (data.max() + data.min()) / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > thresh else "black"
            plt.text(j, i, format(data[i, j], fmt), ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def pca_project(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    # Use SVD for stability
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    W = VT[:n_components].T  # (d, n_components)
    return Xc @ W


def plot_pca_scatter_with_regions(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn,
    out_path: Path,
    h: float = 0.5,
) -> None:
    """
    - Reduce X to 2D with PCA
    - Fit a simple linear mapping from 2D back to original space for coarse decision regions:
      we project grid points back by W^T (pseudo-inverse of PCA) around the mean.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    W = VT[:2].T
    X2 = Xc @ W

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # Map grid back to original space: approx inverse by W^T and add mean
    X_grid = grid_2d @ W.T + X.mean(axis=0, keepdims=True)
    Z = predict_fn(X_grid).reshape(xx.shape)

    colors = ListedColormap(["#FFBBBB", "#BBFFBB", "#BBBBFF", "#FFF3AA", "#E0BBFF", "#BBFFFF"])  # up to 6 classes
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.6)

    # Scatter actual points
    unique_labels = np.unique(y)
    for lbl in unique_labels:
        mask = y == lbl
        plt.scatter(X2[mask, 0], X2[mask, 1], s=18, label=str(int(lbl)))

    plt.legend(title="Class", fontsize=9)
    plt.title("PCA scatter with decision regions (approx)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_split_importance(split_counts: np.ndarray, out_path: Path) -> None:
    indices = np.arange(len(split_counts))
    plt.figure(figsize=(7, 3.5))
    plt.bar(indices, split_counts, color="#4C78A8")
    plt.xlabel("Feature index")
    plt.ylabel("Split count")
    plt.title("Feature split importance (counts)")
    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _tree_collect_layout(node, depth=0, layout_list=None, xs=None):
    if layout_list is None:
        layout_list = []
    if xs is None:
        xs = {}
    xs.setdefault(depth, 0)
    x_pos = xs[depth]
    layout_list.append((node, depth, x_pos))
    xs[depth] += 1
    if getattr(node, "left", None) is not None:
        _tree_collect_layout(node.left, depth + 1, layout_list, xs)
    if getattr(node, "right", None) is not None:
        _tree_collect_layout(node.right, depth + 1, layout_list, xs)
    return layout_list


def plot_tree(root, out_path: Path) -> None:
    """
    Simple hierarchical plot of the decision tree using matplotlib.
    Each node shows: split (feature<=thr) or prediction, samples and entropy.
    """
    layout_list = _tree_collect_layout(root)
    depths = [d for (_, d, _) in layout_list]
    max_depth = max(depths) if depths else 0

    # Normalize x positions per depth to [0,1]
    per_depth_counts = {}
    for _, d, x in layout_list:
        per_depth_counts[d] = max(per_depth_counts.get(d, 0), x + 1)

    pos = {}
    node_by_id = {}
    for node, d, x in layout_list:
        count = per_depth_counts[d]
        x_norm = (x + 0.5) / count
        y_norm = 1.0 - (d / max(max_depth, 1)) if max_depth > 0 else 0.5
        nid = id(node)
        pos[nid] = (x_norm, y_norm)
        node_by_id[nid] = node

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.axis("off")

    # Draw edges first
    for node, d, _ in layout_list:
        x, y = pos[id(node)]
        for child in [getattr(node, "left", None), getattr(node, "right", None)]:
            if child is not None:
                cx, cy = pos[id(child)]
                ax.plot([x, cx], [y, cy], color="#888888", linewidth=1)

    # Draw nodes
    for node, d, _ in layout_list:
        x, y = pos[id(node)]
        is_leaf = getattr(node, "prediction", None) is not None and getattr(node, "left", None) is None and getattr(node, "right", None) is None
        box_color = "#F5F5F5" if is_leaf else "#E8F0FE"
        rect = plt.Rectangle((x - 0.06, y - 0.03), 0.12, 0.06, facecolor=box_color, edgecolor="#333333")
        ax.add_patch(rect)
        if getattr(node, "prediction", None) is not None and (getattr(node, "left", None) is None and getattr(node, "right", None) is None):
            text = f"Leaf\npred={int(node.prediction)}\nN={node.num_samples}\nH={node.impurity:.2f}"
        else:
            feat = getattr(node, "feature", None)
            thr = getattr(node, "threshold", None)
            text = f"x[{feat}]<= {thr:.2f}\nN={node.num_samples}\nH={node.impurity:.2f}"
        ax.text(x, y, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


