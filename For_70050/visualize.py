from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> None:
    """
    Ensure that the parent directory of a given path exists.

    Args:
        path (Path): Path to a file or directory.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, classes: list[int], out_path: Path, normalize: bool = False) -> None:
    """
    Plot and save a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix (square 2D array).
        classes (list[int]): List of class labels.
        out_path (Path): Path where the plot will be saved.
        normalize (bool, optional): Whether to normalize each row (default: False).

    Returns:
        None
    """
    if normalize:
        # Normalize confusion matrix by row
        with np.errstate(all="ignore"):
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        data = cm_norm
        fmt = ".2f" # Format with 2 decimal places
    else:
        data = cm
        # Use integer format only if all values are integers
        if np.issubdtype(data.dtype, np.integer):
            fmt = "d"
        else:
            fmt = ".2f"

    plt.figure(figsize=(5.2, 4.5))
    im = plt.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # Tick marks for class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Determine threshold for text color switching (white vs black)
    thresh = (data.max() + data.min()) / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > thresh else "black"
            plt.text(j, i, format(data[i, j], fmt), ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    
    # Save figure to disk
    ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _tree_collect_layout(node, depth=0, layout_list=None, xs=None):
    """
    Recursively collect layout positions for each node in a binary tree.

    Args:
        node (object): Current tree node (with attributes `.left` and `.right`).
        depth (int, optional): Current depth in the tree.
        layout_list (list, optional): Accumulated list of (node, depth, x_position).
        xs (dict, optional): X-position counters per depth level.

    Returns:
        list: A list of tuples (node, depth, x_position).
    """
    if layout_list is None:
        layout_list = []
    if xs is None:
        xs = {}
    
    xs.setdefault(depth, 0)
    x_pos = xs[depth]
    layout_list.append((node, depth, x_pos))
    xs[depth] += 1
    
    # Recurse on children if they exist
    if getattr(node, "left", None) is not None:
        _tree_collect_layout(node.left, depth + 1, layout_list, xs)
    if getattr(node, "right", None) is not None:
        _tree_collect_layout(node.right, depth + 1, layout_list, xs)
    return layout_list


def plot_tree(root, out_path: Path) -> None:
    """
    Plot a simple visualization of a binary decision tree.

    Args:
        root (object): Root node of the decision tree.
                       Expected attributes: `feature`, `threshold`, `left`, `right`, and optionally `prediction`.
        out_path (Path): Output file path to save the plot.
    """
    layout_list = _tree_collect_layout(root)
    depths = [d for (_, d, _) in layout_list]
    max_depth = max(depths) if depths else 0

    # Normalize x positions per depth to [0,1]
    per_depth_counts = {}
    for _, d, x in layout_list:
        per_depth_counts[d] = max(per_depth_counts.get(d, 0), x + 1)

    # Compute normalized (x, y) positions for each node
    pos = {}
    node_by_id = {}
    for node, d, x in layout_list:
        count = per_depth_counts[d]
        x_norm = (x + 0.5) / count # Evenly spaced horizontally
        y_norm = 1.0 - (d / max(max_depth, 1)) if max_depth > 0 else 0.5
        nid = id(node)
        pos[nid] = (x_norm, y_norm)
        node_by_id[nid] = node

    # Figure size scales with horizontal node count and depth to reduce crowding
    max_nodes_at_any_depth = max(per_depth_counts.values()) if per_depth_counts else 1
    fig_width = max(14.0, min(60.0, max_nodes_at_any_depth * 1.2))
    fig_height = max(6.0, min(40.0, (max_depth + 1) * 1.2))
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.axis("off")

    # Draw edges first, color by depth for clarity
    cmap = plt.get_cmap("viridis")
    denom = max(max_depth, 1)
    for node, d, _ in layout_list:
        x, y = pos[id(node)]
        edge_color = cmap(d / denom)
        for child in [getattr(node, "left", None), getattr(node, "right", None)]:
            if child is not None:
                cx, cy = pos[id(child)]
                ax.plot([x, cx], [y, cy], color=edge_color, linewidth=1.2)

    # Draw nodes with rounded boxes
    for node, d, _ in layout_list:
        x, y = pos[id(node)]
        
        # Identify whether node is a leaf
        is_leaf = getattr(node, "prediction", None) is not None and getattr(node, "left", None) is None and getattr(node, "right", None) is None
        
        # Node label
        if is_leaf:
            text = f"leaf:{int(node.prediction)}"
        else:
            feat = getattr(node, "feature", None)
            thr = getattr(node, "threshold", None)
            text = f"[x{feat} < {thr:.1f}]"
        
        # Draw node box with centered text
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#4C78A8",
                "lw": 1.0,
            },
        )

    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


