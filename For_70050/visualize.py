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
    # --- Helpers for balanced layout -------------------------------------------------
    def max_depth_of(node) -> int:
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        if left is None and right is None:
            return 0
        left_depth = max_depth_of(left) + 1 if left is not None else 0
        right_depth = max_depth_of(right) + 1 if right is not None else 0
        return max(left_depth, right_depth)

    def count_leaves(node, cache: dict[int, int]) -> int:
        nid = id(node)
        if nid in cache:
            return cache[nid]
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        if left is None and right is None:
            cache[nid] = 1
            return 1
        leaves_left = count_leaves(left, cache) if left is not None else 0
        leaves_right = count_leaves(right, cache) if right is not None else 0
        cache[nid] = leaves_left + leaves_right
        return cache[nid]

    def assign_positions(node, depth: int, x_left: float, leaves_cache: dict[int, int], pos_xy: dict[int, tuple[float, float]]):
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        if left is None and right is None:
            pos_xy[id(node)] = (x_left + 0.5, depth)
            return
        leaves_left = leaves_cache.get(id(left), 0) if left is not None else 0
        if left is not None:
            assign_positions(left, depth + 1, x_left, leaves_cache, pos_xy)
        if right is not None:
            assign_positions(right, depth + 1, x_left + leaves_left, leaves_cache, pos_xy)
        if left is not None and right is not None:
            x_parent = (pos_xy[id(left)][0] + pos_xy[id(right)][0]) / 2.0
        elif left is not None:
            x_parent = pos_xy[id(left)][0]
        else:
            x_parent = pos_xy[id(right)][0]
        pos_xy[id(node)] = (x_parent, depth)

    def preorder_collect(node, depth: int, out: list[tuple[object, int]]):
        out.append((node, depth))
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        if left is not None:
            preorder_collect(left, depth + 1, out)
        if right is not None:
            preorder_collect(right, depth + 1, out)
        return out

    # Compute depth and balanced positions
    max_depth = max_depth_of(root)
    leaves_cache: dict[int, int] = {}
    total_leaves = count_leaves(root, leaves_cache)
    raw_pos: dict[int, tuple[float, float]] = {}
    assign_positions(root, 0, 0.0, leaves_cache, raw_pos)

    # Normalize to [0,1] coordinates
    pos: dict[int, tuple[float, float]] = {}
    for nid, (x_raw, depth) in raw_pos.items():
        x_norm = x_raw / max(total_leaves, 1)
        y_norm = 1.0 - (depth / max(max_depth, 1)) if max_depth > 0 else 0.5
        pos[nid] = (x_norm, y_norm)

    layout_list = preorder_collect(root, 0, [])

    # Figure sizing: make more compact horizontally
    fig_width = max(9.0, min(28.0, total_leaves * 0.50))
    fig_height = max(6.0, min(50.0, (max_depth + 1) * 1.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_xlim(-0.03, 0.995)  # tighter margins while preserving space for depth labels
    ax.set_ylim(-0.005, 1.005)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)

    # Draw edges, color by depth
    cmap = plt.get_cmap("viridis")
    denom = max(max_depth, 1)
    for node, d in layout_list:
        x, y = pos[id(node)]
        edge_color = cmap(d / denom)
        for child in [getattr(node, "left", None), getattr(node, "right", None)]:
            if child is not None:
                cx, cy = pos[id(child)]
                ax.plot([x, cx], [y, cy], color=edge_color, linewidth=1.3)

    # Depth labels on the left side
    for d in range(max_depth + 1):
        y = 1.0 - (d / max(max_depth, 1)) if max_depth > 0 else 0.5
        ax.text(-0.03, y, f"depth {d}", ha="right", va="center", fontsize=14, color="#444444")

    # Draw nodes with rounded boxes
    for node, _ in layout_list:
        x, y = pos[id(node)]
        is_leaf = getattr(node, "prediction", None) is not None and getattr(node, "left", None) is None and getattr(node, "right", None) is None
        if is_leaf:
            text = f"leaf:{int(node.prediction)}"
        else:
            feat = getattr(node, "feature", None)
            thr = getattr(node, "threshold", None)
            text = f"[x{feat} < {thr:.1f}]"
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "#4C78A8",
                "lw": 1.0,
            },
        )

    ensure_dir(out_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close()


