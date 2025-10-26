## WiFi Decision Tree Coursework (For_70050)

This project implements a small end-to-end pipeline to train and evaluate a Decision Tree classifier on WiFi RSSI datasets. It includes data loading utilities, cross-validation, model training, metrics, and multiple visualizations.

**Note**: This implementation strictly follows the coursework requirements - only using **numpy, matplotlib, and standard Python libraries** (no scipy, scikit-learn, or other external ML libraries).

### Quick Start

```bash
# (optional) create and activate a virtualenv
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r For_70050/requirements.txt

# Alternative: if you want to exclude scipy (code uses only numpy)
# pip install matplotlib==3.10.6 numpy==2.3.3

# run (outputs figures under For_70050/figures)
python For_70050/main.py \
  --clean For_70050/wifi_db/clean_dataset.txt \
  --noisy For_70050/wifi_db/noisy_dataset.txt \
  --k 5 --depth_min 1 --depth_max 15 \
  --outdir For_70050/figures

# or use the provided script
bash intro_ml_cw1/bash.sh
```

### Data Flow (Overview)
1. Load datasets (`wifi_utils.load_wifi_dataset`).
2. Cross-validate candidate `max_depth` values (`metrics.cross_val_scores`).
3. Train the final `DecisionTreeClassifier` on the clean dataset.
4. Evaluate on clean/noisy datasets and generate figures.

---

## Module Reference

### decision_tree.py

- `entropy(labels: np.ndarray) -> float`
  - Shannon entropy of integer class labels. Returns 0.0 for empty arrays.

- `best_threshold_for_feature(X_col: np.ndarray, y: np.ndarray) -> tuple[float, float]`
  - Finds the threshold for a single feature column that maximizes information gain.
  - Returns `(best_gain, threshold)`. Split rule is `left <= threshold`, `right > threshold`.

- `@dataclass Node`
  - Fields: `feature: int | None`, `threshold: float | None`, `left: Node | None`, `right: Node | None`, `prediction: int | None`, `num_samples: int | None`, `impurity: float | None`.

- `class DecisionTreeClassifier`
  - `__init__(max_depth: int = 5, min_samples_split: int = 2)`
    - Constraints: `max_depth >= 1`, `min_samples_split >= 2`.
    - Attributes after fit:
      - `root: Node | None` — trained tree root.
      - `split_counts_: np.ndarray | None` — feature split frequency (for visualization).
  - `fit(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier`
    - Builds a binary tree using greedy information gain splits; records per-feature split counts.
  - `predict(X: np.ndarray) -> np.ndarray`
    - Traverses the tree for each row; returns integer class predictions.
  - `_build_tree(X: np.ndarray, y: np.ndarray, depth: int) -> Tuple[Node, int]`
    - Internal recursive method that follows the PDF Algorithm 1 specification.
    - Returns `(node, actual_max_depth)` tuple as per coursework requirements.
    - Performs left-then-right recursion and computes `max(l_depth, r_depth)` for depth tracking.

Notes:
- Algorithm follows PDF Algorithm 1 (decision_tree_learning procedure).
- Stops when `depth >= max_depth`, `num_samples < min_samples_split`, node impurity is 0, or all samples have the same label.
- If no beneficial split found, makes a leaf predicting the most common label.

### metrics.py

- `accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float`
  - Mean of correct predictions. Requires matching shapes.

- `confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, list[int]]`
  - Returns `(cm, classes)` where `cm[i, j]` counts `true=classes[i]`, `pred=classes[j]`.

- `cross_val_scores(model_factory, X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> list[float]`
  - K-fold CV using `wifi_utils.k_fold_indices`. For each fold: `model = model_factory(); model.fit(train); score = accuracy(val)`.

### visualize.py

- `plot_confusion_matrix(cm: np.ndarray, classes: list[int], out_path: Path, normalize: bool = False) -> None`
  - Saves confusion matrix heatmap. If `normalize=True`, rows are normalized.

- `pca_project(X: np.ndarray, n_components: int = 2) -> np.ndarray`
  - PCA via SVD using `numpy.linalg.svd` (no scipy). Returns projected `X` in lower dimensions.

- `plot_pca_scatter_with_regions(X: np.ndarray, y: np.ndarray, predict_fn, out_path: Path, h: float = 0.5) -> None`
  - Reduces to 2D with PCA; approximates inverse mapping to plot decision regions and samples overlay. `predict_fn` is a callable like `model.predict`.

- `plot_feature_split_importance(split_counts: np.ndarray, out_path: Path) -> None`
  - Bar chart of per-feature split counts from the trained tree (`DecisionTreeClassifier.split_counts_`).

- `plot_tree(root, out_path: Path) -> None`
  - Simple hierarchical drawing of the trained tree using `Node` attributes (`feature`, `threshold`, `prediction`, `num_samples`, `impurity`).

### wifi_utils.py

- `@dataclass WiFiDataset`
  - `features: np.ndarray` with shape `(n_samples, n_features)`
  - `labels: np.ndarray` with shape `(n_samples,)`

- `load_wifi_dataset(path: str) -> WiFiDataset`
  - Loads tab- or space-delimited file, last column is integer label, others are float features.

- `train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int | None = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  - Shuffles indices with `seed`; returns `(X_train, X_test, y_train, y_test)`.

- `k_fold_indices(n_samples: int, k: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]`
  - Returns list of `(train_idx, val_idx)` splits. Requires `k >= 2`.

### main.py

- `tune_depth(X: np.ndarray, y: np.ndarray, depths: list[int], k: int = 5) -> int`
  - Runs K-fold CV over candidate `depths` using `DecisionTreeClassifier(max_depth=d)`; returns the depth with the highest mean accuracy.

- `plot_depth_curve(depths: list[int], scores: list[float], out_path: Path) -> None`
  - Saves the cross-validation accuracy curve for visual inspection.

- CLI `main()`
  - Arguments:
    - `--clean`: path to clean dataset (default `wifi_db/clean_dataset.txt`)
    - `--noisy`: path to noisy dataset (default `wifi_db/noisy_dataset.txt`)
    - `--k`: number of CV folds (default `5`)
    - `--depth_min` / `--depth_max`: depth search range (default `1..15`)
    - `--plot`: legacy option for depth plot output (not strictly required)
    - `--outdir`: figures output directory (default `figures`)
  - Prints: best depth, train/test accuracies, class list, confusion matrix.
  - Saves: `depth_cv.png`, `cm_noisy_counts.png`, `cm_noisy_normalized.png`, `pca_regions_clean.png`, `pca_regions_noisy.png`, `feature_importance.png`, `tree.png`.

---

## Tips & Notes
- Ensure labels are integers; features should be numeric (float-compatible).
- Reproducibility: CV splitting uses the provided `seed` for shuffling.
- Avoid training with empty arrays; functions validate shapes where necessary.
- `plot_pca_scatter_with_regions` is an approximation of decision regions via PCA back-projection; interpret qualitatively.
- **Coursework compliance**: Only uses `numpy`, `matplotlib`, and standard Python libraries (as required by the assignment).
- **Implementation**: Decision tree algorithm follows PDF Algorithm 1 exactly, including depth tracking and left-then-right recursion.
- **Note on scipy**: While `requirements.txt` lists scipy, the code has been modified to use `numpy.linalg.svd` instead. Scipy will be installed but not used by the code.

