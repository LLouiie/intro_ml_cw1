## WiFi Decision Tree Coursework (For_70050)

This project implements a small end-to-end pipeline to train and evaluate a Decision Tree classifier on WiFi RSSI datasets. It includes data loading utilities, cross-validation, model training, metrics, and multiple visualizations.

**Note**: This implementation follows the coursework requirements, using only **numpy**, **matplotlib**, and standard Python libraries (no scikit-learn).

### Quick Start

```bash
# (optional) create and activate a virtualenv
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r For_70050/requirements.txt

# Output figures will be saved under For_70050/figures

# 1) Train on the full clean dataset (saves tree.png and other figures)
python For_70050/main.py \
  --clean For_70050/wifi_db/clean_dataset.txt \
  --noisy For_70050/wifi_db/noisy_dataset.txt \
  --dataset clean \
  --outdir For_70050/figures

# 2) 10-fold cross-validation on the clean dataset
python For_70050/main.py \
  --clean For_70050/wifi_db/clean_dataset.txt \
  --noisy For_70050/wifi_db/noisy_dataset.txt \
  --dataset clean \
  --cv --k 10 \
  --outdir For_70050/figures

# 3) 10-fold cross-validation on the noisy dataset
python For_70050/main.py \
  --clean For_70050/wifi_db/clean_dataset.txt \
  --noisy For_70050/wifi_db/noisy_dataset.txt \
  --dataset noisy \
  --cv --k 10 \
  --outdir For_70050/figures

# Or run the helper script (edit dataset paths inside if needed)
bash bash.sh
```

### Data Flow (Overview)
1. Select a single dataset per run (`--dataset clean|noisy`) and load it (`wifi_utils.load_wifi_dataset`).
2. If `--cv` is set, perform K-fold cross-validation (`metrics.cross_val_evaluate`) with `--k` folds and report aggregated metrics/plots.
3. If not `--cv`, train on the full selected dataset, report training accuracy, and save figures: confusion matrix, PCA decision regions, and the tree visualization.

---

## Module Reference

### decision_tree.py

- `entropy(labels: np.ndarray) -> float`
  - Shannon entropy of integer class labels. Returns 0.0 for empty arrays.

- `best_threshold_for_feature(X_col: np.ndarray, y: np.ndarray) -> tuple[float, float]`
  - Finds the threshold for a single feature column that maximizes information gain.
  - Returns `(best_gain, threshold)`. Split rule is `left <= threshold`, `right > threshold`.

- `@dataclass Node`
  - Fields: `feature: int | None`, `threshold: float | None`, `left: Node | None`, `right: Node | None`, `prediction: int | None`.

- `class DecisionTreeClassifier`
  - `__init__()` → sets `root: Node | None = None`.
  - `fit(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier` → builds a binary tree using greedy information gain splits.
  - `predict(X: np.ndarray) -> np.ndarray` → traverses the tree for each row; returns integer class predictions.
  - Internal: `_build_tree(X: np.ndarray, y: np.ndarray, depth: int) -> tuple[Node, int]` returns `(node, actual_max_depth)`; follows the PDF algorithm (left-then-right recursion).

### metrics.py

- `accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float`
  - Mean of correct predictions. Requires matching shapes.

- `confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, list[int]]`
  - Returns `(cm, classes)` where `cm[i, j]` counts `true=classes[i]`, `pred=classes[j]`.

- `cross_val_evaluate(model_factory, X: np.ndarray, y: np.ndarray, k: int = 10, seed: int = 42) -> dict`
  - K-fold CV with aggregated metrics. Returns a dictionary:
    - `mean_confusion_matrix`, `mean_accuracy`, `mean_precision`, `mean_recall`, `mean_f1`.

### visualize.py

- `plot_confusion_matrix(cm: np.ndarray, classes: list[int], out_path: Path, normalize: bool = False) -> None`
  - Saves confusion matrix heatmap. If `normalize=True`, rows are normalized.

- `pca_project(X: np.ndarray, n_components: int = 2) -> np.ndarray`
  - PCA via SVD using `numpy.linalg.svd` (no scipy). Returns projected `X` in lower dimensions.

- `plot_pca_scatter_with_regions(X: np.ndarray, y: np.ndarray, predict_fn, out_path: Path, h: float = 0.5) -> None`
  - Reduces to 2D with PCA; approximates inverse mapping to plot decision regions and samples overlay. `predict_fn` is a callable like `model.predict`.

- `plot_tree(root, out_path: Path) -> None`
  - Simple hierarchical drawing of the trained tree using `Node` attributes (`feature`, `threshold`, `prediction`). Split rule is `<= threshold`.

### wifi_utils.py

- `@dataclass WiFiDataset`
  - `features: np.ndarray` with shape `(n_samples, n_features)`
  - `labels: np.ndarray` with shape `(n_samples,)`

- `load_wifi_dataset(path: str) -> WiFiDataset`
  - Loads tab- or space-delimited file, last column is integer label, others are float features.

- `k_fold_indices(n_samples: int, k: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]`
  - Returns list of `(train_idx, val_idx)` splits. Requires `k >= 2`.

### main.py

- CLI arguments:
  - `--clean`: path to clean dataset (default `wifi_db/clean_dataset.txt`)
  - `--noisy`: path to noisy dataset (default `wifi_db/noisy_dataset.txt`)
  - `--dataset {clean,noisy}`: which dataset to use for this run (default `clean`)
  - `--cv`: enable K-fold cross-validation (aggregated metrics and plots)
  - `--k`: number of folds for cross-validation (default `10`)
  - `--prune-cv`: run pruning evaluation and save pruned visualizations
  - `--outdir`: figures output directory (default `figures`)

- Outputs when training on full dataset:
  - Confusion matrices: `cm_{clean|noisy}_counts.png`, `cm_{clean|noisy}_normalized.png`
  - PCA decision regions: `pca_regions_{clean|noisy}.png`
  - Tree visualization: `tree.png`
  - Pruned model demo: `cm_{*}_pruned_counts.png`, `cm_{*}_pruned_normalized.png`, `pca_regions_{*}_pruned.png`, `tree_{*}_pruned.png`

- Outputs when running cross-validation (`--cv`):
  - Aggregated confusion matrix figures: `cv_cm_{clean|noisy}_counts.png`, `cv_cm_{clean|noisy}_normalized.png`
  - Representative full-data tree image: `tree_{clean|noisy}_cv.png`

---

## Tips & Notes
- Ensure labels are integers; features should be numeric (float-compatible).
- Reproducibility: CV splitting uses the provided `seed` for shuffling.
- Avoid training with empty arrays; functions validate shapes where necessary.
- `plot_pca_scatter_with_regions` approximates decision regions via PCA back-projection; interpret qualitatively.
- Coursework compliance: Implementation uses `numpy`, `matplotlib`, and standard Python libraries. `requirements.txt` includes `scipy` for environment convenience, but the code does not require it.

