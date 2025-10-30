## WiFi Decision Tree Coursework (For_70050)

This project implements a small end-to-end pipeline to train and evaluate a Decision Tree classifier on WiFi RSSI datasets, including data loading, cross-validation, metrics, and multiple visualizations. It does not use scikit-learn.

Note: The implementation uses only `numpy`, `matplotlib`, and Python's standard library.

### Layout (key locations)
- **For_70050/wifi_db/**: dataset files (defaults: `clean_dataset.txt`, `noisy_dataset.txt`)
- **For_70050/figures/**: output figures
- **For_70050/main.py**: CLI entry (dataset paths are provided via arguments)
- **run.sh**: convenience commands (you can edit dataset paths here)

### Environment & install
```bash
# optional: create a virtual environment
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r For_70050/requirements.txt
```

### Quick Start
```bash
# 1) Train on full clean dataset (saves tree.png and other figures)
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset clean \
  --outdir "For_70050/figures"

# 2) 10-fold cross-validation on clean dataset
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset clean \
  --cv --k 10 \
  --outdir "For_70050/figures"

# 3) 10-fold cross-validation on noisy dataset
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset noisy \
  --cv --k 10 \
  --outdir "For_70050/figures"

# 4) Nested 10-fold cross-validation on clean dataset
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset clean \
  --prune-cv \
  --cv --k 10 \
  --outdir "For_70050/figures"

# 5) Nested 10-fold cross-validation on noisy dataset
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset noisy \
  --prune-cv \
  --cv --k 10 \
  --outdir "For_70050/figures"

# Or run the helper script (you can edit dataset paths inside)
bash run.sh
```

---

## Where to change dataset paths (for marker to import new data)

### Pass paths via CLI arguments 
Use this template and replace placeholders with your filenames:

Template:
```bash
python For_70050/main.py \
  --clean "<PATH_TO_CLEAN_TXT>" \
  --noisy "<PATH_TO_NOISY_TXT>" \
  --dataset clean \
  --outdir "For_70050/figures"
```

Notes:
- To use the noisy dataset instead, set `--dataset noisy`.
- You can pass both paths every time, only the one matching `--dataset` is used.

Absolute/relative path examples:
```bash
# absolute paths
python For_70050/main.py \
  --clean "/Users/marker/Desktop/data/my_clean.txt" \
  --noisy "/Users/marker/Desktop/data/my_noisy.txt" \
  --dataset clean \
  --outdir "For_70050/figures"

# relative to your current working directory
python For_70050/main.py \
  --clean "../data/my_clean.txt" \
  --noisy "../data/my_noisy.txt" \
  --dataset noisy \
  --outdir "For_70050/figures"
```

---

## Data format
- Text file, whitespace-delimited (spaces or tabs)
- Last column is the integer class label, preceding columns are float features

---

## Workflow overview
1. Select dataset with `--dataset clean|noisy` and load it via `wifi_utils.load_wifi_dataset`
2. If `--cv` is set, run K-fold cross-validation (`metrics.cross_val_evaluate`) and save averaged metrics/plots
3. Otherwise, train on full data and save confusion matrices, and the tree visualization

---

## Key modules (at a glance)
- **decision_tree.py**: binary tree with information gain splits (`DecisionTreeClassifier`)
- **metrics.py**: accuracy, confusion matrix, K-fold evaluation
- **visualize.py**: confusion matrix plots, tree drawing
- **wifi_utils.py**: dataset loader, K-fold indices
- **main.py**: CLI entry and training/evaluation logic

---

## Notes
- Ensure labels are integers, features are numeric
- CV uses a fixed random seed for reproducibility
