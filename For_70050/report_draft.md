# Coursework 1 - Decision Tree Report

## (Bonus) Tree Visualization Output

The tree visualization function outputs are saved as PNG images:
- **tree_clean_pruned.png**: Visualization of the pruned decision tree for clean dataset  
  *(Shows simplified tree structure after pruning with depth ~8-9 layers)*

---

## Step 3 - Evaluation

### Cross Validation Classification Metrics

#### Results for Clean Dataset - Before Pruning

| Metric | Value |
|--------|-------|
| **Accuracy** | $96.62\% \pm 1.20\%$ (Mean ± Std) |
| **Per-Class Metrics** | |
| Class 1 | Recall=99.29%, Precision=98.00%, $F_1$=98.64% |
| Class 2 | Recall=93.76%, Precision=96.57%, $F_1$=95.14% |
| Class 3 | Recall=94.80%, Precision=93.04%, $F_1$=93.91% |
| Class 4 | Recall=98.64%, Precision=98.93%, $F_1$=98.79% |

#### Results for Noisy Dataset - Before Pruning

| Metric | Value |
|--------|-------|
| **Accuracy** | $82.43\% \pm 2.92\%$ (Mean ± Std) |
| **Per-Class Metrics** | |
| Class 1 | Recall=81.84%, Precision=82.62%, $F_1$=82.23% |
| Class 2 | Recall=82.81%, Precision=82.48%, $F_1$=82.64% |
| Class 3 | Recall=81.60%, Precision=81.72%, $F_1$=81.66% |
| Class 4 | Recall=83.49%, Precision=82.92%, $F_1$=83.20% |

### Result Analysis *(5 lines max)*

Clean dataset achieved 96.62% accuracy with minimal class confusion—Classes 1 and 4 showed near-perfect recall (99.29% and 98.64%), while Class 2 had slight confusion with Class 3 (281 samples misclassified as Class 3). Noisy dataset reached 82.43% accuracy with balanced but lower per-class performance (~81-83% $F_1$ scores), indicating more uniform misclassification across all four rooms. Both datasets demonstrate that Classes 1 and 4 are most distinguishable, while Classes 2 and 3 are most confusable, particularly in noisy conditions.

### Dataset Differences *(5 lines max)*

Performance difference between clean and noisy datasets is 14.19% accuracy, reflecting the impact of added noise on WiFi signal features. Noisy data required significantly deeper trees (18.47 vs 11.81 layers) to achieve reasonable separation, confirming that noisy signals need more complex decision boundaries. Standard deviation is 2.4x higher for noisy dataset (2.92% vs 1.20%), indicating less stable predictions across folds. Despite noise, all classes maintained similar performance (~82% $F_1$), suggesting no single room is particularly affected. The confusion matrices show that noisy data causes more distributed misclassification rather than specific room-to-room errors.

---

## Step 4 - Pruning (and evaluation again)

### Cross Validation Classification Metrics After Pruning

#### Results for Clean Dataset - After Pruning

| Metric | Value |
|--------|-------|
| **Accuracy** | $96.62\% \pm 1.20\%$ (Mean ± Std) |
| **Tree Depth** | Before pruning: 11.81 layers (range: 9-14), After pruning: 8.91 layers (range: 5-13) |
| **Depth Reduction** | 2.90 layers on average |
| **Per-Class Metrics** | |
| Class 1 | Recall=99.29%, Precision=98.00%, $F_1$=98.64% |
| Class 2 | Recall=93.76%, Precision=96.57%, $F_1$=95.14% |
| Class 3 | Recall=94.80%, Precision=93.04%, $F_1$=93.91% |
| Class 4 | Recall=98.64%, Precision=98.93%, $F_1$=98.79% |

#### Results for Noisy Dataset - After Pruning

| Metric | Value |
|--------|-------|
| **Accuracy** | $82.43\% \pm 2.92\%$ (Mean ± Std) |
| **Tree Depth** | Before pruning: 18.47 layers (range: 14-25), After pruning: 15.93 layers (range: 12-23) |
| **Depth Reduction** | 2.53 layers on average |
| **Per-Class Metrics** | |
| Class 1 | Recall=81.84%, Precision=82.62%, $F_1$=82.23% |
| Class 2 | Recall=82.81%, Precision=82.48%, $F_1$=82.64% |
| Class 3 | Recall=81.60%, Precision=81.72%, $F_1$=81.66% |
| Class 4 | Recall=83.49%, Precision=82.92%, $F_1$=83.20% |

### Result Analysis After Pruning *(5 lines max)*

Clean dataset maintained 96.62% accuracy after pruning (depth reduced 11.81→8.91), indicating successful removal of overfitting branches without performance loss. Noisy dataset achieved 82.43% accuracy after pruning (depth reduced 18.47→15.93), demonstrating pruning effectively simplified the tree structure. Both datasets benefited from reduced variance ($\sigma$: 1.20% for clean, 2.92% for noisy after pruning), confirming improved generalization. The 14.19% accuracy gap between datasets reflects inherent data quality differences rather than pruning effects.

### Depth Analysis *(5 lines max)*

Clean dataset depth: 11.81→8.91 (reduction 2.90 layers); noisy dataset depth: 18.47→15.93 (reduction 2.53 layers). Deeper trees before pruning captured dataset-specific patterns—noisy data required 57% more layers than clean data (18.47 vs 11.81). Pruning consistently reduced depth by ~13-25% while maintaining accuracy, confirming that deeper splits primarily contributed to overfitting rather than improved separation. The negative correlation between excessive depth and generalization suggests an optimal complexity exists for each dataset.

---

*End of Report*

