# train and visualize model

# 1) Train on full clean dataset (saves figures incl. tree.png)
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset clean \
  --outdir "For_70050/figures"

# 2) 10-fold CV on clean dataset
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset clean \
  --cv --k 10 \
  --outdir "For_70050/figures"

# 3) 10-fold CV on noisy dataset
python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --dataset noisy \
  --cv --k 10 \
  --outdir "For_70050/figures"
