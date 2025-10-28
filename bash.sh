# train and visualize model 

python For_70050/main.py \
  --clean "For_70050/wifi_db/clean_dataset.txt" \
  --noisy "For_70050/wifi_db/noisy_dataset.txt" \
  --k 5 --depth_min 1 --depth_max 15 \
  --plot "For_70050/figures/depth_cv.png" \
  --outdir "For_70050/figures"
