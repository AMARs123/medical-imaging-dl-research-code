"""Convert the raw .mat files into compact arrays for training.

Reads every file listed in dataset_index.csv (written by inspect_dataset.py), rescales each image to
8-bit using its own 0.5-99.5 percentile window, resizes image and mask to SIZE x SIZE, and stores:

  images.npy  uint8  [N, SIZE, SIZE]
  masks.npy   uint8  [N, SIZE, SIZE]   (1 = tumour, expert annotation shipped with the dataset)
  meta.csv    idx, label (0..2), pid, fold (authors' cvind), original height/width, mask fraction

No information from any split is used here: the intensity window is computed per image.
"""
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd

RAW, INSPECT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 256
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(INSPECT, "dataset_index.csv"), dtype={"pid": str, "root": str}).sort_values("idx").reset_index(drop=True)
images = np.zeros((len(df), SIZE, SIZE), np.uint8)
masks = np.zeros((len(df), SIZE, SIZE), np.uint8)
for i, row in df.iterrows():
    with h5py.File(os.path.join(RAW, row["file"]), "r") as f:
        g = f["cjdata"]
        img = np.array(g["image"]).astype(np.float32).T  # MATLAB stores column-major
        msk = np.array(g["tumorMask"]).astype(np.uint8).T
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    images[i] = cv2.resize((img * 255).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    masks[i] = (cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

np.save(os.path.join(OUT, "images.npy"), images)
np.save(os.path.join(OUT, "masks.npy"), masks)
meta = df[["idx", "label", "pid", "root", "height", "width", "mask_fraction"] + (["fold"] if "fold" in df else [])].copy()
meta["label"] = meta["label"] - 1  # dataset labels are 1..3
meta["mask_fraction_resized"] = masks.reshape(len(df), -1).mean(1)
meta.to_csv(os.path.join(OUT, "meta.csv"), index=False)
print("PREPROCESS_OK", images.shape, "empty masks after resize:", int((masks.reshape(len(df), -1).sum(1) == 0).sum()))
