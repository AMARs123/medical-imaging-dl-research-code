"""External test set: BRISC 2025 (Fateh et al., Scientific Data 2026; Zenodo 10.5281/zenodo.17524350, CC BY 4.0).

Uses the tumour slices of the segmentation task (train and test parts; glioma, meningioma, pituitary), which come
with expert masks. Each image is processed exactly like the Figshare slices (preprocess.py): own 0.5-99.5
percentile window, 8 bit, 256 x 256 with area interpolation; masks with nearest neighbour.

Because BRISC was collated from earlier public collections, every BRISC slice is compared with every Figshare
slice (and its left-right mirror image): Pearson correlation of 64 x 64 area-downsampled images. Slices whose best
match reaches the threshold are near-duplicates of the training data and are excluded from the external test.
Writes into data/external/processed/: images.npy, masks.npy, meta.csv (with best_corr and the matching Figshare
index), and dedup_hist.csv.
Usage: python preprocess_brisc.py <brain-tumour-mri-explainability dir> [threshold=0.95]
"""
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

R = sys.argv[1]
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
SIZE = 256
src = os.path.join(R, "data", "external", "brisc2025", "brisc2025", "segmentation_task")
out = os.path.join(R, "data", "external", "processed")
os.makedirs(out, exist_ok=True)
LAB = {"me": 0, "gl": 1, "pi": 2}  # same coding as the Figshare meta.csv (dataset labels 1..3 minus 1)

files = sorted(glob.glob(os.path.join(src, "*", "images", "*.jpg")))
rows, imgs, msks = [], [], []
for f in files:
    name = os.path.basename(f)
    parts = name[:-4].split("_")  # brisc2025_<split>_<index>_<tumor>_<view>_<seq>
    if parts[3] not in LAB:
        continue
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    mf = os.path.join(os.path.dirname(os.path.dirname(f)), "masks", name[:-4] + ".png")
    msk = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        rows.append(dict(file=name, ok=False))
        continue
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    imgs.append(cv2.resize((img * 255).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    m = (cv2.resize((msk > 127).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    msks.append(m)
    rows.append(dict(file=name, ok=True, brisc_split=parts[1], label=LAB[parts[3]], plane=parts[4], height=msk.shape[0],
                     width=msk.shape[1], mask_fraction_resized=float(m.mean())))
meta = pd.DataFrame([r for r in rows if r["ok"]]).reset_index(drop=True)
images, masks = np.stack(imgs), np.stack(msks)
print("BRISC tumour slices read:", len(meta), "unreadable:", sum(not r["ok"] for r in rows), "empty masks:", int((masks.reshape(len(meta), -1).sum(1) == 0).sum()))


def small(a):
    v = np.stack([cv2.resize(x, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for x in a])
    v -= v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)


fig = np.load(os.path.join(R, "data", "processed", "images.npy"))
F = small(fig)
Fm = small(fig[:, :, ::-1])
B = small(images)
c1, c2 = B @ F.T, B @ Fm.T
best = np.maximum(c1.max(1), c2.max(1))
meta["best_corr"] = best
meta["best_figshare_idx"] = np.where(c1.max(1) >= c2.max(1), c1.argmax(1), c2.argmax(1))
meta["mirrored_match"] = c2.max(1) > c1.max(1)
meta["near_duplicate"] = best >= THR
hist = pd.cut(pd.Series(best), [-1, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.01]).value_counts().sort_index()
hist.to_csv(os.path.join(out, "dedup_hist.csv"))
print(hist.to_string())
print("near-duplicates of Figshare slices (corr >= %.2f): %d of %d" % (THR, meta.near_duplicate.sum(), len(meta)))
print(meta.groupby(["label", "near_duplicate"]).size().unstack(fill_value=0))
np.save(os.path.join(out, "images.npy"), images)
np.save(os.path.join(out, "masks.npy"), masks)
meta.to_csv(os.path.join(out, "meta.csv"), index=False)
# side-by-side examples of the closest matches around the threshold, for a visual check
ex = meta.iloc[np.argsort(-np.abs(meta.best_corr.values - THR))[::-1][:12]]
tiles = [np.hstack([images[i], fig[int(j)] if not mm else fig[int(j)][:, ::-1]]) for i, j, mm in zip(ex.index, ex.best_figshare_idx, ex.mirrored_match)]
cv2.imwrite(os.path.join(out, "dedup_examples_near_threshold.png"), np.vstack(tiles))
print("PREP_BRISC_OK")
