"""Paper 09 preprocessing: Kvasir-SEG and HAM10000 (images + lesion masks) -> 256 x 256 arrays.

Usage: python preprocess.py <Paper_09 dir>
Kvasir-SEG: kvasir-seg.zip (images/*.jpg, masks/*.jpg; masks thresholded at 127). Whole image resized (no crop).
HAM10000: images from the HAM10000 archives of Paper 02 (same 256 x 256 whole-image resize), masks from
HAM10000_segmentations_lesion_tschandl.zip resized with nearest-neighbour interpolation.
Splits (fixed, seed 0):
  Kvasir: no patient identifiers are published; random 70/10/20 split of the 1,000 images (limitation: possible
          repeated polyps of one patient across splits cannot be excluded).
  HAM   : 'source' domains. Development = vidir_modern + rosendahl + vienna_dias, split 80/10/10 by lesion_id;
          shifted test set = all vidir_molemax images (a different acquisition system), never used in training.
Outputs: data/processed/{kvasir,ham}_{images,masks}.npy, {kvasir,ham}_meta.csv
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

R = sys.argv[1]
out = os.path.join(R, "data", "processed")
os.makedirs(out, exist_ok=True)
S = 256
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ Kvasir-SEG
z = zipfile.ZipFile(os.path.join(R, "data", "raw", "kvasir-seg.zip"))
names = sorted(n for n in z.namelist() if n.startswith("Kvasir-SEG/images/") and n.endswith(".jpg"))
X = np.zeros((len(names), S, S, 3), np.uint8); M = np.zeros((len(names), S, S), np.uint8); rows = []
for i, n in enumerate(names):
    im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
    mk = Image.open(io.BytesIO(z.read(n.replace("/images/", "/masks/")))).convert("L")
    rows.append(dict(image=os.path.basename(n), h=im.height, w=im.width, mask_frac_orig=float((np.asarray(mk) > 127).mean())))
    X[i] = np.asarray(im.resize((S, S), Image.BICUBIC))
    M[i] = (np.asarray(mk.resize((S, S), Image.NEAREST)) > 127).astype(np.uint8)
perm = rng.permutation(len(names))
split = np.empty(len(names), object)
split[perm[:700]], split[perm[700:800]], split[perm[800:]] = "train", "val", "test"
km = pd.DataFrame(rows); km["split"] = split; km["mask_frac"] = M.reshape(len(M), -1).mean(1)
np.save(os.path.join(out, "kvasir_images.npy"), X); np.save(os.path.join(out, "kvasir_masks.npy"), M)
km.to_csv(os.path.join(out, "kvasir_meta.csv"), index=False)
print("kvasir", km.split.value_counts().to_dict(), flush=True)

# ------------------------------------------------------------------ HAM10000
p2 = os.path.join(R, "..", "Paper_02")
hm = pd.read_csv(os.path.join(p2, "data", "processed", "ham_meta.csv"))
himg = np.load(os.path.join(p2, "data", "processed", "ham_images.npy"), mmap_mode="r")
zs = zipfile.ZipFile(os.path.join(p2, "data", "raw", "HAM10000_segmentations_lesion_tschandl.zip"))
mnames = {os.path.basename(n).replace("_segmentation.png", ""): n for n in zs.namelist() if n.endswith(".png")}
HM = np.zeros((len(hm), S, S), np.uint8)
has = np.zeros(len(hm), bool)
for i, iid in enumerate(hm.image_id):
    if iid in mnames:
        mk = Image.open(io.BytesIO(zs.read(mnames[iid]))).convert("L")
        HM[i] = (np.asarray(mk.resize((S, S), Image.NEAREST)) > 127).astype(np.uint8); has[i] = True
hm["has_mask"] = has
hm["mask_frac"] = HM.reshape(len(HM), -1).mean(1)
dev = hm.dataset != "vidir_molemax"
les = hm.loc[dev, "lesion_id"].unique()
les = les[rng.permutation(len(les))]
n = len(les)
lsplit = {l: ("train" if k < 0.8 * n else "val" if k < 0.9 * n else "test") for k, l in enumerate(les)}
hm["split"] = np.where(dev, hm.lesion_id.map(lsplit), "shift")
np.save(os.path.join(out, "ham_images.npy"), np.asarray(himg)); np.save(os.path.join(out, "ham_masks.npy"), HM)
hm.to_csv(os.path.join(out, "ham_meta.csv"), index=False)
print("ham", hm.split.value_counts().to_dict(), "masks", int(has.sum()), flush=True)
print("PREPROCESS_DONE")
