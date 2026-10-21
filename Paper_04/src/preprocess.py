"""Build the Paper 04 image arrays: Guangzhou (Kermany) and NIH ChestX-ray14 subsets, grayscale 256 x 256.

Usage: python preprocess.py <Paper_04 dir>
Every image is padded to a square with black borders (keeping its aspect ratio) and resized to 256 x 256 (area
interpolation). A second array holds the same images after CLAHE (contrast-limited adaptive histogram equalisation,
clip limit 2.0, 8 x 8 tiles), used by the histogram-equalisation condition.
Outputs in data/processed/: images.npy, images_clahe.npy (uint8 [N, 256, 256]), meta.csv, file_check.csv
Kermany v3 file names are <CLASS>-<patient>-<number>.jpeg with CLASS in NORMAL, BACTERIA, VIRUS.
"""
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

R = sys.argv[1]
raw = os.path.join(R, "data", "raw")
out = os.path.join(R, "data", "processed")
os.makedirs(out, exist_ok=True)
S = 256
rows = []
for split in ["train", "test"]:
    for f in sorted(glob.glob(os.path.join(raw, "kermany", "CellData", "chest_xray", split, "*", "*.jpeg"))):
        b = os.path.basename(f)[:-5]
        cls, pat, _ = b.split("-")
        rows.append(dict(path=f, image=os.path.basename(f), group="G", label=int(cls != "NORMAL"), subtype=cls.lower(),
                         patient="G" + pat, official_split=split, age=np.nan, view="", sex=""))
nih = pd.read_csv(os.path.join(raw, "nih_selection.csv"))
for r in nih.to_dict("records"):
    rows.append(dict(path=os.path.join(raw, "nih", "images", r["image"]), image=r["image"], group={"nih_adult": "NA", "nih_paed": "NP"}[r["group"]],
                     label=int(r["label"]), subtype="pneumonia" if r["label"] else "no_finding", patient="N%d" % r["patient"],
                     official_split="", age=r["age"], view=r["view"], sex=r["sex"]))
meta = pd.DataFrame(rows)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
A = np.lib.format.open_memmap(os.path.join(out, "images.npy"), "w+", np.uint8, (len(meta), S, S))
C = np.lib.format.open_memmap(os.path.join(out, "images_clahe.npy"), "w+", np.uint8, (len(meta), S, S))
chk = []
for i, r in enumerate(meta.to_dict("records")):
    im = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
    ok = im is not None
    chk.append(dict(image=r["image"], group=r["group"], ok=ok, h=im.shape[0] if ok else 0, w=im.shape[1] if ok else 0,
                    mean=float(im.mean()) if ok else np.nan))
    if not ok:
        continue
    h, w = im.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    A[i] = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA)
    C[i] = clahe.apply(A[i])
    if i % 2000 == 0:
        print(i, flush=True)
A.flush(); C.flush()
chk = pd.DataFrame(chk)
chk.to_csv(os.path.join(out, "file_check.csv"), index=False)
meta["ok"] = chk.ok.values
meta["height"], meta["width"] = chk.h.values, chk.w.values
meta.drop(columns=["path"]).to_csv(os.path.join(out, "meta.csv"), index=False)
print(meta.groupby(["group", "label"]).agg(images=("image", "size"), patients=("patient", "nunique"), ok=("ok", "sum")))
print("PREPROCESS_DONE")
