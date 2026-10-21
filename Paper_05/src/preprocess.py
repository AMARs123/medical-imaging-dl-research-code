"""Paper 05 preprocessing: DDR (grading part) and DeepDRiD regular fundus images -> disc-cropped 384 x 384 RGB arrays.

Usage: python preprocess.py <Paper_05 dir>
Fundus crop: pixels brighter than 10 (max over channels, after a 5 x 5 median filter) define the visible retina; the
image is cropped to their bounding box, padded to a square with black and resized to 384 x 384 (area interpolation).
DDR: labels from DR_grading/{train,valid,test}.txt (official split); grade 5 (ungradable) is excluded and counted.
DeepDRiD: all regular fundus images with a DR grade in the released label files (external test only).
Outputs: data/processed/images.npy (uint8 [N,384,384,3]), meta.csv (dataset, split, image, grade, patient, eye),
file_check.csv, excluded.csv
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
S = 384
rows, excluded = [], []
ddr = os.path.join(raw, "ddr_mirror", "DR_grading")
for split in ["train", "valid", "test"]:
    lab = pd.read_csv(os.path.join(ddr, split + ".txt"), sep=r"\s+", header=None, names=["image", "grade"])
    for r in lab.itertuples():
        if r.grade == 5:
            excluded.append(dict(dataset="DDR", split=split, image=r.image, reason="ungradable (grade 5)"))
            continue
        rows.append(dict(dataset="DDR", split=split, image=r.image, grade=int(r.grade), path=os.path.join(ddr, split, r.image),
                         patient="", eye=""))

# DeepDRiD regular fundus images: label csv files are found by name; columns are read defensively and logged
dd = glob.glob(os.path.join(raw, "deepdrid", "**", "regular_fundus_images", "**", "*.csv"), recursive=True)
for f in sorted(dd):
    t = pd.read_csv(f)
    cols = {c.lower().strip(): c for c in t.columns}
    print("DeepDRiD label file", os.path.relpath(f, raw), list(t.columns)[:8], len(t), flush=True)
    img_col = next((cols[c] for c in cols if c in ("image_path", "image_id", "image")), None)
    if img_col is None:
        continue
    grade_cols = [cols[c] for c in cols if c in ("patient_dr_level", "dr_level", "left_eye_dr_level", "right_eye_dr_level")]
    for r in t.to_dict("records"):
        g = None
        for c in grade_cols:
            if pd.notna(r.get(c)) and str(r.get(c)).strip() != "":
                g = int(float(r[c])); break
        if g is None:
            continue
        name = str(r[img_col]).replace("\\", "/")
        cand = glob.glob(os.path.join(os.path.dirname(f), "**", os.path.basename(name) + "*"), recursive=True)
        cand = [c for c in cand if c.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not cand:
            excluded.append(dict(dataset="DeepDRiD", split=os.path.basename(f), image=name, reason="image file not found"))
            continue
        rows.append(dict(dataset="DeepDRiD", split=os.path.basename(os.path.dirname(f)), image=os.path.basename(cand[0]), grade=g,
                         path=cand[0], patient=str(r.get("patient_id", "")), eye=""))
meta = pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)
A = np.lib.format.open_memmap(os.path.join(out, "images.npy"), "w+", np.uint8, (len(meta), S, S, 3))
chk = []
for i, r in enumerate(meta.to_dict("records")):
    im = cv2.imread(r["path"], cv2.IMREAD_COLOR)
    ok = im is not None
    row = dict(image=r["image"], dataset=r["dataset"], ok=ok)
    if ok:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        row.update(h=im.shape[0], w=im.shape[1])
        m = cv2.medianBlur(im.max(2), 5) > 10
        ys, xs = np.where(m)
        if len(ys) > 100:
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = im.shape[:2]; s = max(h, w)
        sq = np.zeros((s, s, 3), np.uint8)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
        A[i] = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA)
        row.update(crop_h=h, crop_w=w)
    chk.append(row)
    if i % 2000 == 0:
        print(i, len(meta), flush=True)
A.flush()
chk = pd.DataFrame(chk)
meta["ok"] = chk.ok.values
meta.drop(columns=["path"]).to_csv(os.path.join(out, "meta.csv"), index=False)
chk.to_csv(os.path.join(out, "file_check.csv"), index=False)
pd.DataFrame(excluded).to_csv(os.path.join(out, "excluded.csv"), index=False)
print(meta.groupby(["dataset", "split", "grade"]).size().unstack(fill_value=0))
print("excluded", len(excluded), "| unreadable", int((~meta.ok).sum()))
print("PREPROCESS_DONE")
