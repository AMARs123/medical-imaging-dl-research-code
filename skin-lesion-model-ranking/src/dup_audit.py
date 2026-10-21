"""Second duplicate audit (replaces the whole-image dHash rule of inspect_dataset.py).

Why: a visual check (2026-10-15) showed that whole-image dHash + 32x32 grey correlation matched many *different*
lesions, because most dermoscopy images share the same dark circular vignette. This audit compares only the central
region (central 50% of the image, where the lesion is) in colour.

Feature: central 128 x 128 crop of the 256 x 256 array -> 24 x 24 RGB -> per-image z-score (1728 values).
For every image, the most similar image with a *different* lesion_id (HAM10000 vs HAM10000) and the most similar
HAM10000 image for every ISIC 2018 test image are recorded with their Pearson correlation.
Candidates are then labelled by eye on contact sheets (dup_audit_labels.csv); only labelled pairs are reported.
Usage: python dup_audit.py <skin-lesion-model-ranking dir>
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

root = sys.argv[1]
proc = os.path.join(root, "data", "processed")
out = os.path.join(root, "data", "inspection")
ham = pd.read_csv(os.path.join(proc, "ham_meta.csv"))
test = pd.read_csv(os.path.join(proc, "test_meta.csv"))
hi = np.load(os.path.join(proc, "ham_images.npy"), mmap_mode="r")
ti = np.load(os.path.join(proc, "test_images.npy"), mmap_mode="r")


def feats(arr):
    fs = []
    for s in range(0, len(arr), 1000):
        x = torch.from_numpy(np.asarray(arr[s:s + 1000, 64:192, 64:192])).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(24, 24), mode="area").flatten(1)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        fs.append(x / np.sqrt(x.shape[1]))
    return torch.cat(fs)


fh, ft = feats(hi), feats(ti)
les = ham["lesion_id"].values
rows = []
for s in range(0, len(fh), 1000):
    c = fh[s:s + 1000] @ fh.T
    for i in range(c.shape[0]):
        same = les == les[s + i]
        c[i, torch.from_numpy(same)] = -2  # exclude itself and its own lesion's images
    v, j = c.max(1)
    for i in range(len(v)):
        a, b = s + i, int(j[i])  # (a, b) and (b, a) may both appear; duplicates are dropped below
        rows.append(dict(set="ham-ham", a=ham.image_id[a], b=ham.image_id[b], corr=float(v[i]), dx_a=ham.dx[a], dx_b=ham.dx[b],
                         lesion_a=les[a], lesion_b=les[b]))
c = ft @ fh.T
v, j = c.max(1)
for i in range(len(v)):
    b = int(j[i])
    rows.append(dict(set="ham-test", a=ham.image_id[b], b=test.image_id[i], corr=float(v[i]), dx_a=ham.dx[b], dx_b=test.dx[i],
                     lesion_a=les[b], lesion_b=test.lesion_id[i]))
df = pd.DataFrame(rows)
df["pair"] = df.apply(lambda r: "|".join(sorted([r.a, r.b])), axis=1)
df = df.sort_values("corr", ascending=False).drop_duplicates("pair")
df.to_csv(os.path.join(out, "dup_audit_candidates.csv"), index=False)
for st in ["ham-ham", "ham-test"]:
    print(st, df[df.set == st]["corr"].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).round(4).to_dict())

# contact sheets of the top 64 candidates of each set, for labelling by eye
hidx, tidx = dict(zip(ham.image_id, ham.index)), dict(zip(test.image_id, test.index))
S = 110
for st in ["ham-ham", "ham-test"]:
    top = df[df.set == st].head(64).reset_index(drop=True)
    sheet = Image.new("RGB", (8 * (2 * S + 8), 8 * (S + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in top.iterrows():
        x0, y0 = (k % 8) * (2 * S + 8), (k // 8) * (S + 14) + 12
        A = hi[hidx[r.a]]
        B = ti[tidx[r.b]] if st == "ham-test" else hi[hidx[r.b]]
        sheet.paste(Image.fromarray(np.asarray(A)).resize((S, S)), (x0, y0))
        sheet.paste(Image.fromarray(np.asarray(B)).resize((S, S)), (x0 + S, y0))
        d.text((x0, y0 - 11), "#%d %s/%s %.3f" % (k, r.dx_a, r.dx_b, r["corr"]), fill="black")
    sheet.save(os.path.join(out, "dup_audit_sheet_%s.png" % st))
print("DUP_AUDIT_DONE")
