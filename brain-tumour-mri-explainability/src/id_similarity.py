"""Are images whose patient IDs differ only by a trailing letter more alike than images of unrelated IDs?

For every image, find its most similar other image (Pearson correlation of 64x64 downsampled,
z-scored pixels) within three pools of the SAME tumour class:
  same_pid     : other slices with the identical PID string
  same_root    : slices whose PID shares the root but differs in the trailing letter
  other_root   : slices from a different ID root
Writes id_similarity.csv (one row per image) and id_similarity_summary.json. CPU only.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

proc, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
images = np.load(os.path.join(proc, "images.npy"), mmap_mode="r")
meta = pd.read_csv(os.path.join(proc, "meta.csv"), dtype={"pid": str, "root": str})
X = np.stack([cv2.resize(np.asarray(im), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for im in images])
X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
C = (X @ X.T) / X.shape[1]
np.fill_diagonal(C, -2)
pid, root, lab = meta["pid"].values, meta["root"].values, meta["label"].values
same_class = lab[:, None] == lab[None, :]
same_pid = pid[:, None] == pid[None, :]
same_root = root[:, None] == root[None, :]
cols = []
for name, pool in (("same_pid", same_pid), ("same_root", same_root & ~same_pid), ("other_root", ~same_root & same_class)):
    best = np.where(pool, C, -2).max(1)
    cols.append(pd.Series(np.where(best > -2, best, np.nan), name=name))
df = pd.concat([meta[["idx", "label", "pid", "root"]], *cols], axis=1)
df.to_csv(os.path.join(out, "id_similarity.csv"), index=False)
multi = df[df["same_root"].notna()]
s = dict(
    n_images=len(df), n_images_with_root_variant=len(multi),
    median_best_corr_all_images=dict(same_pid=float(df["same_pid"].median()), other_root=float(df["other_root"].median())),
    among_images_with_variant=dict(
        same_pid=float(multi["same_pid"].median()), same_root=float(multi["same_root"].median()),
        other_root=float(multi["other_root"].median()),
        share_same_root_beats_other_root=float((multi["same_root"] > multi["other_root"]).mean())),
    quantiles={k: {str(q): float(df[k].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} for k in ("same_pid", "same_root", "other_root")},
)
try:
    from scipy.stats import wilcoxon

    w = wilcoxon(multi["same_root"], multi["other_root"])
    s["wilcoxon_same_root_vs_other_root"] = dict(statistic=float(w.statistic), p=float(w.pvalue), n=len(multi))
except Exception as e:  # recorded, not hidden
    s["wilcoxon_error"] = repr(e)
json.dump(s, open(os.path.join(out, "id_similarity_summary.json"), "w"), indent=1)
print(json.dumps(s, indent=1))
