"""Diagnose where the class-evidence maps put their maximum (CPU only, reads saved outputs).

Usage: python diag_evidence.py <brain-tumour-mri-explainability dir> <out_csv>
For each finished run: share of test images whose evidence maximum sits on the outermost ring of cells,
the share taken by the single most frequent argmax cell (a fixed 'artefact' cell would push this up),
the share of argmax cells that touch the tumour mask (pointing at map resolution), and the chance level
of that pointing (mean share of cells touching the mask).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

root, out = sys.argv[1], sys.argv[2]
masks = np.load(os.path.join(root, "data", "processed", "masks.npy"), mmap_mode="r")
reg = pd.read_csv(os.path.join(root, "experiments", "experiment_registry.csv"))
rows = []
for r in reg.to_dict("records"):
    d = os.path.join(root, "experiments", r["exp_id"])
    if not os.path.exists(os.path.join(d, "results.json")):
        continue
    ev = np.load(os.path.join(d, "evidence_test.npy")).astype(np.float32)  # [N, K, h, w]
    pr = pd.read_csv(os.path.join(d, "predictions.csv"))
    e = ev[np.arange(len(pr)), pr["pred"].values]  # predicted-class map
    n, h, w = e.shape
    am = e.reshape(n, -1).argmax(1)
    yy, xx = am // w, am % w
    m = torch.from_numpy(np.ascontiguousarray(masks[pr["array_index"].values]).astype(np.float32))[:, None]
    mc = F.adaptive_max_pool2d(m, (h, w))[:, 0].numpy() > 0
    hit = mc.reshape(n, -1)[np.arange(n), am]
    rows.append(dict(r, map_h=h, border_argmax=float(((yy == 0) | (yy == h - 1) | (xx == 0) | (xx == w - 1)).mean()),
                     border_cells=float(1 - (h - 2) * (w - 2) / (h * w)),
                     top_cell_share=float(np.bincount(am, minlength=h * w).max() / n),
                     pointing_cell=float(hit.mean()), pointing_cell_chance=float(mc.reshape(n, -1).mean()),
                     evidence_positive_share=float((e > 0).mean())))
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
cols = ["border_argmax", "border_cells", "top_cell_share", "pointing_cell", "pointing_cell_chance", "evidence_positive_share"]
print(df[df["protocol"] == "root"].groupby(["model", "lambda_sal"])[cols].mean().round(3).to_string())
