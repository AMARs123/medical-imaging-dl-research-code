"""Append ablation phases E and F to the registry (existing rows and IDs unchanged).

E_dilation: mask dilation of the evidence loss, 0 and 2 cells (default 1 is in phase C), lambda = 0.25,
            ConvNeXt-T and ViT-S, ID-root folds, 5 folds x 3 seeds: 60 runs.
F_datafrac: data efficiency. 25% and 50% of training ID roots (per class, same subset for both lambdas),
            lambda 0 and 0.25, ConvNeXt-T and ViT-S, 5 folds x 3 seeds: 120 runs.
"""
import itertools
import sys

import pandas as pd

path = sys.argv[1]
reg = pd.read_csv(path)
if reg["phase"].isin(["E_dilation", "F_datafrac"]).any():
    sys.exit("phases E/F already registered")
for col, default in [("dilate", 1), ("train_frac", 1.0)]:
    if col not in reg.columns:
        reg[col] = default
M = ["convnext_tiny", "vit_small_patch16_224"]
rows = []
for m, dil, f, s in itertools.product(M, [0, 2], range(5), range(3)):
    rows.append(dict(phase="E_dilation", model=m, protocol="root", fold=f, seed=s, lambda_sal=0.25, split_seed=0,
                     dilate=dil, train_frac=1.0, note="mask dilation ablation", status="PLANNED"))
for m, frac, lam, f, s in itertools.product(M, [0.25, 0.5], [0.0, 0.25], range(5), range(3)):
    rows.append(dict(phase="F_datafrac", model=m, protocol="root", fold=f, seed=s, lambda_sal=lam, split_seed=0,
                     dilate=1, train_frac=frac, note="training-data fraction ablation", status="PLANNED"))
n0 = int(reg["exp_id"].str[3:].astype(int).max())
d = pd.DataFrame(rows)
d.insert(0, "exp_id", ["EXP%03d" % (n0 + 1 + i) for i in range(len(d))])
pd.concat([reg, d], ignore_index=True).to_csv(path, index=False)
print("added", len(d), "runs:", d.exp_id.iloc[0], "-", d.exp_id.iloc[-1])
