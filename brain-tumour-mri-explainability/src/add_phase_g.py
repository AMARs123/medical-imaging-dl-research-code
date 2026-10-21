"""Append the Research_Expert round runs to the registry (existing rows and IDs unchanged).

Added 2026-10-16 so that the main comparisons have five seeds and eight architectures:
  B_baseline: seeds 3 and 4 for the seven original architectures (ID-root folds, lambda 0): 70 runs.
  B_baseline: MobileViT-S (CNN-transformer hybrid), seeds 0-4: 25 runs.
  C_saliency: lambda 0.25 (the recommended weight), seeds 3 and 4, for the four phase-C architectures: 40 runs.
The lambda sweep (0.5, 1, 2) and the ablations D-F keep three seeds; the paper states n per table.
"""
import itertools
import sys

import pandas as pd

path = sys.argv[1]
reg = pd.read_csv(path)
if (reg["note"] == "added seeds, Research_Expert round").any():
    sys.exit("phase G rows already registered")
B7 = ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "vit_small_patch16_224", "deit_small_patch16_224",
      "swin_tiny_patch4_window7_224"]
C4 = ["resnet50", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]
rows = []
common = dict(protocol="root", split_seed=0, dilate=1, train_frac=1.0, note="added seeds, Research_Expert round", status="PLANNED")
for m, f, s in itertools.product(B7, range(5), [3, 4]):
    rows.append(dict(phase="B_baseline", model=m, fold=f, seed=s, lambda_sal=0.0, **common))
for f, s in itertools.product(range(5), range(5)):
    rows.append(dict(phase="B_baseline", model="mobilevit_s", fold=f, seed=s, lambda_sal=0.0, **common))
for m, f, s in itertools.product(C4, range(5), [3, 4]):
    rows.append(dict(phase="C_saliency", model=m, fold=f, seed=s, lambda_sal=0.25, **common))
n0 = int(reg["exp_id"].str[3:].astype(int).max())
d = pd.DataFrame(rows)[reg.columns.drop("exp_id")]
d.insert(0, "exp_id", ["EXP%03d" % (n0 + 1 + i) for i in range(len(d))])
pd.concat([reg, d], ignore_index=True).to_csv(path, index=False)
print("added", len(d), "runs:", d.exp_id.iloc[0], "-", d.exp_id.iloc[-1])
