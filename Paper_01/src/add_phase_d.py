"""Append phase D to the experiment registry (existing rows and IDs are kept unchanged).

Phase D: partition variability. Phase A compares single partitions; a difference between two fold rules
could come from the luck of one partition. Re-draw the grouped folds with split seeds 1-4 (seed 0 is
phase A) for ID-string and ID-root grouping, ResNet-50 and ViT-Small, 5 folds each: 80 runs. The spread
of pooled scores over the five partitions is the yardstick for the authors' single fixed partition.
"""
import itertools
import sys

import pandas as pd

path = sys.argv[1]
reg = pd.read_csv(path)
if "split_seed" not in reg.columns:
    reg["split_seed"] = 0
if (reg["phase"] == "D_partition").any():
    sys.exit("phase D already registered")
n0 = int(reg["exp_id"].str[3:].astype(int).max())
rows = [dict(phase="D_partition", model=m, protocol=pr, fold=f, seed=0, lambda_sal=0.0, split_seed=s,
             note="partition variability (split seed)", status="PLANNED")
        for m, pr, s, f in itertools.product(["resnet50", "vit_small_patch16_224"], ["pid", "root"], [1, 2, 3, 4], range(5))]
d = pd.DataFrame(rows)
d.insert(0, "exp_id", ["EXP%03d" % (n0 + 1 + i) for i in range(len(d))])
pd.concat([reg, d], ignore_index=True).to_csv(path, index=False)
print("added", len(d), "runs:", d.exp_id.iloc[0], "-", d.exp_id.iloc[-1])
