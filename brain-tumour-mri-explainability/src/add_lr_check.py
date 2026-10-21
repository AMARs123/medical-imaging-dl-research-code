"""Register the learning-rate check (Research_Expert rule A-058, 2026-10-16) in its own registry,
experiments/lr_registry.csv, so that these runs never enter the main analyses.

Every architecture gets the same three-value search: learning rate 3e-5, 1e-4 (the shared recipe; existing
seed-0 runs of group B) and 3e-4, on the five ID-root folds with seed 0. Only the validation macro-F1 (the
selection criterion) is used to judge the search; test scores of these runs are not reported as results.
"""
import itertools
import os
import sys

import pandas as pd

root = sys.argv[1]
main = pd.read_csv(os.path.join(root, "experiments", "experiment_registry.csv"))
out = os.path.join(root, "experiments", "lr_registry.csv")
if os.path.exists(out):
    sys.exit("lr_registry.csv exists")
M = ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilevit_s", "vit_small_patch16_224",
     "deit_small_patch16_224", "swin_tiny_patch4_window7_224"]
rows = [dict(phase="H_lr", model=m, protocol="root", fold=f, seed=0, lambda_sal=0.0, note="learning-rate check",
             status="PLANNED", split_seed=0, dilate=1, train_frac=1.0, lr=lr)
        for m, lr, f in itertools.product(M, [3e-5, 3e-4], range(5))]
n0 = int(main["exp_id"].str[3:].astype(int).max())
d = pd.DataFrame(rows)
d.insert(0, "exp_id", ["EXP%03d" % (n0 + 1 + i) for i in range(len(d))])
d.to_csv(out, index=False)
print("registered", len(d), "runs:", d.exp_id.iloc[0], "-", d.exp_id.iloc[-1])
