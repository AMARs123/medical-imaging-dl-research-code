"""Collect every finished run into tables and run the planned statistical tests.

Usage: python aggregate.py <experiments_dir> <out_dir>
Reads experiments/experiment_registry.csv and each EXPxxx/results.json; nothing is recomputed or imputed.
Runs without results.json are listed as missing and excluded. Writes:
  all_runs.csv          one row per finished run (registry columns + scalar test metrics)
  missing_runs.csv      planned runs with no results yet
  summary_A_protocol.csv, summary_B_baseline.csv, summary_C_saliency.csv   mean, sd, n per cell
  tests_A_protocol.csv  paired Wilcoxon signed-rank over folds: each fold-formation rule vs "root"
  tests_C_saliency.csv  paired Wilcoxon over (fold, seed): each lambda vs the lambda = 0 baseline, Holm-corrected
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

exp_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
reg = pd.read_csv(os.path.join(exp_dir, "experiment_registry.csv"))

rows, missing = [], []
for r in reg.to_dict("records"):
    p = os.path.join(exp_dir, r["exp_id"], "results.json")
    if not os.path.exists(p):
        missing.append(r)
        continue
    res = json.load(open(p))
    row = {k: v for k, v in r.items() if k != "status"}
    row.update({k: v for k, v in res["test"].items() if isinstance(v, (int, float))})
    row.update(best_epoch=res["best_epoch"], best_val_macro_f1=res["best_val_macro_f1"],
               train_seconds=res["train_seconds"], n_params=res["n_params"], peak_gpu_mb=res["peak_gpu_mb"])
    for k, v in res["split"].items():
        if isinstance(v, (int, float)):
            row["split_" + k] = v
    rows.append(row)

runs = pd.DataFrame(rows)
for _c, _v in [("split_seed", 0), ("dilate", 1), ("train_frac", 1.0)]:
    runs[_c] = runs[_c].fillna(_v) if _c in runs else _v
runs["main"] = (runs.split_seed == 0) & (runs.dilate == 1) & (runs.train_frac == 1.0)  # not a D/E/F ablation run
runs.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
pd.DataFrame(missing).to_csv(os.path.join(out_dir, "missing_runs.csv"), index=False)
print("finished runs: %d of %d" % (len(runs), len(reg)))
if runs.empty:
    sys.exit(0)

METRICS = [m for m in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "ece", "nll",
                       "loc_pointing", "loc_mass_inside", "loc_iou_area"] if m in runs.columns]


def summarise(df, keys):
    g = df.groupby(keys)[METRICS].agg(["mean", "std", "count"])
    g.columns = ["%s_%s" % c for c in g.columns]
    return g.reset_index()


def holm(p):
    p = np.asarray(p, float)
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a, b, on):
    """Wilcoxon signed-rank on runs matched by `on`; returns n pairs, mean difference (a - b), p."""
    m = a.merge(b, on=on, suffixes=("_a", "_b"))
    out = {}
    for met in METRICS:
        d = (m[met + "_a"] - m[met + "_b"]).dropna().values
        if len(d) < 5 or np.allclose(d, 0):
            out[met] = (len(d), float(d.mean()) if len(d) else np.nan, np.nan)
        else:
            out[met] = (len(d), float(d.mean()), float(wilcoxon(d).pvalue))
    return out


# all runs of one (model, protocol, lambda) cell, whatever phase they were registered in
base = runs[(runs["lambda_sal"] == 0) & runs["main"]]

# A. fold-formation rules
A = runs[runs["phase"] == "A_protocol"]
if len(A):
    summarise(A, ["model", "protocol"]).to_csv(os.path.join(out_dir, "summary_A_protocol.csv"), index=False)
    t = []
    for model, d in A.groupby("model"):
        ref = d[d["protocol"] == "root"]
        for pr in ["slice", "authors", "pid"]:
            for met, (n, diff, p) in paired(d[d["protocol"] == pr], ref, ["fold"]).items():
                t.append(dict(model=model, protocol=pr, reference="root", metric=met, n_pairs=n, mean_diff=diff, p_wilcoxon=p))
    pd.DataFrame(t).to_csv(os.path.join(out_dir, "tests_A_protocol.csv"), index=False)

# B. architecture baselines under the root protocol (phase A root runs are seed 0 of two of the models)
B = base[base["protocol"] == "root"]
if len(B):
    summarise(B, ["model"]).to_csv(os.path.join(out_dir, "summary_B_baseline.csv"), index=False)

# C. saliency-loss weight, compared with the matching lambda = 0 run of the same model, fold and seed
C = runs[runs["main"] & (runs["protocol"] == "root") & runs["model"].isin(runs.loc[runs["phase"] == "C_saliency", "model"].unique())]
if (C["lambda_sal"] > 0).any():
    summarise(C, ["model", "lambda_sal"]).to_csv(os.path.join(out_dir, "summary_C_saliency.csv"), index=False)
    t = []
    for model, d in C.groupby("model"):
        ref = d[d["lambda_sal"] == 0]
        for lam in sorted(x for x in d["lambda_sal"].unique() if x > 0):
            for met, (n, diff, p) in paired(d[d["lambda_sal"] == lam], ref, ["fold", "seed"]).items():
                t.append(dict(model=model, lambda_sal=lam, reference="lambda=0", metric=met, n_pairs=n, mean_diff=diff, p_wilcoxon=p))
    t = pd.DataFrame(t)
    t["p_holm"] = np.nan
    for met, idx in t.groupby("metric").groups.items():  # one family per metric
        ok = t.loc[idx, "p_wilcoxon"].notna()
        if ok.any():
            t.loc[ok[ok].index, "p_holm"] = holm(t.loc[ok[ok].index, "p_wilcoxon"].values)
    t.to_csv(os.path.join(out_dir, "tests_C_saliency.csv"), index=False)
print("written to", out_dir)
