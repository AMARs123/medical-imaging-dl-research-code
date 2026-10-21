"""Paper 10 analysis from the stored run outputs (no network is run).

Per run (results/processed/runs.csv): condition (architecture, weights, augmentation, mode), seed, full-split accuracies of the
final model (ID validation, OOD validation, test), and the test accuracy under three selection rules (Eq. select), all three read
from the evaluation points of metrics.csv (fixed random subsets of 10,000 patches per split, so that the rules are compared on
the same patches): id = point with the best ID-validation accuracy, ood = point with the best OOD-validation accuracy,
final = last point.
Statistics (results/statistics/):
  conditions.csv  per condition: number of seeds, mean/SD/min/max of full test accuracy and of each rule's test accuracy, ID accuracy
  gains.csv       gain over the reference of the same architecture (ImageNet, base augmentation, fine-tuned) under each rule
                  (Eq. gain; seeds paired by index), 95% bootstrap interval over seeds (2000 resamples), Mann-Whitney U, Holm over
                  the planned family (3 fine-tuned conditions x 2 architectures x 3 rules), share of the OOD gain kept by ID selection
  spread.csv      seed spread of full test accuracy per condition and Brown-Forsythe test against the reference
  probit.csv      per condition: probit-scale fit of final test on final ID accuracy over seeds (Eq. probit) and Spearman rho
Usage: python analyze.py <Paper_10 dir>"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R = sys.argv[1]
P = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
os.makedirs(P, exist_ok=True)
os.makedirs(S, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
ARCH = {"r50_imagenet": "r50", "r50_lunit": "r50", "vits_imagenet": "vits", "vits_lunit": "vits"}
WTS = {"r50_imagenet": "imagenet", "r50_lunit": "pathology", "vits_imagenet": "imagenet", "vits_lunit": "pathology"}
rows = []
for r in reg.itertuples():
    d = os.path.join(R, "experiments", r.exp_id)
    if not os.path.exists(os.path.join(d, "results.json")):
        continue
    res = json.load(open(os.path.join(d, "results.json")))
    m = pd.read_csv(os.path.join(d, "metrics.csv"))
    row = dict(exp_id=r.exp_id, group=r.group, backbone=r.backbone, arch=ARCH[r.backbone], weights=WTS[r.backbone], aug=r.aug, mode=r.mode,
               seed=int(r.seed), acc_id=res["id_val"]["acc"], acc_ood=res["ood_val"]["acc"], acc_test=res["test"]["acc"], n_points=len(m),
               train_seconds=res.get("train_seconds", np.nan))
    for rule, col in [("id", "id_val_acc"), ("ood", "ood_val_acc")]:
        i = int(m[col].values.argmax())  # first point with the best accuracy
        row["test_" + rule] = float(m.test_acc.values[i]); row["point_" + rule] = int(m.step.values[i])
    row["test_final"] = float(m.test_acc.values[-1])
    rows.append(row)
runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(P, "runs.csv"), index=False)
print("runs with results: %d of %d" % (len(runs), len(reg)))
if runs.empty:
    sys.exit(0)
KEY = ["arch", "weights", "aug", "mode"]
RULES = ["id", "ood", "final"]
cond = runs.groupby(KEY).agg(n=("seed", "size"), acc_test=("acc_test", "mean"), sd_test=("acc_test", "std"), min_test=("acc_test", "min"),
                             max_test=("acc_test", "max"), acc_id=("acc_id", "mean"), acc_ood=("acc_ood", "mean"),
                             **{"rule_" + q: ("test_" + q, "mean") for q in RULES}).reset_index()
cond.to_csv(os.path.join(S, "conditions.csv"), index=False)

rng = np.random.default_rng(0)
gains = []
for arch in ["r50", "vits"]:
    ref = runs[(runs.arch == arch) & (runs.weights == "imagenet") & (runs.aug == "none") & (runs["mode"] == "finetune")].set_index("seed")
    for w, a, mo in [("imagenet", "hed", "finetune"), ("pathology", "none", "finetune"), ("pathology", "hed", "finetune"),
                     ("imagenet", "none", "linear"), ("pathology", "none", "linear")]:
        c = runs[(runs.arch == arch) & (runs.weights == w) & (runs.aug == a) & (runs["mode"] == mo)].set_index("seed")
        common = sorted(set(ref.index) & set(c.index))
        if len(common) < 2:
            continue
        for q in RULES:
            d = (c.loc[common, "test_" + q] - ref.loc[common, "test_" + q]).values
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            u = stats.mannwhitneyu(c.loc[common, "test_" + q], ref.loc[common, "test_" + q], alternative="two-sided")
            gains.append(dict(arch=arch, weights=w, aug=a, mode=mo, rule=q, n=len(common), gain=d.mean(), lo=np.percentile(bs, 2.5),
                              hi=np.percentile(bs, 97.5), p=u.pvalue, planned=mo == "finetune"))
g = pd.DataFrame(gains)
if len(g):
    g["p_holm"] = np.nan
    pl = g.planned
    p = g.loc[pl, "p"].values
    o = np.argsort(p)
    adj, run_max = np.empty_like(p), 0.0
    for i, k in enumerate(o):
        run_max = max(run_max, (len(p) - i) * p[k]); adj[k] = min(1.0, run_max)
    g.loc[pl, "p_holm"] = adj
    piv = g.pivot_table(index=["arch", "weights", "aug", "mode"], columns="rule", values="gain")
    g = g.merge((piv["id"] / piv["ood"]).rename("id_share_of_ood").reset_index(), on=["arch", "weights", "aug", "mode"], how="left")
g.to_csv(os.path.join(S, "gains.csv"), index=False)

spread = []
for arch in ["r50", "vits"]:
    ref = runs[(runs.arch == arch) & (runs.weights == "imagenet") & (runs.aug == "none") & (runs["mode"] == "finetune")].acc_test
    for (w, a, mo), q in runs[runs.arch == arch].groupby(["weights", "aug", "mode"]):
        bf = stats.levene(q.acc_test, ref, center="median").pvalue if len(q) > 2 and len(ref) > 2 and not (w == "imagenet" and a == "none" and mo == "finetune") else np.nan
        spread.append(dict(arch=arch, weights=w, aug=a, mode=mo, n=len(q), sd=q.acc_test.std(), range=q.acc_test.max() - q.acc_test.min(), p_bf=bf))
pd.DataFrame(spread).to_csv(os.path.join(S, "spread.csv"), index=False)

pr = []
for k, q in runs.groupby(KEY):
    if len(q) < 4:
        continue
    x = stats.norm.ppf(q.acc_id.clip(1e-4, 1 - 1e-4)); y = stats.norm.ppf(q.acc_test.clip(1e-4, 1 - 1e-4))
    fit = stats.linregress(x, y) if np.ptp(x) > 0 else None
    pr.append(dict(zip(KEY, k), n=len(q), slope=fit.slope if fit else np.nan, intercept=fit.intercept if fit else np.nan,
                   r2=fit.rvalue ** 2 if fit else np.nan, spearman=stats.spearmanr(q.acc_id, q.acc_test).statistic))
pd.DataFrame(pr).to_csv(os.path.join(S, "probit.csv"), index=False)
print("ANALYZE_DONE")
