"""Paper 06 analysis: per-run metrics, split inflation, laboratory-to-field gap and condition comparisons.

Reads experiments/experiment_registry.csv, data/index.csv and every finished run's predictions_<set>.csv
(sets: pv_test = PlantVillage test images of the run's split, pd_all = PlantDoc images not used for training,
pd_test = official PlantDoc test split).
Writes results/processed/runs.csv and results/statistics/
  inflation.csv    per backbone: PlantVillage test accuracy under the random and the leaf split, inflation iota
                   (random - leaf, paired by seed), also restricted to the classes with leaf identity
  gap.csv          per backbone and split: laboratory-to-field gap Delta = Acc(PV) - Acc(PD, pd_all)
  conditions.csv   every condition of groups B and C against the reference of the same backbone (leaf split, colour
                   images, base augmentation, k = 0, fine-tuned): mean difference over seeds of PlantDoc accuracy,
                   balanced accuracy and crop retention, and a paired bootstrap 95% interval of the accuracy difference over
                   the PlantDoc images evaluated by every compared run (2000 resamples; seed-mean accuracy per resample)
Usage: python analyze.py <Paper_06 dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

R = sys.argv[1]
OUT_P = os.path.join(R, "results", "processed")
OUT_S = os.path.join(R, "results", "statistics")
os.makedirs(OUT_P, exist_ok=True)
os.makedirs(OUT_S, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"), keep_default_na=False)
idx = pd.read_csv(os.path.join(R, "data", "index.csv"), keep_default_na=False)
cls = idx.drop_duplicates("label").set_index("label").cls.sort_index()
CROP = np.array([c.split("___")[0] for c in cls.values])
K = len(cls)
pv = idx[idx.source == "PV"]
leaf_classes = set(pv[pv.has_leaf_id.astype(str) == "True"].label.unique())


def ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


rows, pdpred = [], {}
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    if not os.path.exists(os.path.join(d, "results.json")):
        continue
    for s in ["pv_test", "pd_all", "pd_test"]:
        p = pd.read_csv(os.path.join(d, "predictions_%s.csv" % s))
        lo = p[["logit_%d" % k for k in range(K)]].values
        pr = np.exp(lo - lo.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        y, h = p.label.values, p.pred.values
        same_crop = CROP[y] == CROP[h]
        row = dict(exp_id=r["exp_id"], phase=r["phase"], model=r["model"], split=r["split"], source=r["source"], aug=r["aug"],
                   fewshot=int(r["fewshot"]), freeze=int(r["freeze"] or 0), seed=int(r["seed"]), set=s, n=len(p), acc=(y == h).mean(),
                   bacc=balanced_accuracy_score(y, h), macro_f1=f1_score(y, h, average="macro"), ece=ece(pr.max(1), (y == h).astype(float)),
                   retention=same_crop.mean(), acc_within_crop=(y == h)[same_crop].mean() if same_crop.any() else np.nan)
        if s == "pv_test":
            m = np.isin(y, list(leaf_classes))
            row["acc_leafclasses"] = (y[m] == h[m]).mean()
        rows.append(row)
        if s == "pd_all":
            pdpred[r["exp_id"]] = pd.Series((y == h).astype(float), index=p.path.values)
runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(OUT_P, "runs.csv"), index=False)
if not len(runs):
    print("runs with results: 0 of", len(reg))
    sys.exit(0)
print("runs with results:", runs.exp_id.nunique(), "of", len(reg))

A = runs[(runs.phase == "A")]
inf, gap = [], []
for m, g in A.groupby("model"):
    pvq = g[g.set == "pv_test"].pivot_table(index="seed", columns="split", values=["acc", "acc_leafclasses"])
    if ("acc", "random") in pvq and ("acc", "leaf") in pvq:
        d = (pvq[("acc", "random")] - pvq[("acc", "leaf")]).dropna()
        dl = (pvq[("acc_leafclasses", "random")] - pvq[("acc_leafclasses", "leaf")]).dropna()
        inf.append(dict(model=m, acc_random=pvq[("acc", "random")].mean(), acc_leaf=pvq[("acc", "leaf")].mean(), iota_mean=d.mean(), iota_sd=d.std(),
                        iota_leafclasses_mean=dl.mean(), iota_leafclasses_sd=dl.std(), n_pairs=len(d)))
    for sp, h in g.groupby("split"):
        q = h.pivot_table(index="seed", columns="set", values="acc")
        if {"pv_test", "pd_all"} <= set(q.columns):
            dd = q.pv_test - q.pd_all
            gap.append(dict(model=m, split=sp, acc_pv=q.pv_test.mean(), acc_pd=q.pd_all.mean(), gap_mean=dd.mean(), gap_sd=dd.std(), n=len(dd)))
pd.DataFrame(inf).to_csv(os.path.join(OUT_S, "inflation.csv"), index=False)
pd.DataFrame(gap).to_csv(os.path.join(OUT_S, "gap.csv"), index=False)

# ---------------------------------------------------------------- conditions against the reference of each backbone
key = ["model", "split", "source", "aug", "fewshot", "freeze"]
pdr = runs[runs.set == "pd_all"]
rng = np.random.default_rng(0)
B = 2000
cond = []
for m in ["convnext_tiny.fb_in1k", "vit_small_patch14_dinov2.lvd142m"]:
    ref = pdr[(pdr.model == m) & (pdr.split == "leaf") & (pdr.source == "color") & (pdr.aug == "base") & (pdr.fewshot == 0) & (pdr.freeze == 0)]
    if not len(ref):
        continue
    for ck, g in pdr[(pdr.model == m) & (pdr.split == "leaf")].groupby(key[1:]):
        if ck == ("leaf", "color", "base", 0, 0):
            continue
        ex_c, ex_r = list(g.exp_id), list(ref.exp_id)
        common = set.intersection(*[set(pdpred[e].index) for e in ex_c + ex_r])
        common = np.array(sorted(common))
        Mc = np.stack([pdpred[e].loc[common].values for e in ex_c])
        Mr = np.stack([pdpred[e].loc[common].values for e in ex_r])
        diff = Mc.mean(0) - Mr.mean(0)
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
        cond.append(dict(model=m, source=ck[1], aug=ck[2], fewshot=ck[3], freeze=ck[4], n_seeds=len(g),
                         acc=g.acc.mean(), acc_sd=g.acc.std(), bacc=g.bacc.mean(), retention=g.retention.mean(), acc_within_crop=g.acc_within_crop.mean(),
                         ref_acc=ref.acc.mean(), ref_bacc=ref.bacc.mean(), ref_retention=ref.retention.mean(),
                         d_acc=g.acc.mean() - ref.acc.mean(), d_bacc=g.bacc.mean() - ref.bacc.mean(), d_retention=g.retention.mean() - ref.retention.mean(),
                         d_acc_common=diff.mean(), ci_low=np.quantile(bs, 0.025), ci_high=np.quantile(bs, 0.975), n_common=len(common)))
pd.DataFrame(cond).to_csv(os.path.join(OUT_S, "conditions.csv"), index=False)
print(pd.DataFrame(inf).round(4).to_string())
print(pd.DataFrame(cond).round(3).to_string())
print("ANALYZE_DONE")
