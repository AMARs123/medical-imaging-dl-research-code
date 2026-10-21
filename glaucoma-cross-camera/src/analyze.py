"""glaucoma-cross-camera analysis from the stored predictions (no network is run).

Per run and evaluation set: AUC, AP, sensitivity, specificity and balanced accuracy at the run's validation (Youden) threshold,
Brier score and ECE (15 bins), and, on Chaksu images, the error rate by expert minority size m (Eq. minority) and the Spearman
correlation between the binary entropy of the prediction and m. Sets: chaksu_test (all-camera models), held_out_camera
(leave-one-camera-out models), papila (glaucoma vs healthy), rimone.
Camera gap Delta_c (Eq. gap): AUC of the all-camera model on the official test-split images of camera c minus the AUC of the
model that left c out on the same images, paired by target, backbone and seed.
Hard vs soft (Eq. diff): paired by configuration, backbone and seed; mean difference, pairs better, rank-biserial r, Wilcoxon,
Holm over the test sets.
Writes results/processed/runs.csv, results/statistics/{gap.csv, hard_soft.csv, minority.csv, summary.csv}.
Usage: python analyze.py <glaucoma-cross-camera dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

R = sys.argv[1]
P = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
os.makedirs(P, exist_ok=True)
os.makedirs(S, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))


def ece(p, y, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def metrics(y, p, thr):
    out = dict(n=len(y), n_pos=int(y.sum()), brier=float(np.mean((p - y) ** 2)), ece=ece(p, y))
    if 0 < y.sum() < len(y):
        g = p >= thr
        out.update(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), sens=g[y == 1].mean(), spec=(~g)[y == 0].mean())
        out["bacc"] = 0.5 * (out["sens"] + out["spec"])
    return out


rows, preds = [], []
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    if not os.path.exists(os.path.join(d, "results.json")):
        continue
    thr = json.load(open(os.path.join(d, "results.json")))["threshold"]
    p = pd.read_csv(os.path.join(d, "predictions.csv"))
    p["prob"] = 1 / (1 + np.exp(-p.logit.values))
    p["split"] = meta.split.values[p.array_index.values]
    p["patient"] = meta.patient.values[p.array_index.values]
    for k, v in r.items():
        p[k] = v
    preds.append(p)
    # positive weight of Eq. loss exactly as in train.py: training pool of the configuration minus the validation images
    ch = meta[meta.dataset == "Chaksu"]
    pool = ch[ch.split == "train"] if r["config"] == "all" else ch[ch.camera != r["config"].replace("loco_", "")]
    ytr = meta.label.values[np.setdiff1d(pool.index.values, p[p.set == "val"].array_index.values)]
    wp_run = (1 - ytr.mean()) / max(ytr.mean(), 1e-3)
    for s, g in p[p.set != "val"].groupby("set"):
        if s == "papila_suspect":
            rows.append(dict(**r, set=s, n=len(g), w_pos=wp_run, flagged=float((g.prob >= thr).mean()), mean_prob=g.prob.mean()))
            continue
        y = g.label.values.astype(int)
        row = dict(**r, set=s, w_pos=wp_run, **metrics(y, g.prob.values, thr))
        if g.n_suspect.notna().all():
            m = np.minimum(g.n_suspect.values, 5 - g.n_suspect.values)
            err = ((g.prob.values >= thr).astype(int) != y)
            for k in range(3):
                row["err_m%d" % k] = err[m == k].mean() if (m == k).any() else np.nan
                row["n_m%d" % k] = int((m == k).sum())
            pp = np.clip(g.prob.values, 1e-6, 1 - 1e-6)
            H = -pp * np.log(pp) - (1 - pp) * np.log(1 - pp)
            row["rho_H_m"] = spearmanr(H, m).correlation
            if r["target"] == "soft":
                a = g.n_suspect.values / 5.0
                row["mae_pstar"] = np.mean(np.abs(g.prob.values - wp_run * a / (wp_run * a + 1 - a + 1e-12)))
        rows.append(row)
runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(P, "runs.csv"), index=False)
if not len(runs):
    sys.exit(0)
# the size-matched control (config all_n271, added after the main results) is analysed in analyze_extra.py only
runs = runs[runs.config != "all_n271"]
pr = pd.concat(preds)
pr.to_csv(os.path.join(P, "predictions_all.csv.gz"), index=False)

# ---------------------------------------------------------------- camera gap on the same images
gap = []
for cam in ["Remidio", "Bosch", "Forus"]:
    for (tg, m, sd), _ in reg.groupby(["target", "model", "seed"]):
        a = pr[(pr.config == "all") & (pr.target == tg) & (pr.model == m) & (pr.seed == sd) & (pr.set == "chaksu_test") & (pr.camera == cam)]
        b = pr[(pr.config == "loco_" + cam) & (pr.target == tg) & (pr.model == m) & (pr.seed == sd) & (pr.set == "held_out_camera") & (pr.split == "test")]
        if len(a) and len(b) and 0 < a.label.sum() < len(a):
            j = a[["array_index", "label", "prob"]].merge(b[["array_index", "prob"]], on="array_index", suffixes=("_all", "_loco"))
            auc_all, auc_loco = roc_auc_score(j.label, j.prob_all), roc_auc_score(j.label, j.prob_loco)
            gap.append(dict(camera=cam, target=tg, model=m, seed=sd, n=len(j), n_pos=int(j.label.sum()), auc_all=auc_all, auc_loco=auc_loco, gap=auc_all - auc_loco))
pd.DataFrame(gap, columns=["camera", "target", "model", "seed", "n", "n_pos", "auc_all", "auc_loco", "gap"]).to_csv(os.path.join(S, "gap.csv"), index=False)


# ---------------------------------------------------------------- hard vs soft
def rb(d):
    d = d[d != 0]
    if not len(d):
        return 0.0
    rr = rankdata(np.abs(d))
    return (rr[d > 0].sum() - rr[d < 0].sum()) / rr.sum()


hs = []
for s in ["chaksu_test", "held_out_camera", "papila", "rimone"]:
    q = runs[runs.set == s]
    if not len(q):
        continue
    pv = q.pivot_table(index=["config", "model", "seed"], columns="target", values=["auc", "ece"])
    if ("auc", "soft") not in pv or ("auc", "hard") not in pv:
        continue
    for mt in ["auc", "ece"]:
        d = (pv[(mt, "soft")] - pv[(mt, "hard")]).dropna().values
        if len(d) < 3:
            continue
        hs.append(dict(set=s, metric=mt, n_pairs=len(d), mean_diff=d.mean(), soft_better=int((d > 0).sum() if mt == "auc" else (d < 0).sum()),
                       rb=rb(d), p=wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
hs = pd.DataFrame(hs, columns=["set", "metric", "n_pairs", "mean_diff", "soft_better", "rb", "p"])
for mt in ["auc", "ece"]:
    q = hs[hs.metric == mt].sort_values("p")
    k = len(q)
    adj, run = [], 0.0
    for i, pv_ in enumerate(q.p.values):
        run = max(run, min(1.0, (k - i) * pv_))
        adj.append(run)
    hs.loc[q.index, "p_holm"] = adj
hs.to_csv(os.path.join(S, "hard_soft.csv"), index=False)

mn = runs[runs.set.isin(["chaksu_test", "held_out_camera"])].groupby(["set", "target"])[[c for c in ["err_m0", "err_m1", "err_m2", "rho_H_m", "mae_pstar"] if c in runs]].mean()
mn.reset_index().to_csv(os.path.join(S, "minority.csv"), index=False)
num = [c for c in ["auc", "ap", "sens", "spec", "bacc", "brier", "ece", "flagged"] if c in runs]
su = runs.groupby(["config", "target", "model", "set"])[num].agg(["mean", "std"])
su.columns = ["%s_%s" % c for c in su.columns]
su.reset_index().to_csv(os.path.join(S, "summary.csv"), index=False)
print(pd.DataFrame(gap).groupby(["camera", "target"]).gap.agg(["mean", "std", "count"]).round(3).to_string() if gap else "no gap yet")
print(hs.round(4).to_string())
print("ANALYZE_DONE")
