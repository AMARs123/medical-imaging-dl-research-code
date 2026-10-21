"""Paper 04 analysis from the stored predictions and the post-training evaluations (no network is run).
NOTE: group and source names include "NA" (NIH adults); read every output CSV with keep_default_na=False.

Per run and group (G, NA, NP): AUC, AP, sensitivity, specificity, balanced accuracy at the run's validation threshold,
PPV at the group's prevalence (Eq. ppv), Brier score and ECE; for a training group the values come from the run's held-out
test fold, otherwise from all images of the group. AdaBN (adabn.json) and corruption results (robust.json) are added when present.
Statistics (results/statistics/):
  gaps.csv        transfer gap Delta_{s->t} = AUC_s(s) - AUC_s(t) (Eq. gap) per run, summarised per backbone, source and seed set
  transfers.csv   the three single- and double-change transfers (Eq. transfers) pooled over backbones, folds and seeds
  contrast.csv    source contrast delta_NP = AUC_NA(NP) - AUC_G(NP) (Eq. contrast) and pooled-vs-single transfer to NP, paired by
                  backbone, fold and seed; Wilcoxon, rank-biserial
  remedies.csv    strong augmentation, CLAHE (phase B vs phase A seed 0) and AdaBN (phase A, BatchNorm backbones) against the base
                  condition on every target group, paired by backbone, source and fold (Eq. diff); Wilcoxon, Holm within each family
  robust.csv      relative AUC per corruption (Eq. rauc), backbone, source and group (phase A)
  ci.csv          patient-bootstrap 95% CI of the mean AUC over the runs of a condition on each external group (Eq. boot)
Usage: python analyze.py <Paper_04 dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

R = sys.argv[1]
P = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
os.makedirs(P, exist_ok=True)
os.makedirs(S, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"), keep_default_na=False)
G3 = ["G", "NA", "NP"]


def ece(p, y, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def metrics(y, p, thr):
    out = dict(n=len(y), n_pos=int(y.sum()), prev=float(y.mean()), brier=float(np.mean((p - y) ** 2)), ece=ece(p, y))
    if 0 < y.sum() < len(y):
        g = p >= thr
        out.update(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), sens=g[y == 1].mean(), spec=(~g)[y == 0].mean())
        out["bacc"] = 0.5 * (out["sens"] + out["spec"])
        pi = out["prev"]
        den = out["sens"] * pi + (1 - out["spec"]) * (1 - pi)
        out["ppv"] = out["sens"] * pi / den if den > 0 else np.nan
    return out


rows, preds = [], []
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    if not os.path.exists(os.path.join(d, "results.json")):
        continue
    res = json.load(open(os.path.join(d, "results.json")))
    thr = res["threshold"]
    p = pd.read_csv(os.path.join(d, "predictions.csv"), keep_default_na=False, na_values=[""])
    p["prob"] = 1 / (1 + np.exp(-p.logit.values))
    for k in ["exp_id", "phase", "model", "train_groups", "fold", "seed", "aug", "prep"]:
        p[k] = r[k]
    preds.append(p[p.part != "val"])
    tg = r["train_groups"].split(",")
    ad = json.load(open(os.path.join(d, "adabn.json"))) if os.path.exists(os.path.join(d, "adabn.json")) else {}
    rb_ = json.load(open(os.path.join(d, "robust.json"))) if os.path.exists(os.path.join(d, "robust.json")) else {}
    for g in G3:
        q = p[(p.group == g) & (p.part != "val")]
        if not len(q):
            continue
        row = dict(r, group=g, in_train=g in tg, **metrics(q.label.values.astype(float), q.prob.values, thr))
        if g in ad:
            row["auc_adabn"] = ad[g]["auc"]
            row["sens_adabn"], row["spec_adabn"] = ad[g]["sens"], ad[g]["spec"]
        if g in rb_:
            clean = rb_[g]["clean"]["auc"]
            for kind, lst in rb_[g].items():
                if kind != "clean":
                    row["rauc_" + kind] = np.mean([(m["auc"] - 0.5) / (clean - 0.5) for m in lst])
        rows.append(row)
runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(P, "runs.csv"), index=False)
print("runs with results:", runs.exp_id.nunique() if len(runs) else 0, "of", len(reg))
if not len(runs):
    sys.exit(0)
pr = pd.concat(preds)


def rb(d):
    d = d[d != 0]
    if not len(d):
        return 0.0
    rr = rankdata(np.abs(d))
    return (rr[d > 0].sum() - rr[d < 0].sum()) / rr.sum()


def test(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return dict(n_pairs=len(d), mean_diff=d.mean() if len(d) else np.nan, better=int((d > 0).sum()), rb=np.nan, p=np.nan)
    return dict(n_pairs=len(d), mean_diff=d.mean(), sd_diff=d.std(ddof=1), better=int((d > 0).sum()), rb=rb(d),
                p=wilcoxon(d).pvalue if np.any(d != 0) else 1.0)


def holm(df, fam):
    for _, q in df.groupby(fam):
        q = q[q.p.notna()].sort_values("p")
        run = 0.0
        for i, (ix, pv) in enumerate(q.p.items()):
            run = max(run, min(1.0, (len(q) - i) * pv))
            df.loc[ix, "p_holm"] = run
    return df


# ---------------------------------------------------------------- transfer gaps (phases A and C: base condition)
base = runs[(runs.aug == "base") & (runs.prep == "raw")]
gap = []
for (m, tg_, f, sd), g in base.groupby(["model", "train_groups", "fold", "seed"]):
    src = tg_.split(",")
    a_in = g[g.in_train]
    if len(src) == 1 and len(a_in):
        for _, t in g[~g.in_train].iterrows():
            gap.append(dict(model=m, source=tg_, target=t.group, fold=f, seed=sd, auc_source=a_in.auc.iloc[0], auc_target=t.auc,
                            gap=a_in.auc.iloc[0] - t.auc))
gap = pd.DataFrame(gap)
gap.to_csv(os.path.join(S, "gaps.csv"), index=False)
if len(gap):
    gs = gap.groupby(["source", "target", "model"])[["auc_source", "auc_target", "gap"]].agg(["mean", "std", "count"])
    gs.columns = ["%s_%s" % c for c in gs.columns]
    gs.reset_index().to_csv(os.path.join(S, "gaps_summary.csv"), index=False)
    TR = {("G", "NP"): "hospital only", ("NA", "NP"): "age only", ("G", "NA"): "hospital and age", ("NA", "G"): "hospital and age (reverse)"}
    tt = []
    for (s, t), q in gap.groupby(["source", "target"]):
        if (s, t) in TR:
            tt.append(dict(source=s, target=t, change=TR[(s, t)], n_runs=len(q), auc_source=q.auc_source.mean(), auc_target=q.auc_target.mean(),
                           gap_mean=q.gap.mean(), gap_sd=q.gap.std(), gap_min=q.gap.min(), gap_max=q.gap.max(), **{"p_vs_zero": test(q.gap.values)["p"]}))
    pd.DataFrame(tt).to_csv(os.path.join(S, "transfers.csv"), index=False)

# ---------------------------------------------------------------- source contrast and pooling on NP
np_ = base[base.group == "NP"].pivot_table(index=["model", "fold", "seed"], columns="train_groups", values="auc")
con = []
for a, b, name in [("NA", "G", "delta_NP: NA minus G"), ("G,NA", "G", "pooled minus G"), ("G,NA", "NA", "pooled minus NA")]:
    if a in np_ and b in np_:
        d = (np_[a] - np_[b]).dropna().values
        con.append(dict(comparison=name, target="NP", **test(d)))
for tgt, single in [("G", "G"), ("NA", "NA")]:
    q = base[base.group == tgt].pivot_table(index=["model", "fold", "seed"], columns="train_groups", values="auc")
    if "G,NA" in q and single in q:
        con.append(dict(comparison="pooled minus %s (in-domain %s)" % (single, tgt), target=tgt, **test((q["G,NA"] - q[single]).dropna().values)))
con = pd.DataFrame(con)
if len(con):
    con["family"] = "contrast"
    con = holm(con, "family")
con.to_csv(os.path.join(S, "contrast.csv"), index=False)

# ---------------------------------------------------------------- remedies against the base condition
A0 = runs[(runs.phase == "A") & (runs.seed == 0)]
B = runs[runs.phase == "B"]
rem = []
for name, cond in [("strong augmentation", (B.aug == "strong") & (B.prep == "raw")), ("CLAHE", (B.aug == "base") & (B.prep == "clahe"))]:
    q = B[cond]
    for (src, g), h in q.groupby(["train_groups", "group"]):
        j = h.merge(A0[(A0.train_groups == src) & (A0.group == g)], on=["model", "fold"], suffixes=("", "_base"))
        rem.append(dict(remedy=name, source=src, target=g, in_train=bool(h.in_train.iloc[0]), auc=j.auc.mean(), auc_base=j.auc_base.mean(),
                        **test((j.auc - j.auc_base).values)))
if "auc_adabn" in A0:
    for (src, g), h in A0[A0.auc_adabn.notna()].groupby(["train_groups", "group"]):
        rem.append(dict(remedy="AdaBN", source=src, target=g, in_train=False, auc=h.auc_adabn.mean(), auc_base=h.auc.mean(),
                        **test((h.auc_adabn - h.auc).values)))
rem = pd.DataFrame(rem)
if len(rem):
    rem = holm(rem, "remedy")
rem.to_csv(os.path.join(S, "remedies.csv"), index=False)

# ---------------------------------------------------------------- corruption robustness
rc = [c for c in runs.columns if c.startswith("rauc_")]
if rc:
    rob = runs[runs.phase == "A"].groupby(["model", "train_groups", "group"])[rc].mean().reset_index()
    rob.to_csv(os.path.join(S, "robust.csv"), index=False)

# ---------------------------------------------------------------- patient bootstrap of the mean AUC on external groups
def fast_auc(y, s):
    """AUC (Eq. auc) with mid-ranks for every row of s (runs x images)."""
    r = np.apply_along_axis(rankdata, -1, np.atleast_2d(s))
    n1 = y.sum()
    return (r[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


rng = np.random.default_rng(0)
ci = []
for (ph, m, src, aug, prep), q in pr[pr.part == "test_out"].groupby(["phase", "model", "train_groups", "aug", "prep"]):
    for g, h in q.groupby("group"):
        piv = h.pivot_table(index="array_index", columns="exp_id", values="logit")
        lab = h.drop_duplicates("array_index").set_index("array_index").reindex(piv.index)
        y, pat, Z = lab.label.values.astype(int), lab.patient.values, piv.values.T
        if not 0 < y.sum() < len(y):
            continue
        aucs0 = fast_auc(y, Z).mean()
        up = np.unique(pat)
        pos = pd.Series(np.arange(len(pat))).groupby(pat).apply(np.array)
        bs = []
        for _ in range(2000):
            i = np.concatenate(pos.loc[rng.choice(up, len(up))].values)
            if 0 < y[i].sum() < len(i):
                bs.append(fast_auc(y[i], Z[:, i]).mean())
        ci.append(dict(phase=ph, model=m, source=src, aug=aug, prep=prep, target=g, n_runs=Z.shape[0], n=len(y), n_pos=int(y.sum()),
                       n_patients=len(up), auc_mean=aucs0, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975)))
pd.DataFrame(ci).to_csv(os.path.join(S, "ci.csv"), index=False)

su = runs.groupby(["phase", "model", "train_groups", "aug", "prep", "seed", "group"])[[c for c in ["auc", "ap", "sens", "spec", "bacc", "ppv", "brier", "ece", "auc_adabn"] if c in runs]].agg(["mean", "std"])
su.columns = ["%s_%s" % c for c in su.columns]
su.reset_index().to_csv(os.path.join(S, "summary.csv"), index=False)
if len(gap):
    print(pd.read_csv(os.path.join(S, "transfers.csv"), keep_default_na=False).round(3).to_string())
print(con.round(4).to_string() if len(con) else "no contrast yet")
print(rem.round(4).to_string() if len(rem) else "no remedies yet")
print("ANALYZE_DONE")
