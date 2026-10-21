"""segmentation-failure-detection statistics, from results/processed (written by analyze.py). Usage: python stats.py <segmentation-failure-detection dir>

Outputs in results/statistics/:
  failure_summary.csv  per (data, set, score): mean and SD over runs of AURC, e-AURC, AUROC(failure), Spearman; ensemble rows use
                       a 95% bootstrap interval over images instead of an SD (one ensemble per condition)
  paired_tests.csv     9 planned paired comparisons of AURC between single-model scores; unit = run (20 = 2 data sets x 2 encoders
                       x 5 seeds), value = mean AURC over the run's evaluation sets; Wilcoxon signed-rank, Holm over the 9,
                       rank-biserial effect size
  paired_by_set.csv    the same differences per data set and evaluation set, descriptive only (runs in which a beats b)
  decomposition.csv    per (data, set, encoder, seed): aggregation effect AURC(ent_det) - AURC(ent_det_fg) and sampling effects
                       AURC(ent_det_fg) - AURC(unc_pdice_mc / unc_pdice_tta), and the same for the ensemble
  ens_vs_single.csv    paired bootstrap over images: AURC(unc_pdice_ens) - AURC(ent_det_fg of each seed) per condition and set
  ranking.csv          Kendall tau between the rankings of the 10 single-model scores (by mean AURC) for every pair of settings
  threshold.csv        AUROC for failure at Dice thresholds 0.5, 0.7, 0.8 (sensitivity of the failure definition)
  summary.json         headline values used by make_numbers.py
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R = sys.argv[1]
O = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
os.makedirs(S, exist_ok=True)
f = pd.read_csv(os.path.join(O, "failure.csv"))
img = pd.read_csv(os.path.join(O, "image_scores.csv"), low_memory=False)
f = f[f.set != "val"]
img = img[img.set != "val"]
rng = np.random.default_rng(0)
B = 2000
SINGLE = ["ent_det", "ent_det_fg", "ent_det_fg_e", "ent_mc", "mi_mc", "unc_pdice_mc", "ent_tta", "unc_pdice_tta", "ref_area", "ref_parts", "ref_irreg"]
JUDGE = {"ent_det": "dice_det", "ent_det_fg": "dice_det", "ent_det_fg_e": "dice_det", "ref_area": "dice_det", "ref_parts": "dice_det", "ref_irreg": "dice_det",
         "ent_mc": "dice_mc", "mi_mc": "dice_mc", "unc_pdice_mc": "dice_mc", "ent_tta": "dice_tta", "unc_pdice_tta": "dice_tta"}
summary = {}


def aurc(risk, score):
    o = np.argsort(score, kind="stable")
    r = risk[o]
    return float(np.mean(np.cumsum(r) / np.arange(1, len(r) + 1)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


# ---------------------------------------------------------------- summary per score
rows = []
for (data, st, sc), g in f.groupby(["data", "set", "score"]):
    single = g[g.member != "ens"]
    ens = g[g.member == "ens"]
    for kind, h in [("single", single), ("ens", ens)]:
        if len(h) == 0:
            continue
        rows.append(dict(data=data, set=st, score=sc, kind=kind, runs=len(h), aurc=h.aurc.mean(), aurc_sd=h.aurc.std(ddof=1) if len(h) > 1 else np.nan,
                         eaurc=h.eaurc.mean(), auroc_fail=h.auroc_fail.mean(), auroc_sd=h.auroc_fail.std(ddof=1) if len(h) > 1 else np.nan,
                         spearman=h.spearman.mean(), failures=h.failures.mean(), n=int(h.n.iloc[0]), mean_dice=h.mean_dice.mean()))
fs = pd.DataFrame(rows)
# bootstrap intervals over images for the ensemble scores (AURC), per condition, then averaged over the two encoders
ens_ci = []
for (data, enc, st), d in img[img.member == "ens"].groupby(["data", "encoder", "set"]):
    risk = 1 - d.dice_ens.values
    for sc in ["ens_ent", "ens_ent_fg", "unc_pdice_ens"]:
        s = d[sc].values
        bs = [aurc(risk[i], s[i]) for i in (rng.integers(0, len(risk), len(risk)) for _ in range(B))]
        ens_ci.append(dict(data=data, encoder=enc, set=st, score=sc, aurc=aurc(risk, s), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
ens_ci = pd.DataFrame(ens_ci)
ens_ci.to_csv(os.path.join(S, "ensemble_ci.csv"), index=False)
fs.to_csv(os.path.join(S, "failure_summary.csv"), index=False)

# ---------------------------------------------------------------- planned paired comparisons (single models, unit = run)
PAIRS = [("ent_det_fg", "ent_det"), ("ent_tta", "ent_det"), ("ent_mc", "ent_det"), ("mi_mc", "ent_mc"),
         ("unc_pdice_mc", "ent_det_fg"), ("unc_pdice_tta", "ent_det_fg"), ("unc_pdice_tta", "unc_pdice_mc"),
         ("ent_det_fg", "ref_irreg"), ("ent_det_fg", "ref_area")]
sf = f[f.member != "ens"]
# primary (planned): unit = run (2 data sets x 2 encoders x 5 seeds = 20), value = mean AURC over the run's evaluation sets,
# Wilcoxon signed-rank for each of the 9 planned pairs, Holm over the 9
run_mean = sf.groupby(["data", "encoder", "member", "score"]).aurc.mean().unstack("score")
pt = []
for a, b in PAIRS:
    d = (run_mean[a] - run_mean[b]).values  # negative = a has the lower (better) AURC
    pt.append(dict(a=a, b=b, n=len(d), mean_diff=d.mean(), sd_diff=d.std(ddof=1), a_better=int((d < 0).sum()),
                   p=float(stats.wilcoxon(d).pvalue), r_rb=rank_biserial(d)))
pt = pd.DataFrame(pt)
pt["p_holm"] = holm(pt.p.values)
pt.to_csv(os.path.join(S, "paired_tests.csv"), index=False)
# post-hoc comparisons (not in the planned family; reported with unadjusted p and labelled exploratory)
ph = []
for a, b in [("ent_det_fg_e", "ent_det_fg"), ("unc_pdice_mc", "ent_det_fg_e"), ("unc_pdice_tta", "ent_det_fg_e")]:
    d = (run_mean[a] - run_mean[b]).values
    ph.append(dict(a=a, b=b, n=len(d), mean_diff=d.mean(), sd_diff=d.std(ddof=1), a_better=int((d < 0).sum()),
                   p=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else np.nan, r_rb=rank_biserial(d)))
pd.DataFrame(ph).to_csv(os.path.join(S, "posthoc_tests.csv"), index=False)
# descriptive consistency per data set and evaluation set (no tests): in how many of the 10 runs a beats b
cons = []
for (data, st), g in sf.groupby(["data", "set"]):
    piv = g.pivot_table(index=["encoder", "member"], columns="score", values="aurc")
    for a, b in PAIRS:
        d = (piv[a] - piv[b]).dropna().values
        cons.append(dict(data=data, set=st, a=a, b=b, n=len(d), mean_diff=d.mean(), a_better=int((d < 0).sum())))
pd.DataFrame(cons).to_csv(os.path.join(S, "paired_by_set.csv"), index=False)

# ---------------------------------------------------------------- decomposition of the gain: aggregation vs sampling
dec = []
for (data, enc, member, st), g in sf.groupby(["data", "encoder", "member", "set"]):
    a = g.set_index("score").aurc
    dec.append(dict(data=data, encoder=enc, member=member, set=st, aggregation=a["ent_det"] - a["ent_det_fg"],
                    sampling_mc=a["ent_det_fg"] - a["unc_pdice_mc"], sampling_tta=a["ent_det_fg"] - a["unc_pdice_tta"],
                    total_tta=a["ent_det"] - a["unc_pdice_tta"]))
dec = pd.DataFrame(dec)
dec["agg_share_tta"] = dec.aggregation / dec.total_tta
dec.to_csv(os.path.join(S, "decomposition.csv"), index=False)

# ---------------------------------------------------------------- ensemble agreement vs single-pass foreground entropy (paired bootstrap)
evs = []
for (data, enc, st), d in img.groupby(["data", "encoder", "set"]):
    e = d[d.member == "ens"].set_index("array_index")
    risk_e = 1 - e.dice_ens.values
    for s_, ds in d[d.member != "ens"].groupby("member"):
        ds = ds.set_index("array_index").loc[e.index]
        risk_s = 1 - ds.dice_det.values
        obs = aurc(risk_e, e.unc_pdice_ens.values) - aurc(risk_s, ds.ent_det_fg.values)
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(risk_e), len(risk_e))
            bs.append(aurc(risk_e[i], e.unc_pdice_ens.values[i]) - aurc(risk_s[i], ds.ent_det_fg.values[i]))
        evs.append(dict(data=data, encoder=enc, set=st, seed=s_, diff=obs, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
evs = pd.DataFrame(evs)
evs.to_csv(os.path.join(S, "ens_vs_single.csv"), index=False)

# ---------------------------------------------------------------- ranking agreement of the single-model scores across settings
rk = sf.groupby(["data", "encoder", "set", "score"]).aurc.mean().unstack("score")[SINGLE]
tau = []
for a, b in itertools.combinations(rk.index, 2):
    tau.append(dict(a="|".join(a), b="|".join(b), same_data=a[0] == b[0], same_encoder=a[1] == b[1], same_set=a[2] == b[2],
                    tau=float(stats.kendalltau(rk.loc[a], rk.loc[b]).statistic)))
tau = pd.DataFrame(tau)
tau.to_csv(os.path.join(S, "ranking.csv"), index=False)
best = rk.idxmin(axis=1)
best.to_csv(os.path.join(S, "best_score_per_setting.csv"), header=["best_score"])

# ---------------------------------------------------------------- sensitivity to the failure threshold
th = []
for (data, enc, member, st), d in img[img.member != "ens"].groupby(["data", "encoder", "member", "set"]):
    for t in [0.5, 0.7, 0.8]:
        for sc in SINGLE:
            y = (d[JUDGE[sc]].values < t).astype(int)
            if 0 < y.sum() < len(y):
                s = d[sc].values
                au = stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic / (y.sum() * (len(y) - y.sum()))
                th.append(dict(data=data, encoder=enc, member=member, set=st, threshold=t, score=sc, auroc=au, failures=int(y.sum())))
th = pd.DataFrame(th)
th.to_csv(os.path.join(S, "threshold.csv"), index=False)

# ---------------------------------------------------------------- headline values
summary["mean_by_score_kind"] = fs.groupby(["kind", "score"]).aurc.mean().round(4).reset_index().to_dict("records")
summary["agg_share_tta_median"] = float(dec.agg_share_tta.median())
summary["tau_all"] = dict(mean=float(tau.tau.mean()), min=float(tau.tau.min()), max=float(tau.tau.max()), n=len(tau))
summary["best_counts"] = best.value_counts().to_dict()
json.dump(summary, open(os.path.join(S, "summary.json"), "w"), indent=1, default=float)
print(pt.round(4).to_string())
print(dec.groupby(["data", "set"])[["aggregation", "sampling_mc", "sampling_tta", "agg_share_tta"]].median().round(4).to_string())
print(json.dumps(summary["tau_all"]), summary["best_counts"])
print("STATS_DONE")
