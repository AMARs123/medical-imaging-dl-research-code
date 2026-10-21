"""waste-sorting-edge analysis, part 2: wild gap, pooling effect, paired tests and the cost/accuracy trade-off.

Inputs: results/processed/runs.csv (analyze.py), results/efficiency.csv (efficiency.py; optional).
Writes results/statistics/
  summary.csv      mean and SD over seeds of balanced accuracy for every backbone, training source and test set
  gap.csv          wild gap Delta (Eq. gap) per backbone: TrashNet-trained BAcc on TrashNet minus on TACO, mean and SD over seeds
  pool.csv         pooling effect pi_t (Eq. pool) per backbone and test set, paired by seed
  pool_tests.csv   per test set over the 18 (backbone, seed) pairs: mean difference, pairs > 0, rank-biserial r, Wilcoxon p
  efficiency.csv   per backbone (pooled models, mean over seeds): parameters, GMACs, latency fp32/int8, BAcc fp32/int8,
                   accuracy cost of int8, Pareto membership (latency at one thread against TACO balanced accuracy)
Usage: python stats.py <waste-sorting-edge dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
os.makedirs(S, exist_ok=True)
runs = pd.read_csv(os.path.join(R, "results", "processed", "runs.csv"))

summ = runs.groupby(["model", "train_on", "test"]).agg(bacc_mean=("bacc", "mean"), bacc_sd=("bacc", "std"), acc_mean=("accuracy", "mean"),
                                                     f1_mean=("macro_f1", "mean"), ece_mean=("ece", "mean"), n_seeds=("seed", "nunique")).reset_index()
summ.to_csv(os.path.join(S, "summary.csv"), index=False)

piv = runs.pivot_table(index=["model", "seed"], columns=["train_on", "test"], values="bacc")
gap = []
for m, g in piv.groupby(level=0):
    if ("trashnet", "TN") in g and ("trashnet", "TA") in g:
        d = (g[("trashnet", "TN")] - g[("trashnet", "TA")]).dropna()
        rev = (g[("taco", "TA")] - g[("taco", "TN")]).dropna() if ("taco", "TA") in g and ("taco", "TN") in g else pd.Series(dtype=float)
        gap.append(dict(model=m, gap_mean=d.mean(), gap_sd=d.std(), n=len(d), reverse_gap_mean=rev.mean() if len(rev) else np.nan,
                        reverse_gap_sd=rev.std() if len(rev) else np.nan))
pd.DataFrame(gap, columns=["model", "gap_mean", "gap_sd", "n", "reverse_gap_mean", "reverse_gap_sd"]).to_csv(os.path.join(S, "gap.csv"), index=False)

pool, tests = [], []
for t, single in [("TN", "trashnet"), ("TA", "taco")]:
    if ("both", t) not in piv or (single, t) not in piv:
        continue
    d = (piv[("both", t)] - piv[(single, t)]).dropna()
    for (m, sd), v in d.items():
        pool.append(dict(model=m, seed=sd, test=t, pool_effect=v))
    if len(d) >= 6:
        nz = d[d != 0]
        rk = rankdata(np.abs(nz.values))
        tp, tm = rk[nz.values > 0].sum(), rk[nz.values < 0].sum()
        tests.append(dict(test=t, n_pairs=len(d), mean_diff=d.mean(), sd_diff=d.std(), pairs_positive=int((d > 0).sum()),
                          rank_biserial=(tp - tm) / (tp + tm) if tp + tm > 0 else 0.0, p_wilcoxon=wilcoxon(d.values).pvalue))
pd.DataFrame(pool, columns=["model", "seed", "test", "pool_effect"]).to_csv(os.path.join(S, "pool.csv"), index=False)
pd.DataFrame(tests, columns=["test", "n_pairs", "mean_diff", "sd_diff", "pairs_positive", "rank_biserial", "p_wilcoxon"]).to_csv(os.path.join(S, "pool_tests.csv"), index=False)

ef = os.path.join(R, "results", "efficiency.csv")
if os.path.exists(ef):
    e = pd.read_csv(ef)
    num = [c for c in e.columns if c not in ("model", "exp_id", "seed", "int8_status") and pd.api.types.is_numeric_dtype(e[c])]
    em = e.groupby("model")[num].mean().reset_index()
    em["int8_ok"] = e.groupby("model").int8_status.apply(lambda s: (s == "ok").all()).values
    for t, c in [("TN", "trashnet"), ("TA", "taco")]:
        if "int8_bacc_" + c in em:
            em["dq_" + t] = em["fp32_bacc_" + c] - em["int8_bacc_" + c]

    def pareto(lat, acc):
        out = []
        for i in range(len(lat)):
            dom = any((lat[j] <= lat[i]) and (acc[j] >= acc[i]) and ((lat[j] < lat[i]) or (acc[j] > acc[i])) for j in range(len(lat)) if j != i)
            out.append(not dom)
        return out
    em["pareto_fp32"] = pareto(em.fp32_ms_1t.values, em.fp32_bacc_taco.values)
    em.to_csv(os.path.join(S, "efficiency.csv"), index=False)
print(pd.DataFrame(tests).round(4).to_string())
print("STATS_DONE")
