"""Paper 05 statistics from results/processed/eval.csv and the per-image files of evaluate.py.

Writes results/statistics/
  summary.csv      mean and SD over runs for every backbone x formulation x source x set (single and tta: 3 seeds each)
  pooled.csv       the same, pooled over the two backbones (6 runs for single/tta, 2 for ens)
  paired_ce.csv    single-model formulations against CE on paired runs (same backbone and seed), test and ext:
                   mean difference of SER and QWK, pairs in which the formulation is better, rank-biserial r
  boot_ser.csv     paired bootstrap (2000 resamples; DDR: images, DeepDRiD: patients) of the SER difference formulation - CE, per backbone, using
                   the per-image severe-error indicator averaged over seeds (single source), test and ext
  sources.csv      for every formulation and set: AUSER of tta and ens against single (CE/SORD/CORAL/CORN) and U against entropy
Usage: python stats.py <Paper_05 dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = sys.argv[1]
P = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
os.makedirs(S, exist_ok=True)
ev = pd.read_csv(os.path.join(P, "eval.csv"))
M = ["qwk", "severe_error_rate", "accuracy", "mae", "referable_sens", "referable_spec", "vt_sens", "ece", "auser", "auser_entropy", "amiss"]

summ = ev.groupby(["model", "form", "source", "set"])[M].agg(["mean", "std"])
summ.columns = ["%s_%s" % c for c in summ.columns]
summ.reset_index().to_csv(os.path.join(S, "summary.csv"), index=False)
pool = ev.groupby(["form", "source", "set"])[M].agg(["mean", "std", "count"])
pool.columns = ["%s_%s" % c for c in pool.columns]
pool.reset_index().to_csv(os.path.join(S, "pooled.csv"), index=False)


def rb(d):
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = rankdata(np.abs(d))
    tp, tm = r[d > 0].sum(), r[d < 0].sum()
    return (tp - tm) / (tp + tm)


single = ev[ev.source == "single"].set_index(["model", "seed", "set", "form"])
rows = []
for s in ["test", "ext"]:
    for f in sorted(set(ev.form) - {"ce"}):
        a = single.xs((s, f), level=("set", "form"))
        c = single.xs((s, "ce"), level=("set", "form"))
        j = a.join(c, lsuffix="_f", rsuffix="_ce", how="inner")
        if not len(j):
            continue
        dser = (j.severe_error_rate_f - j.severe_error_rate_ce).values
        dq = (j.qwk_f - j.qwk_ce).values
        rows.append(dict(set=s, form=f, n_pairs=len(j), d_ser=dser.mean(), ser_better=int((dser < 0).sum()), rb_ser=-rb(dser),
                         d_qwk=dq.mean(), qwk_better=int((dq > 0).sum()), rb_qwk=rb(dq)))
pd.DataFrame(rows, columns=["set", "form", "n_pairs", "d_ser", "ser_better", "rb_ser", "d_qwk", "qwk_better", "rb_qwk"]).to_csv(os.path.join(S, "paired_ce.csv"), index=False)

rng = np.random.default_rng(0)
B = 2000
boot = []
# DeepDRiD has four images per patient, so its bootstrap resamples patients (DDR releases no patient identifiers: images)
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"), keep_default_na=False)
first = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv")).exp_id.iloc[0]
pid_ext = meta.patient.values[np.load(os.path.join(R, "experiments", first, "index_ext.npy"))]
groups_ext = [np.flatnonzero(pid_ext == p) for p in np.unique(pid_ext)]


def resample(n, s):
    if s != "ext":
        return rng.integers(0, n, n)
    return np.concatenate([groups_ext[k] for k in rng.integers(0, len(groups_ext), len(groups_ext))])


for s in ["test", "ext"]:
    for m in sorted(set(ev.model)):
        mm = m.split(".")[0]
        fce = os.path.join(P, "perimage_%s_ce_%s.npz" % (mm, s))
        if not os.path.exists(fce):
            continue
        ce = np.load(fce)["single_sev"]
        for f in sorted(set(ev.form) - {"ce"}):
            ff = os.path.join(P, "perimage_%s_%s_%s.npz" % (mm, f, s))
            if not os.path.exists(ff):
                continue
            d = np.load(ff)["single_sev"] - ce
            bs = np.array([d[resample(len(d), s)].mean() for _ in range(B)])
            boot.append(dict(set=s, model=m, form=f, d_ser=d.mean(), ci_low=np.quantile(bs, 0.025), ci_high=np.quantile(bs, 0.975)))
pd.DataFrame(boot, columns=["set", "model", "form", "d_ser", "ci_low", "ci_high"]).to_csv(os.path.join(S, "boot_ser.csv"), index=False)

src = []
pm = ev.groupby(["model", "form", "source", "set"])[["auser", "auser_entropy", "severe_error_rate", "qwk"]].mean().reset_index()
for (m, f, s), g in pm.groupby(["model", "form", "set"]):
    g = g.set_index("source")
    for so in g.index:
        src.append(dict(model=m, form=f, set=s, source=so, auser=g.loc[so, "auser"], auser_entropy=g.loc[so, "auser_entropy"],
                        ser=g.loc[so, "severe_error_rate"], qwk=g.loc[so, "qwk"]))
pd.DataFrame(src).to_csv(os.path.join(S, "sources.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string())
print("STATS_DONE")
