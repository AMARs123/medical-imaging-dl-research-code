"""Paper 05 evaluation from the stored raw outputs (no network is run).

For every backbone, formulation and data set (val = DDR validation, test = DDR test, ext = DeepDRiD) and every uncertainty source:
  single   identity view of one run (one row per seed)
  tta      mean of the four views of one run (Eq. views; one row per seed)
  ens      mean of the identity views of the three seeds (Eq. ens; one row)
the grade is decoded as in common.decode (CE/SORD: argmax of the mean distribution; CORAL/CORN: cumulative rule on the mean
cumulative probabilities; REG: thresholds tuned on DDR validation for the same source applied to the mean severity), and
QWK, SER, accuracy, MAE, referable sensitivity/specificity, vision-threatening sensitivity, ECE of the top-grade confidence
(not defined for REG), and the abstention measures AUSER and missed-referral AUC with the ordinal variance U (Eq. ovar) and,
as a reference, the entropy H. For REG, U is the variance of the severity over views or seeds (undefined for single).
Writes results/processed/eval.csv and results/processed/perimage_<model>_<form>_<set>.npz (per-image severe-error indicator
and U for single (seed mean), tta (seed mean) and ens) for the bootstrap of stats.py.
Usage: python evaluate.py <Paper_05 dir>
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import K, grading_metrics, tune_thresholds  # noqa: E402

R = sys.argv[1]
OUT = os.path.join(R, "results", "processed")
os.makedirs(OUT, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
SETS = ["val", "test", "ext"]


def probs(form, o):
    """o: [N, n_out] raw outputs -> (p [N, K] for CE/SORD, cum [N, K-1] for CORAL/CORN, s [N] for REG)."""
    if form in ("ce", "sord"):
        e = np.exp(o - o.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    if form == "reg":
        return o[:, 0]
    sig = 1 / (1 + np.exp(-o))
    cum = sig if form == "coral" else np.cumprod(sig, 1)
    return np.minimum.accumulate(cum, axis=1)


def from_cum(cum):
    ext = np.concatenate([np.ones((len(cum), 1)), cum, np.zeros((len(cum), 1))], 1)
    return np.clip(ext[:, :-1] - ext[:, 1:], 0, 1), (cum > 0.5).sum(1)


def ece(p, y, bins=15):
    c, h = p.max(1), p.argmax(1)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (c > a) & (c <= b)
        if m.any():
            e += m.mean() * abs((h[m] == y[m]).mean() - c[m].mean())
    return e


def area(err, u):
    """mean of the error rate among the m least uncertain images, m = 1..N (Eq. auser)."""
    order = np.argsort(u, kind="stable")
    e = err[order].astype(float)
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def combine(form, outs, thr):
    """outs: list of [N, n_out] raw outputs to average -> grade, distribution p (or None), U, H."""
    if form == "reg":
        s = np.stack([o[:, 0] for o in outs])
        g = np.searchsorted(thr, s.mean(0))
        u = s.var(0) if len(outs) > 1 else np.full(s.shape[1], np.nan)
        return g, None, u, np.full(s.shape[1], np.nan)
    if form in ("ce", "sord"):
        p = np.mean([probs(form, o) for o in outs], 0)
        g = p.argmax(1)
    else:
        cum = np.mean([probs(form, o) for o in outs], 0)
        p, g = from_cum(cum)
    mu = p @ np.arange(K)
    u = (p * (np.arange(K)[None] - mu[:, None]) ** 2).sum(1)
    h = -(p * np.log(p + 1e-12)).sum(1)
    return g, p, u, h


rows = []
for (m, form), grp in reg.groupby(["model", "form"]):
    runs = [r for r in grp.sort_values("seed").to_dict("records") if os.path.exists(os.path.join(R, "experiments", r["exp_id"], "results.json"))]
    if not runs:
        continue
    O = {r["exp_id"]: {s: np.load(os.path.join(R, "experiments", r["exp_id"], "outputs_%s.npy" % s)) for s in SETS} for r in runs}
    Y = {s: np.load(os.path.join(R, "experiments", runs[0]["exp_id"], "labels_%s.npy" % s)) for s in SETS}
    per = {s: {} for s in SETS}
    sources = [("single", r["exp_id"], [(r["exp_id"], [0])]) for r in runs] + [("tta", r["exp_id"], [(r["exp_id"], [0, 1, 2, 3])]) for r in runs]
    if len(runs) == 3:
        sources.append(("ens", "ens", [(r["exp_id"], [0]) for r in runs]))
    for src, eid, parts in sources:
        thr = None
        if form == "reg":
            gv_outs = [O[e]["val"][:, v] for e, vs in parts for v in vs]
            thr = tune_thresholds(np.mean([o[:, 0] for o in gv_outs], 0), Y["val"])
        for s in SETS:
            outs = [O[e][s][:, v] for e, vs in parts for v in vs]
            g, p, u, h = combine(form, outs, thr)
            y = Y[s]
            met = grading_metrics(y, g)
            met.pop("confusion")
            sev = np.abs(g - y) >= 2
            miss = (y >= 2) & (g < 2)
            row = dict(model=m, form=form, source=src, run=eid, seed=int([r["seed"] for r in runs if r["exp_id"] == eid][0]) if eid != "ens" else -1,
                       set=s, **met, ece=ece(p, y) if p is not None else np.nan,
                       auser=area(sev, u) if np.isfinite(u).all() else np.nan, auser_entropy=area(sev, h) if np.isfinite(h).all() else np.nan,
                       amiss=area(miss, u) if np.isfinite(u).all() else np.nan)
            rows.append(row)
            per[s].setdefault(src, []).append((sev, u, miss))
    for s in SETS:
        np.savez_compressed(os.path.join(OUT, "perimage_%s_%s_%s.npz" % (m.split(".")[0], form, s)),
                            **{"%s_sev" % src: np.mean([x[0] for x in v], 0) for src, v in per[s].items()},
                            **{"%s_u" % src: np.mean([x[1] for x in v], 0) for src, v in per[s].items()},
                            **{"%s_miss" % src: np.mean([x[2] for x in v], 0) for src, v in per[s].items()}, y=Y[s])
    print(m, form, "runs", len(runs), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "eval.csv"), index=False)
print("EVALUATE_DONE", len(rows))
