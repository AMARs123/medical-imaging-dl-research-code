"""All statistics of Paper 03 from stored outputs (experiments/*/results.json, results/attributions/*).

Usage: python stats.py <Paper_03 dir>
Writes results/statistics/*.csv and summary.json.

Performance : test macro AUC and per-class AUC, mean +- sd over 5 seeds per architecture.
RQ1 stability: Spearman correlation and top-10% overlap (Jaccard) between relevance profiles of two models for the
               same ECG and label. Pair types: same architecture / different seed ("seed"), different architectures
               ("arch"), trained vs randomly initialised model of the same architecture ("random", the floor).
RQ2 localisation: share of relevance in P, QRS, ST-T, other; lift = share / time share of the segment.
RQ3 faithfulness: logit drop after deleting the top-10% time steps minus the drop after deleting 10% random steps.
RQ4: per-item seed stability vs whether the model ensemble detected the label (prob > 0.5).
Confidence intervals: percentile bootstrap over test patients (2000 resamples). Tests: Wilcoxon signed-rank on
per-item values; Holm correction within each family of tests.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R = sys.argv[1]
A = os.path.join(R, "results", "attributions")
S = os.path.join(R, "results", "statistics")
os.makedirs(S, exist_ok=True)
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
METHODS = ["saliency", "gxi", "ig", "occlusion", "cam"]
# methods added in the Research_Expert round (attribute_extra.py), used when their outputs exist
METHODS += [m for m in ["smoothgrad", "gradshap", "gradcam_mid"] if os.path.exists(os.path.join(A, "profiles_%s.npy" % m))]
SEG = ["P", "QRS", "STT", "other"]
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
items = pd.read_csv(os.path.join(A, "items.csv"))
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
pid = meta.set_index("ecg_id").patient_id.loc[items.ecg_id].values
rng = np.random.default_rng(0)
summary = {}


def boot_ci(values, groups, n=2000):
    """Percentile CI of the mean, resampling groups (patients)."""
    df = pd.DataFrame(dict(v=values, g=groups)).dropna()
    gs = df.groupby("g")["v"].agg(["sum", "count"])
    s, c = gs["sum"].values, gs["count"].values
    idx = rng.integers(0, len(gs), (n, len(gs)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return float(df.v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ performance
perf = []
for r in reg.to_dict("records"):
    res = json.load(open(os.path.join(R, "experiments", r["exp_id"], "results.json")))
    perf.append(dict(exp_id=r["exp_id"], model=r["model"], seed=r["seed"], random_init=r["random_init"], macro_auc=res["test"]["macro_auc"],
                     macro_ap=res["test"]["macro_ap"], **{"auc_" + c: res["test"]["auc"][k] for k, c in enumerate(SUPER)},
                     n_params=res["n_params"], best_epoch=res.get("best_epoch"), train_seconds=res.get("train_seconds")))
perf = pd.DataFrame(perf)
perf.to_csv(os.path.join(S, "performance_runs.csv"), index=False)
pt = perf[perf.random_init == 0].groupby("model")[["macro_auc", "macro_ap"] + ["auc_" + c for c in SUPER]].agg(["mean", "std"])
pt.to_csv(os.path.join(S, "performance_by_model.csv"))
summary["performance"] = {m: dict(macro_auc_mean=float(pt.loc[m, ("macro_auc", "mean")]), macro_auc_sd=float(pt.loc[m, ("macro_auc", "std")]))
                          for m in pt.index}

# ------------------------------------------------------------------ pairs of models
idx = {r["exp_id"]: i for i, r in enumerate(reg.to_dict("records"))}
pairs = []
archs = ["xresnet1d50", "inception1d", "convtran1d"]
tr = {a: [idx[e] for e in reg[(reg.model == a) & (reg.random_init == 0)].exp_id] for a in archs}
rnd = {a: idx[reg[(reg.model == a) & (reg.random_init == 1)].exp_id.iloc[0]] for a in archs}
for a in archs:
    for i in range(5):
        for j in range(i + 1, 5):
            pairs.append(("seed", a, a, tr[a][i], tr[a][j]))
        pairs.append(("random", a, a, tr[a][i], rnd[a]))
for x in range(3):
    for y in range(x + 1, 3):
        for i in tr[archs[x]]:
            for j in tr[archs[y]]:
                pairs.append(("arch", archs[x], archs[y], i, j))
pairs = pd.DataFrame(pairs, columns=["type", "a", "b", "i", "j"])

stab_rows, item_seed_stab = [], {}
for mt in METHODS:
    P = np.load(os.path.join(A, "profiles_%s.npy" % mt), mmap_mode="r")
    n = P.shape[0]
    rho = np.zeros((n, len(pairs)), np.float32)
    jac = np.zeros((n, len(pairs)), np.float32)
    for s in range(0, n, 256):
        p = np.asarray(P[s:s + 256], np.float32)
        rk = stats.rankdata(p, axis=2)
        rk = (rk - rk.mean(2, keepdims=True)) / (rk.std(2, keepdims=True) + 1e-9)
        top = np.argsort(-p, axis=2)[:, :, :100]
        tm = np.zeros(p.shape, bool)
        np.put_along_axis(tm, top, True, axis=2)
        for k, pr in enumerate(pairs.itertuples()):
            rho[s:s + 256, k] = (rk[:, pr.i] * rk[:, pr.j]).mean(1)
            inter = (tm[:, pr.i] & tm[:, pr.j]).sum(1)
            jac[s:s + 256, k] = inter / (200 - inter)
        # constant (all-zero) profiles have no ranking: set to NaN
        flat = p.std(2) < 1e-12
        for k, pr in enumerate(pairs.itertuples()):
            bad = flat[:, pr.i] | flat[:, pr.j]
            rho[s:s + 256, k][bad] = np.nan
            jac[s:s + 256, k][bad] = np.nan
    np.save(os.path.join(S, "pair_spearman_%s.npy" % mt), rho)
    pairs["pair"] = np.where(pairs.type == "arch", pairs.a + "|" + pairs.b, pairs.a)
    for (t, a), g in pairs.groupby(["type", "pair"]):
        cols = g.index.values
        v_r = np.nanmean(rho[:, cols], 1)
        v_j = np.nanmean(jac[:, cols], 1)
        if t == "seed":
            item_seed_stab[(mt, a)] = v_r
        for lab in ["all"] + SUPER:
            sel = np.ones(n, bool) if lab == "all" else (items.label_name.values == lab)
            m, lo, hi = boot_ci(v_r[sel], pid[sel])
            mj, loj, hij = boot_ci(v_j[sel], pid[sel])
            stab_rows.append(dict(method=mt, type=t, arch=a, label=lab, n=int(np.isfinite(v_r[sel]).sum()), spearman=m, ci_low=lo, ci_high=hi,
                                  jaccard_top10=mj, jaccard_ci_low=loj, jaccard_ci_high=hij))
    # seed vs random floor, per architecture (paired over items)
stab = pd.DataFrame(stab_rows)
stab.to_csv(os.path.join(S, "rq1_stability.csv"), index=False)
tests = []
for mt in METHODS:
    P_r = np.load(os.path.join(S, "pair_spearman_%s.npy" % mt))
    for a in archs:
        seed_v = np.nanmean(P_r[:, pairs[(pairs.type == "seed") & (pairs.a == a)].index.values], 1)
        rand_v = np.nanmean(P_r[:, pairs[(pairs.type == "random") & (pairs.a == a)].index.values], 1)
        arch_v = np.nanmean(P_r[:, pairs[(pairs.type == "arch") & ((pairs.a == a) | (pairs.b == a))].index.values], 1)
        ok = np.isfinite(seed_v) & np.isfinite(rand_v) & np.isfinite(arch_v)
        tests.append(dict(method=mt, arch=a, comparison="seed_vs_random", n=int(ok.sum()), mean_diff=float(np.mean(seed_v[ok] - rand_v[ok])),
                          p=float(stats.wilcoxon(seed_v[ok], rand_v[ok]).pvalue) if ok.sum() > 10 else np.nan))
        tests.append(dict(method=mt, arch=a, comparison="seed_vs_crossarch", n=int(ok.sum()), mean_diff=float(np.mean(seed_v[ok] - arch_v[ok])),
                          p=float(stats.wilcoxon(seed_v[ok], arch_v[ok]).pvalue) if ok.sum() > 10 else np.nan))
tests = pd.DataFrame(tests)
tests["p_holm"] = holm(tests.p.fillna(1).values)
tests.to_csv(os.path.join(S, "rq1_tests.csv"), index=False)
summary["rq1_tests"] = tests.to_dict("records")

# ------------------------------------------------------------------ localisation
LOC_METHODS = METHODS + (["amplitude"] if os.path.exists(os.path.join(A, "local_amplitude.csv")) else [])
loc_rows, lift_tests, learned = [], [], []
for mt in LOC_METHODS:
    L = pd.read_csv(os.path.join(A, "local_%s.csv" % mt))
    L["exp_id"] = reg.exp_id.values[L.model_idx]
    L["arch"] = reg.model.values[L.model_idx]
    L["random_init"] = reg.random_init.values[L.model_idx]
    L["label_name"] = items.label_name.values[L.item]
    L["pid"] = pid[L.item]
    L = L[L.fiducials_ok & ~L.zero_map]
    for sg in SEG:
        L["lift_" + sg] = L["share_" + sg] / L["time_" + sg].clip(lower=1e-6)
    # average the five seeds per item first (the item is the unit of analysis)
    g = L.groupby(["item", "arch", "random_init", "label_name", "pid"])[["share_" + s for s in SEG] + ["time_" + s for s in SEG]].mean().reset_index()
    for sg in SEG:
        g["lift_" + sg] = g["share_" + sg] / g["time_" + sg].clip(lower=1e-6)
    for (a, ri, lab), h in g.groupby(["arch", "random_init", "label_name"]):
        row = dict(method=mt, arch=a, random_init=int(ri), label=lab, n=len(h))
        for sg in SEG:
            m, lo, hi = boot_ci(h["share_" + sg].values, h.pid.values)
            row.update({"share_" + sg: m, "share_%s_lo" % sg: lo, "share_%s_hi" % sg: hi, "time_" + sg: h["time_" + sg].mean(),
                        "lift_" + sg: float(np.exp(np.mean(np.log(h["lift_" + sg].clip(lower=1e-6)))))})
            if ri == 0 and sg != "other":
                tr_ = h["share_" + sg] - h["time_" + sg]
                lift_tests.append(dict(method=mt, arch=a, label=lab, segment=sg, n=len(h), mean_excess=float(tr_.mean()),
                                       p=float(stats.wilcoxon(tr_).pvalue) if len(h) > 10 else np.nan))
        loc_rows.append(row)
    # learned excess: share of the trained models (seed mean) minus share of the untrained model, same item and architecture
    tr_ = g[g.random_init == 0].set_index(["item", "arch"]); ra_ = g[g.random_init == 1].set_index(["item", "arch"])
    j = tr_.join(ra_[["share_" + s_ for s_ in SEG]], rsuffix="_rand", how="inner")
    for (a, lab), h in j.reset_index().groupby(["arch", "label_name"]):
        for sg in SEG:
            dlt = h["share_" + sg] - h["share_%s_rand" % sg]
            learned.append(dict(method=mt, arch=a, label=lab, segment=sg, n=len(h), trained=float(h["share_" + sg].mean()),
                                untrained=float(h["share_%s_rand" % sg].mean()), learned_excess=float(dlt.mean()),
                                p=float(stats.wilcoxon(dlt).pvalue) if len(h) > 10 and np.any(dlt != 0) else np.nan))
le = pd.DataFrame(learned)
le["p_holm"] = holm(le.p.fillna(1).values)
le.to_csv(os.path.join(S, "rq2_learned_excess.csv"), index=False)
loc = pd.DataFrame(loc_rows)
loc.to_csv(os.path.join(S, "rq2_localisation.csv"), index=False)
lt = pd.DataFrame(lift_tests)
lt["p_holm"] = holm(lt.p.fillna(1).values)
lt.to_csv(os.path.join(S, "rq2_excess_tests.csv"), index=False)

# ------------------------------------------------------------------ faithfulness
fa = []
for mt in LOC_METHODS:
    D = pd.read_csv(os.path.join(A, "deletion_%s.csv" % mt))
    Lz = pd.read_csv(os.path.join(A, "local_%s.csv" % mt), usecols=["item", "model_idx", "zero_map"])
    D = D.merge(Lz, on=["item", "model_idx"])
    zero_rate = D.groupby("model_idx").zero_map.mean()
    D = D[~D.zero_map]  # an all-zero profile has no ranking; its 'top 10%' would be arbitrary
    D["arch"] = reg.model.values[D.model_idx]
    D["random_init"] = reg.random_init.values[D.model_idx]
    D["gain"] = D.drop_top10 - D.drop_rand10
    D["pid"] = pid[D.item]
    for (a, ri), h in D.groupby(["arch", "random_init"]):
        hh = h.groupby(["item", "pid"]).gain.mean().reset_index()
        m, lo, hi = boot_ci(hh.gain.values, hh.pid.values)
        fa.append(dict(method=mt, arch=a, random_init=int(ri), n=len(hh), zero_map_rate=float(zero_rate[h.model_idx.unique()].mean()), gain=m, ci_low=lo, ci_high=hi,
                       drop_top10=float(h.drop_top10.mean()), drop_rand10=float(h.drop_rand10.mean()),
                       share_positive=float((hh.gain > 0).mean()),
                       p=float(stats.wilcoxon(hh.gain).pvalue) if len(hh) > 10 else np.nan))
fa = pd.DataFrame(fa)
fa["p_holm"] = holm(fa.p.fillna(1).values)
fa.to_csv(os.path.join(S, "rq3_faithfulness.csv"), index=False)

# ------------------------------------------------------------------ RQ4 stability vs detection
r4 = []
for (mt, a), v in item_seed_stab.items():
    cols = ["prob_" + e for e in reg[(reg.model == a) & (reg.random_init == 0)].exp_id]
    p = items[cols].mean(1).values
    det = p > 0.5
    ok = np.isfinite(v)
    r4.append(dict(method=mt, arch=a, n_detected=int((det & ok).sum()), n_missed=int((~det & ok).sum()),
                   stab_detected=float(np.mean(v[det & ok])), stab_missed=float(np.mean(v[~det & ok])),
                   p_mwu=float(stats.mannwhitneyu(v[det & ok], v[~det & ok]).pvalue) if (~det & ok).sum() > 5 else np.nan,
                   spearman_prob=float(stats.spearmanr(v[ok], p[ok]).statistic)))
r4 = pd.DataFrame(r4)
r4["p_holm"] = holm(r4.p_mwu.fillna(1).values)
r4.to_csv(os.path.join(S, "rq4_stability_vs_detection.csv"), index=False)
json.dump(summary, open(os.path.join(S, "summary.json"), "w"), indent=1, default=float)
print("STATS_DONE")
