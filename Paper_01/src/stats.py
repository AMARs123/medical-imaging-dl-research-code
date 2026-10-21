"""Statistical analysis on stored predictions (CPU only). Usage: python stats.py <Paper_01 dir>

Writes to results/statistics/:
  A_pooled.csv      per model x fold rule: macro-F1 / accuracy of the pooled out-of-fold predictions
                    (every image is predicted exactly once per rule) with a 95% CI from a cluster bootstrap
                    over ID roots (2000 resamples).
  A_paired.csv      each rule vs the ID-root rule on the same 3064 images: difference in pooled macro-F1
                    with cluster-bootstrap 95% CI, and an exact McNemar test on per-image correctness.
  B_friedman.csv    Friedman test of macro-F1 across the seven architectures over the 15 (fold, seed) blocks,
                    and Holm-corrected Wilcoxon tests of every architecture against the best one.
  B_table.csv       mean and SD over the 15 runs of each architecture (all scalar test metrics).
  C_table.csv       mean and SD per architecture and saliency weight.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, wilcoxon
from sklearn.metrics import f1_score

R = sys.argv[1]
OUT = os.path.join(R, "results", "statistics")
os.makedirs(OUT, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
runs = pd.read_csv(os.path.join(R, "results", "processed", "all_runs.csv"))
rng = np.random.default_rng(2026)
NB = 2000


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def pooled(model, protocol):
    ids = reg[(reg.phase == "A_protocol") & (reg.model == model) & (reg.protocol == protocol)]["exp_id"]
    df = pd.concat([pd.read_csv(os.path.join(R, "experiments", e, "predictions.csv"))[["array_index", "root", "label", "pred"]] for e in ids])
    assert df["array_index"].is_unique, "an image was predicted twice under one rule"
    return df.sort_values("array_index").reset_index(drop=True)


def boot_groups(roots):
    u = np.unique(roots)
    pos = {g: np.where(roots == g)[0] for g in u}
    for _ in range(NB):
        yield np.concatenate([pos[g] for g in rng.choice(u, len(u), replace=True)])


mf1 = lambda y, p: f1_score(y, p, average="macro")

# ---------------------------------------------------------------- A: fold-formation rules
rows, prow = [], []
for model in sorted(reg.loc[reg.phase == "A_protocol", "model"].unique()):
    P = {pr: pooled(model, pr) for pr in ["slice", "authors", "pid", "root"]}
    ref = P["root"]
    roots = ref["root"].values
    samples = list(boot_groups(roots))
    for pr, d in P.items():
        assert (d["array_index"].values == ref["array_index"].values).all()
        f = mf1(d.label, d.pred)
        bs = [mf1(d.label.values[i], d.pred.values[i]) for i in samples]
        rows.append(dict(model=model, protocol=pr, n_images=len(d), macro_f1=f, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         accuracy=float((d.label == d.pred).mean())))
        if pr == "root":
            continue
        diff = [mf1(d.label.values[i], d.pred.values[i]) - mf1(ref.label.values[i], ref.pred.values[i]) for i in samples]
        a, b = (d.label == d.pred).values, (ref.label == ref.pred).values
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        prow.append(dict(model=model, protocol=pr, reference="root", diff_macro_f1=f - mf1(ref.label, ref.pred),
                         diff_ci_lo=np.percentile(diff, 2.5), diff_ci_hi=np.percentile(diff, 97.5),
                         only_rule_correct=n01, only_root_correct=n10,
                         p_mcnemar_exact=binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "A_pooled.csv"), index=False)
pa = pd.DataFrame(prow)
pa["p_holm"] = holm(pa["p_mcnemar_exact"])
pa.to_csv(os.path.join(OUT, "A_paired.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(pa.round(4).to_string(index=False))

# ---------------------------------------------------------------- B: architectures
B = runs[(runs.protocol == "root") & (runs.lambda_sal == 0) & runs.main]
num = [c for c in B.columns if B[c].dtype != object and c not in ("fold", "seed", "lambda_sal")]
tb = B.groupby("model")[num].agg(["mean", "std", "count"])
tb.columns = ["%s_%s" % c for c in tb.columns]
tb.reset_index().to_csv(os.path.join(OUT, "B_table.csv"), index=False)
wide = B.pivot_table(index=["fold", "seed"], columns="model", values="macro_f1")
fr = friedmanchisquare(*[wide[c].values for c in wide.columns])
best = wide.mean().idxmax()
bt = [dict(model=c, mean=wide[c].mean(), vs=best, p_wilcoxon=wilcoxon(wide[c], wide[best]).pvalue) for c in wide.columns if c != best]
bt = pd.DataFrame(bt)
bt["p_holm"] = holm(bt["p_wilcoxon"])
bt["friedman_chi2"], bt["friedman_p"], bt["n_blocks"] = fr.statistic, fr.pvalue, len(wide)
bt.to_csv(os.path.join(OUT, "B_friedman.csv"), index=False)
print("Friedman chi2 %.2f p %.2g over %d blocks; best %s" % (fr.statistic, fr.pvalue, len(wide), best))
print(bt.round(4).to_string(index=False))

# ---------------------------------------------------------------- C: saliency weight
C = runs[runs.main & (runs.protocol == "root") & runs.model.isin(runs.loc[runs.phase == "C_saliency", "model"].unique())]
tc = C.groupby(["model", "lambda_sal"])[num].agg(["mean", "std", "count"])
tc.columns = ["%s_%s" % c for c in tc.columns]
tc.reset_index().to_csv(os.path.join(OUT, "C_table.csv"), index=False)
# ---------------------------------------------------------------- D: partition variability (pooled macro-F1 per partition)
if "split_seed" in reg.columns:
    d_rows = []
    for (model, pr), g in reg[reg.protocol.isin(["pid", "root"]) & reg.model.isin(["resnet50", "vit_small_patch16_224"])
                              & (reg.lambda_sal == 0) & ((reg.phase == "A_protocol") | (reg.phase == "D_partition"))].groupby(["model", "protocol"]):
        for s, gs in g.groupby(g.split_seed.fillna(0).astype(int)):
            files = [os.path.join(R, "experiments", e, "predictions.csv") for e in gs.exp_id]
            if len(files) != 5 or not all(os.path.exists(f) for f in files):
                continue
            df = pd.concat([pd.read_csv(f) for f in files])
            assert df.array_index.is_unique and len(df) == 3064
            d_rows.append(dict(model=model, protocol=pr, split_seed=s, macro_f1=mf1(df.label, df.pred), accuracy=float((df.label == df.pred).mean())))
    if d_rows:
        D = pd.DataFrame(d_rows)
        D.to_csv(os.path.join(OUT, "D_partition.csv"), index=False)
        print(D.groupby(["model", "protocol"]).macro_f1.agg(["mean", "std", "min", "max", "count"]).round(4).to_string())

# ---------------------------------------------------------------- faithfulness (run-level means from explain.json)
import json

ex = []
for r in reg.itertuples():
    p = os.path.join(R, "experiments", r.exp_id, "explain.json")
    if os.path.exists(p) and r.protocol == "root" and int(getattr(r, "split_seed", 0) or 0) == 0 \
            and int(getattr(r, "dilate", 1)) == 1 and float(getattr(r, "train_frac", 1.0)) == 1.0:
        s = json.load(open(p))
        ps = os.path.join(R, "experiments", r.exp_id, "explain_signed.json")
        s["del_auc_signed"] = json.load(open(ps))["del_auc_signed"] if os.path.exists(ps) else np.nan
        ex.append(dict(exp_id=r.exp_id, model=r.model, lambda_sal=r.lambda_sal, fold=r.fold, seed=r.seed,
                       **{k: v for k, v in s.items() if not isinstance(v, list)}))
if ex:
    E = pd.DataFrame(ex)
    E.to_csv(os.path.join(OUT, "faithfulness_runs.csv"), index=False)
    t = []
    for (m, lam), g in E.groupby(["model", "lambda_sal"]):
        row = dict(model=m, lambda_sal=lam, n_runs=len(g))
        for k in ["p_intact", "drop_tumour", "drop_control", "flip_tumour", "flip_control", "del_auc_evidence", "del_auc_random", "del_auc_signed"]:
            row[k + "_mean"], row[k + "_sd"] = g[k].mean(), g[k].std()
        if len(g) >= 5:
            row["p_tumour_vs_control"] = wilcoxon(g.drop_tumour, g.drop_control).pvalue
            row["p_evidence_vs_random"] = wilcoxon(g.del_auc_evidence, g.del_auc_random).pvalue
            if g.del_auc_signed.notna().sum() >= 5:
                row["p_signed_vs_random"] = wilcoxon(g.del_auc_signed, g.del_auc_random, nan_policy="omit").pvalue
        t.append(row)
    T = pd.DataFrame(t)
    for c in ["p_tumour_vs_control", "p_evidence_vs_random", "p_signed_vs_random"]:
        if c in T:
            ok = T[c].notna()
            T.loc[ok, c + "_holm"] = holm(T.loc[ok, c].values)
    T.to_csv(os.path.join(OUT, "faithfulness_summary.csv"), index=False)
    print(T.round(3).to_string(index=False))
    # supervision vs none on the same (fold, seed): does lambda = 0.25 change the reliance on the tumour or the map's faithfulness?
    s = []
    for m, g in E.groupby("model"):
        a, b = g[g.lambda_sal == 0].set_index(["fold", "seed"]), g[g.lambda_sal == 0.25].set_index(["fold", "seed"])
        j = a.join(b, lsuffix="_0", rsuffix="_1", how="inner")
        for k in ["drop_tumour", "drop_control", "del_auc_evidence", "del_auc_random", "del_auc_signed"]:
            d = (j[k + "_1"] - j[k + "_0"]).dropna()
            if len(d) >= 5:
                s.append(dict(model=m, metric=k, n_pairs=len(d), mean_diff=d.mean(), p_wilcoxon=wilcoxon(d).pvalue))
    if s:
        S = pd.DataFrame(s)
        S["p_holm"] = np.nan
        for met, idx in S.groupby("metric").groups.items():
            S.loc[idx, "p_holm"] = holm(S.loc[idx, "p_wilcoxon"].values)
        S.to_csv(os.path.join(OUT, "faithfulness_supervision.csv"), index=False)
        print(S.round(4).to_string(index=False))
# ---------------------------------------------------------------- E: mask dilation (lambda 0.25), paired with the dilation-1 runs of phase C
rr = runs[(runs.protocol == "root") & (runs.split_seed.fillna(0) == 0) & runs.model.isin(["convnext_tiny", "vit_small_patch16_224"])]
MET = ["macro_f1", "ece", "loc_pointing", "loc_mass_inside"]
E_rows = []
for m in ["convnext_tiny", "vit_small_patch16_224"]:
    ref = rr[(rr.model == m) & (rr.lambda_sal == 0.25) & (rr.dilate == 1) & (rr.train_frac == 1.0)].set_index(["fold", "seed"])
    for dil in (0, 2):
        cur = rr[(rr.model == m) & (rr.lambda_sal == 0.25) & (rr.dilate == dil)].set_index(["fold", "seed"])
        j = cur.join(ref, lsuffix="_x", rsuffix="_ref", how="inner")
        for met in MET:
            if len(j) >= 5:
                d = j[met + "_x"] - j[met + "_ref"]
                E_rows.append(dict(model=m, dilate=dil, metric=met, n_pairs=len(j), mean=j[met + "_x"].mean(), mean_ref=j[met + "_ref"].mean(),
                                   diff=d.mean(), p_wilcoxon=wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0))
if E_rows:
    Et = pd.DataFrame(E_rows)
    for met, idx in Et.groupby("metric").groups.items():
        Et.loc[idx, "p_holm"] = holm(Et.loc[idx, "p_wilcoxon"].values)
    Et.to_csv(os.path.join(OUT, "E_dilation.csv"), index=False)
    print(Et.round(4).to_string(index=False))

# ---------------------------------------------------------------- F: training-data fraction; lambda 0.25 vs 0 at each fraction (paired over fold, seed)
F_rows = []
for m in ["convnext_tiny", "vit_small_patch16_224"]:
    for frac in (0.25, 0.5, 1.0):
        g = rr[(rr.model == m) & (rr.dilate == 1) & (rr.train_frac == frac)]
        a, b = g[g.lambda_sal == 0].set_index(["fold", "seed"]), g[g.lambda_sal == 0.25].set_index(["fold", "seed"])
        j = b.join(a, lsuffix="_l1", rsuffix="_l0", how="inner")
        for met in MET:
            if len(j) >= 5:
                d = j[met + "_l1"] - j[met + "_l0"]
                F_rows.append(dict(model=m, train_frac=frac, metric=met, n_pairs=len(j), mean_l0=j[met + "_l0"].mean(), sd_l0=j[met + "_l0"].std(),
                                   mean_l025=j[met + "_l1"].mean(), sd_l025=j[met + "_l1"].std(), diff=d.mean(),
                                   p_wilcoxon=wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0))
if F_rows:
    Ft = pd.DataFrame(F_rows)
    for met, idx in Ft.groupby("metric").groups.items():
        Ft.loc[idx, "p_holm"] = holm(Ft.loc[idx, "p_wilcoxon"].values)
    Ft.to_csv(os.path.join(OUT, "F_datafrac.csv"), index=False)
    print(Ft.round(4).to_string(index=False))

# ---------------------------------------------------------------- robustness: lambda = 0.25 vs 0 per corruption (paired over fold, seed)
rob = []
for r in reg.itertuples():
    p = os.path.join(R, "experiments", r.exp_id, "robust.json")
    if os.path.exists(p):
        for c, v in json.load(open(p)).items():
            rob.append(dict(model=r.model, lambda_sal=r.lambda_sal, fold=r.fold, seed=r.seed, corruption=c, **v))
if rob:
    Rb = pd.DataFrame(rob)
    rows = []
    for (m, c), g in Rb.groupby(["model", "corruption"]):
        a = g[g.lambda_sal == 0].set_index(["fold", "seed"]).macro_f1
        b = g[g.lambda_sal == 0.25].set_index(["fold", "seed"]).macro_f1
        j = pd.concat([a, b], axis=1, keys=["l0", "l1"]).dropna()
        clean0 = Rb[(Rb.model == m) & (Rb.lambda_sal == 0) & (Rb.corruption == "clean")].macro_f1.mean()
        rows.append(dict(model=m, corruption=c, n_pairs=len(j), f1_l0=j.l0.mean(), f1_l025=j.l1.mean(), diff=(j.l1 - j.l0).mean(),
                         drop_from_clean_l0=clean0 - j.l0.mean(),
                         p_wilcoxon=wilcoxon(j.l1, j.l0).pvalue if len(j) >= 5 and not np.allclose(j.l1, j.l0) else np.nan))
    RS = pd.DataFrame(rows)
    ok = RS.p_wilcoxon.notna()
    RS.loc[ok, "p_holm"] = holm(RS.loc[ok, "p_wilcoxon"].values)
    RS.to_csv(os.path.join(OUT, "robustness_supervision.csv"), index=False)
    print(RS.round(4).to_string(index=False))
print("written", OUT)
