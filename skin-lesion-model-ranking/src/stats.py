"""All statistics of skin-lesion-model-ranking, computed only from stored runs (results/all_runs.csv, post_hoc.csv, predictions files).

Usage: python stats.py <skin-lesion-model-ranking dir>
Writes results/statistics/*.csv and results/statistics/summary.json. Every number in the manuscript comes from here.

RQ1 inflation      : image-split minus lesion-split, per backbone (mean over 5 folds); Wilcoxon signed-rank over
                     the 9 backbones; seen- vs unseen-lesion accuracy inside the image-split test folds.
RQ2 rankings       : backbone scores under image-CV, lesion-CV and the ISIC 2018 test set; Kendall tau between
                     rankings with a bootstrap over runs; Friedman test over backbones; family means.
RQ3 imbalance      : remedies vs CE on lesion-CV and ISIC test, paired over folds (pooled over the 3 backbones,
                     Wilcoxon, Holm-corrected) and a paired image-level bootstrap on the ISIC test set.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, MODELS, SHORT, classification_metrics  # noqa: E402

root = sys.argv[1]
R = os.path.join(root, "results")
S = os.path.join(R, "statistics")
os.makedirs(S, exist_ok=True)
runs = pd.read_csv(os.path.join(R, "all_runs.csv"))
post = pd.read_csv(os.path.join(R, "post_hoc.csv"))
runs = runs.merge(post, on="exp_id", how="left")
rng = np.random.default_rng(0)
summary = {}
METRICS = ["balanced_accuracy", "accuracy", "macro_f1", "macro_auc", "mel_auc", "mcc", "ece"] + ["recall_" + c for c in CLASSES]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def wilcoxon(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.all(d == 0):
        return np.nan
    return float(stats.wilcoxon(d).pvalue)


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (T+ - T-) / (T+ + T-), zero differences dropped."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d) & (d != 0)]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    tp, tm = r[d > 0].sum(), r[d < 0].sum()
    return float((tp - tm) / (tp + tm))


# ----------------------------------------------------------------------------------------------- RQ1
A = runs[runs["phase"] == "A"]
rows = []
for m in MODELS:
    for met in METRICS:
        a = A[(A.model == m) & (A.protocol == "image")].sort_values("fold")["test_" + met].values
        b = A[(A.model == m) & (A.protocol == "lesion")].sort_values("fold")["test_" + met].values
        ia = A[(A.model == m) & (A.protocol == "image")]["isic_" + met].values
        ib = A[(A.model == m) & (A.protocol == "lesion")]["isic_" + met].values
        if len(a) and len(b):
            rows.append(dict(model=m, short=SHORT[m], family=MODELS[m], metric=met, image_cv=a.mean(), lesion_cv=b.mean(),
                             inflation=a.mean() - b.mean(), image_isic=ia.mean() if len(ia) else np.nan,
                             lesion_isic=ib.mean() if len(ib) else np.nan,
                             gap_image=a.mean() - ia.mean() if len(ia) else np.nan, gap_lesion=b.mean() - ib.mean() if len(ib) else np.nan,
                             n_image=len(a), n_lesion=len(b)))
inf = pd.DataFrame(rows)
inf.to_csv(os.path.join(S, "rq1_inflation_by_model.csv"), index=False)
r1 = []
for met in (METRICS if len(inf) else []):
    d = inf[inf.metric == met]
    if len(d) == 0:
        continue
    bi = [rng.choice(d.inflation.values, len(d)).mean() for _ in range(2000)]  # bootstrap over backbones
    r1.append(dict(metric=met, n_models=len(d), mean_image_cv=d.image_cv.mean(), mean_lesion_cv=d.lesion_cv.mean(),
                   mean_inflation=d.inflation.mean(), ci_low=np.percentile(bi, 2.5), ci_high=np.percentile(bi, 97.5),
                   min=d.inflation.min(), max=d.inflation.max(),
                   models_inflated=int((d.inflation > 0).sum()), p_wilcoxon=wilcoxon(d.inflation.values),
                   mean_gap_image_cv_to_isic=d.gap_image.mean(), mean_gap_lesion_cv_to_isic=d.gap_lesion.mean(),
                   mean_isic_image_trained=d.image_isic.mean(), mean_isic_lesion_trained=d.lesion_isic.mean(),
                   p_isic_image_vs_lesion_trained=wilcoxon((d.image_isic - d.lesion_isic).values)))
r1 = pd.DataFrame(r1)
if len(r1):
    # Holm family = the seven summary metrics; per-class recalls are reported descriptively (unadjusted p), because with
    # nine backbones the smallest attainable Wilcoxon p (0.0039) cannot survive a correction over 14 metrics
    main = r1.metric.isin(["balanced_accuracy", "accuracy", "macro_f1", "macro_auc", "mel_auc", "mcc", "ece"])
    r1["p_holm"] = np.nan
    r1.loc[main, "p_holm"] = holm(r1.loc[main, "p_wilcoxon"].fillna(1).values)
r1.to_csv(os.path.join(S, "rq1_inflation_summary.csv"), index=False)
summary["rq1"] = r1.set_index("metric").to_dict("index") if len(r1) else {}

# seen vs unseen lesions inside image-split test folds
seen_rows = []
for r in A[A.protocol == "image"].to_dict("records"):
    p = pd.read_csv(os.path.join(root, "experiments", r["exp_id"], "predictions.csv"))
    for c_i, c in enumerate(CLASSES + ["all"]):
        q = p if c == "all" else p[p.label == c_i]
        for flag in [True, False]:
            s = q[q.lesion_in_train == flag]
            seen_rows.append(dict(exp_id=r["exp_id"], model=r["model"], fold=r["fold"], dx=c, lesion_in_train=flag, n=len(s),
                                  correct=int((s.label == s.pred).sum())))
seen = pd.DataFrame(seen_rows)
seen.to_csv(os.path.join(S, "rq1_seen_unseen_runs.csv"), index=False)
if len(seen):
    g = seen.groupby(["dx", "lesion_in_train"])[["n", "correct"]].sum().reset_index()
    g["accuracy"] = g.correct / g.n.clip(lower=1)
    g.to_csv(os.path.join(S, "rq1_seen_unseen.csv"), index=False)
    # per-run paired difference (all classes)
    sa = seen[seen.dx == "all"].pivot_table(index="exp_id", columns="lesion_in_train", values=["n", "correct"])
    acc_seen = sa[("correct", True)] / sa[("n", True)]
    acc_unseen = sa[("correct", False)] / sa[("n", False)]
    summary["rq1_seen_unseen"] = dict(runs=int(len(sa)), mean_acc_seen=float(acc_seen.mean()), mean_acc_unseen=float(acc_unseen.mean()),
                                      p_wilcoxon=wilcoxon((acc_seen - acc_unseen).values),
                                      share_test_seen=float(sa[("n", True)].sum() / (sa[("n", True)].sum() + sa[("n", False)].sum())))
    # the pooled comparison mixes classes (most nevi are single-image lesions, hence unseen), so compare within class:
    # per run, balanced accuracy over the classes that have both seen and unseen test images
    pc = seen[seen.dx != "all"].copy()
    pc["acc"] = pc.correct / pc.n.where(pc.n > 0)
    w = pc.pivot_table(index=["exp_id", "dx"], columns="lesion_in_train", values="acc").dropna()
    bal = w.groupby(level=0).mean()
    cls = w.groupby(level=1).mean()
    summary["rq1_seen_unseen_within_class"] = dict(
        runs=int(len(bal)), mean_bacc_seen=float(bal[True].mean()), mean_bacc_unseen=float(bal[False].mean()),
        runs_seen_higher=int((bal[True] > bal[False]).sum()), p_wilcoxon=wilcoxon((bal[True] - bal[False]).values),
        classes_seen_higher=int((cls[True] > cls[False]).sum()), n_classes=int(len(cls)),
        per_class_gap={c: float(cls.loc[c, True] - cls.loc[c, False]) for c in cls.index})

# ----------------------------------------------------------------------------------------------- RQ2
CE = runs[runs["loss"] == "ce"]
settings = {
    "image_cv": (CE[(CE.protocol == "image")], "test_"),
    "lesion_cv": (CE[(CE.protocol == "lesion")], "test_"),
    "isic_lesion_trained": (CE[(CE.protocol == "lesion")], "isic_"),
    "isic_image_trained": (CE[(CE.protocol == "image")], "isic_"),
    "isic_internal": (CE[(CE.protocol == "lesion")], "isicint_"),
    "isic_external": (CE[(CE.protocol == "lesion")], "isicext_"),
}
rank_rows, tau_rows = [], []
for met in ["balanced_accuracy", "macro_f1", "accuracy", "macro_auc"]:
    scores = {}
    for s, (d, pre) in settings.items():
        scores[s] = d.groupby("model")[pre + met].mean()
        for m, v in scores[s].items():
            sub = d[d.model == m][pre + met]
            rank_rows.append(dict(metric=met, setting=s, model=m, short=SHORT[m], family=MODELS[m], mean=v, sd=sub.std(ddof=1), n=len(sub)))
    sc = pd.DataFrame(scores).dropna(axis=1, how="all").dropna()
    if len(sc) < 3:
        continue
    for a in sc.columns:
        for b in sc.columns:
            if a >= b:
                continue
            tau = stats.kendalltau(sc[a], sc[b]).statistic
            # bootstrap: resample runs within each model and setting
            boots = []
            for _ in range(2000):
                va, vb = [], []
                for m in sc.index:
                    xa = settings[a][0][settings[a][0].model == m][settings[a][1] + met].values
                    xb = settings[b][0][settings[b][0].model == m][settings[b][1] + met].values
                    va.append(rng.choice(xa, len(xa)).mean()); vb.append(rng.choice(xb, len(xb)).mean())
                boots.append(stats.kendalltau(va, vb).statistic)
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            tau_rows.append(dict(metric=met, a=a, b=b, tau=tau, ci_low=lo, ci_high=hi, n_models=len(sc),
                                 top_a=SHORT[sc[a].idxmax()], top_b=SHORT[sc[b].idxmax()]))
pd.DataFrame(rank_rows).to_csv(os.path.join(S, "rq2_scores.csv"), index=False)
taus = pd.DataFrame(tau_rows)
taus.to_csv(os.path.join(S, "rq2_kendall.csv"), index=False)
summary["rq2_kendall_bacc"] = taus[taus.metric == "balanced_accuracy"].to_dict("records") if len(taus) else []

# seed-to-seed ranking stability on lesion-CV (phase A seed 0 + phase C seeds 1, 2)
L = CE[CE.protocol == "lesion"]
seed_rank = L.groupby(["seed", "model"])["test_balanced_accuracy"].mean().unstack(0).dropna()
if seed_rank.shape[1] >= 2:
    st = [stats.kendalltau(seed_rank[a], seed_rank[b]).statistic for i, a in enumerate(seed_rank) for b in list(seed_rank)[i + 1:]]
    summary["rq2_seed_tau"] = dict(pairs=len(st), n_models=int(len(seed_rank)), mean=float(np.mean(st)), min=float(np.min(st)), max=float(np.max(st)),
                                   top_by_seed={str(s): SHORT[seed_rank[s].idxmax()] for s in seed_rank})
    seed_rank.rename(index=SHORT).to_csv(os.path.join(S, "rq2_seed_scores.csv"))

# Friedman over backbones, blocks = fold x seed runs (lesion-CV) and per ISIC test
for pre, name in [("test_", "lesion_cv"), ("isic_", "isic")]:
    piv = L.pivot_table(index=["seed", "fold"], columns="model", values=pre + "balanced_accuracy").dropna()
    if piv.shape[0] >= 3 and piv.shape[1] >= 3:
        fr = stats.friedmanchisquare(*[piv[c].values for c in piv])
        ranks = piv.rank(axis=1, ascending=False).mean().rename(index=SHORT).sort_values()
        summary["rq2_friedman_" + name] = dict(blocks=int(piv.shape[0]), models=int(piv.shape[1]), chi2=float(fr.statistic),
                                               p=float(fr.pvalue), mean_rank=ranks.round(2).to_dict())
        # Nemenyi critical difference (alpha 0.05, k models); q values from Demsar (2006), Table 5
        q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        k, n = piv.shape[1], piv.shape[0]
        summary["rq2_friedman_" + name]["nemenyi_cd"] = float(q05[k] * np.sqrt(k * (k + 1) / (6 * n)))

# family means (lesion-CV and ISIC), each backbone weighted equally
fam = []
for pre in ["test_", "isic_"]:
    per_model = L.groupby("model")[pre + "balanced_accuracy"].mean()
    for f in ["CNN", "Transformer", "Hybrid"]:
        v = per_model[[m for m in per_model.index if MODELS[m] == f]]
        fam.append(dict(set=pre[:-1], family=f, mean=v.mean(), min=v.min(), max=v.max(), n_models=len(v)))
pd.DataFrame(fam).to_csv(os.path.join(S, "rq2_family.csv"), index=False)
summary["rq2_family"] = fam

# ----------------------------------------------------------------------------------------------- RQ3
B_MODELS = ["convnext_tiny.fb_in1k", "swin_tiny_patch4_window7_224.ms_in1k", "coatnet_0_rw_224.sw_in1k"]
Bset = runs[(runs.model.isin(B_MODELS)) & (runs.protocol == "lesion") & (runs.seed == 0)].copy()
# post-hoc logit adjustment of the CE runs is treated as a sixth "remedy"
ph = Bset[Bset.loss == "ce"].copy()
for pre in ["test_", "isic_"]:
    for met in METRICS:
        ph[pre + met] = ph[pre + "la_" + met]
ph["loss"] = "posthoc_la"
Bset = pd.concat([Bset, ph], ignore_index=True)
REMEDIES = ["wce", "focal", "la", "sampler", "posthoc_la"]
r3, r3_rows = [], []
for pre in ["test_", "isic_"]:
    for met in METRICS:
        pvals, idx = [], []
        for rem in REMEDIES:
            diffs = []
            for m in B_MODELS:
                ce = Bset[(Bset.model == m) & (Bset.loss == "ce")].set_index("fold")[pre + met]
                rv = Bset[(Bset.model == m) & (Bset.loss == rem)].set_index("fold")[pre + met]
                common = ce.index.intersection(rv.index)
                diffs.extend((rv[common] - ce[common]).values)
                if len(common):
                    r3_rows.append(dict(set=pre[:-1], metric=met, remedy=rem, model=m, short=SHORT[m], n=len(common),
                                        ce=ce[common].mean(), remedy_value=rv[common].mean(), diff=(rv[common] - ce[common]).mean()))
            if diffs:
                r3.append(dict(set=pre[:-1], metric=met, remedy=rem, n_pairs=len(diffs), mean_diff=np.mean(diffs), sd_diff=np.std(diffs, ddof=1),
                               pairs_improved=int(np.sum(np.array(diffs) > 0)), p_wilcoxon=wilcoxon(diffs),
                               rank_biserial=rank_biserial(diffs)))
                pvals.append(r3[-1]["p_wilcoxon"]); idx.append(len(r3) - 1)
        if pvals:
            adj = holm(np.nan_to_num(pvals, nan=1.0))
            for i, a in zip(idx, adj):
                r3[i]["p_holm_within_metric"] = a
r3 = pd.DataFrame(r3)
r3.to_csv(os.path.join(S, "rq3_remedies_pooled.csv"), index=False)
pd.DataFrame(r3_rows).to_csv(os.path.join(S, "rq3_remedies_by_model.csv"), index=False)

# paired image-level bootstrap on the ISIC test set: metric of the mean over the 5 fold models' probabilities is NOT used;
# instead the per-run metric is averaged over the 5 folds for each bootstrap resample of the 1512 images.
def load_isic(exp_id, posthoc_tau=None, prior=None):
    p = pd.read_csv(os.path.join(root, "experiments", exp_id, "predictions_isic.csv"))
    z = p[["logit_" + c for c in CLASSES]].values
    if posthoc_tau is not None:
        z = z - posthoc_tau * np.log(prior)
    z = z - z.max(1, keepdims=True)
    pr = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p["label"].values, pr


def bacc(y, pred):
    rec = [np.mean(pred[y == k] == k) for k in range(len(CLASSES)) if np.any(y == k)]
    return float(np.mean(rec))


boot_rows = []
n_boot = 2000
for m in B_MODELS:
    base = Bset[(Bset.model == m) & (Bset.loss == "ce")].sort_values("fold")
    ce_preds = []
    for r in base.to_dict("records"):
        y, pr = load_isic(r["exp_id"])
        ce_preds.append(pr.argmax(1))
    for rem in REMEDIES:
        sub = Bset[(Bset.model == m) & (Bset.loss == rem)].sort_values("fold")
        if len(sub) != len(base) or len(sub) == 0:
            continue
        rem_preds = []
        for r in sub.to_dict("records"):
            if rem == "posthoc_la":
                res = json.load(open(os.path.join(root, "experiments", r["exp_id"], "results.json")))
                cnt = np.array(res["split"]["train_class_counts"], float)
                y, pr = load_isic(r["exp_id"], r["tau"], cnt / cnt.sum())
            else:
                y, pr = load_isic(r["exp_id"])
            rem_preds.append(pr.argmax(1))
        obs = np.mean([bacc(y, a) for a in rem_preds]) - np.mean([bacc(y, b) for b in ce_preds])
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            bs.append(np.mean([bacc(y[i], a[i]) for a in rem_preds]) - np.mean([bacc(y[i], b[i]) for b in ce_preds]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        boot_rows.append(dict(model=m, short=SHORT[m], remedy=rem, isic_bacc_diff=obs, ci_low=lo, ci_high=hi,
                              p_two_sided=float(min(1.0, 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))))))
bt = pd.DataFrame(boot_rows)
bt.to_csv(os.path.join(S, "rq3_isic_bootstrap.csv"), index=False)
summary["rq3_isic_bootstrap"] = bt.to_dict("records")
summary["rq3_pooled_bacc"] = r3[r3.metric == "balanced_accuracy"].to_dict("records") if len(r3) else []

json.dump(summary, open(os.path.join(S, "summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("rq3_isic")}, indent=1, default=float)[:6000])
