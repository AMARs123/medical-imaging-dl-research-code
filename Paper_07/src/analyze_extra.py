"""Paper 07, analyses added after the main results (labelled as such in the paper). Uses only stored logits.

1. Bootstrap confidence intervals (Eq. boot, B=2000): for each configuration, target and test set, the mean AUC over the six
   runs (two backbones x three seeds) is recomputed on resampled images (patients for PAPILA; one resample shared by the six
   runs); for each camera and target the mean camera gap Delta_c (Eq. gap) on resampled test-split images of that camera.
2. Prior correction of the class weight (post hoc): the weighted loss (Eq. loss) shifts every logit by about log w+, so the
   probabilities are re-computed as sigmoid(z - log w+) with the run's own w+; AUC and the errors at the shifted threshold do
   not change, only calibration does. ECE, Brier and rho(H, m) before and after, and the paired soft-minus-hard ECE after the
   correction (Wilcoxon, Holm over the test sets).
3. Mean probability by number of experts a(x) on the Chaksu test split (all-camera models) and on the held-out cameras,
   with p* of Eq. pstar, for fig_calib.
4. Size-matched control (config all_n271, 12 runs added after the main results): on the official test-split Remidio images,
   AUC of the all-camera model, of the all-camera model trained on as many images as the leave-out-Remidio model, and of the
   leave-out-Remidio model; size effect = all - all_n271, camera effect = all_n271 - loco_Remidio, paired by target, backbone
   and seed, with image-bootstrap CIs of the mean (B=2000).
Writes results/statistics/{ci_auc.csv, ci_gap.csv, prior_corr.csv, prior_corr_pairs.csv, prob_by_votes.csv, control.csv}.
Usage: python analyze_extra.py <Paper_07 dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

R = sys.argv[1]
P = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
B = 2000
pr = pd.read_csv(os.path.join(P, "predictions_all.csv.gz"), keep_default_na=False, na_values=[""])
runs = pd.read_csv(os.path.join(P, "runs.csv"))
wpos = runs.drop_duplicates("exp_id").set_index("exp_id").w_pos
pr["z"] = pr.logit
pr["z_corr"] = pr.logit - np.log(pr.exp_id.map(wpos).values)
OWN = {"all": "chaksu_test", "all_n271": "chaksu_test", "loco_Remidio": "held_out_camera", "loco_Bosch": "held_out_camera", "loco_Forus": "held_out_camera"}


def fast_auc(y, s):
    """AUC (Eq. auc) with mid-ranks; s may be 2-D (runs x images)."""
    r = np.apply_along_axis(rankdata, -1, s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[..., y == 1].sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_idx(groups, rng):
    """Indices of one bootstrap resample; groups (patients) are resampled when given."""
    if groups is None:
        return rng.integers(0, n_img, n_img)
    u = np.unique(groups)
    pick = rng.choice(u, len(u))
    pos = {g: np.where(groups == g)[0] for g in u}
    return np.concatenate([pos[g] for g in pick])


# ---------------------------------------------------------------- 1. bootstrap CIs
ci = []
for (cfg, tg), _ in runs.groupby(["config", "target"]):
    for s in [OWN[cfg], "papila", "rimone"]:
        q = pr[(pr.config == cfg) & (pr.target == tg) & (pr.set == s)]
        piv = q.pivot_table(index="array_index", columns="exp_id", values="z")
        lab = q.drop_duplicates("array_index").set_index("array_index").reindex(piv.index)
        y = lab.label.values.astype(int)
        Z = piv.values.T
        groups = lab.patient.values if s == "papila" else None
        n_img = len(y)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(B):
            i = boot_idx(groups, rng)
            if 0 < y[i].sum() < len(i):
                bs.append(fast_auc(y[i], Z[:, i]).mean())
        ci.append(dict(config=cfg, target=tg, set=s, n_runs=Z.shape[0], n=n_img, auc_mean=fast_auc(y, Z).mean(),
                       lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975), n_boot=len(bs)))
ci = pd.DataFrame(ci)
ci.to_csv(os.path.join(S, "ci_auc.csv"), index=False)

cg = []
for cam in ["Remidio", "Bosch", "Forus"]:
    for tg in ["hard", "soft"]:
        a = pr[(pr.config == "all") & (pr.target == tg) & (pr.set == "chaksu_test") & (pr.camera == cam)]
        b = pr[(pr.config == "loco_" + cam) & (pr.target == tg) & (pr.set == "held_out_camera") & (pr.split == "test")]
        A = a.pivot_table(index="array_index", columns=["model", "seed"], values="z")
        Bm = b.pivot_table(index="array_index", columns=["model", "seed"], values="z").reindex(index=A.index, columns=A.columns)
        y = a.drop_duplicates("array_index").set_index("array_index").reindex(A.index).label.values.astype(int)
        n_img = len(y)
        rng = np.random.default_rng(0)
        d0 = (fast_auc(y, A.values.T) - fast_auc(y, Bm.values.T)).mean()
        bs = []
        for _ in range(B):
            i = rng.integers(0, n_img, n_img)
            if 0 < y[i].sum() < n_img:
                bs.append((fast_auc(y[i], A.values.T[:, i]) - fast_auc(y[i], Bm.values.T[:, i])).mean())
        bs = np.array(bs)
        cg.append(dict(camera=cam, target=tg, n=n_img, n_pos=int(y.sum()), gap=d0, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                       share_pos=float((bs > 0).mean())))
cg = pd.DataFrame(cg)
cg.to_csv(os.path.join(S, "ci_gap.csv"), index=False)


# ---------------------------------------------------------------- 2. prior correction
def ece(p, y, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def ent_rho(p, nsus):
    pp = np.clip(p, 1e-6, 1 - 1e-6)
    H = -pp * np.log(pp) - (1 - pp) * np.log(1 - pp)
    return spearmanr(H, np.minimum(nsus, 5 - nsus)).correlation


rows = []
for (e, s), g in pr[pr.set.isin(["chaksu_test", "held_out_camera", "papila", "rimone"])].groupby(["exp_id", "set"]):
    y = g.label.values.astype(float)
    row = dict(exp_id=e, config=g.config.iloc[0], target=g.target.iloc[0], model=g.model.iloc[0], seed=g.seed.iloc[0], set=s, w_pos=wpos[e])
    for k, z in [("raw", g.z.values), ("corr", g.z_corr.values)]:
        p = 1 / (1 + np.exp(-z))
        row["ece_" + k], row["brier_" + k], row["meanp_" + k] = ece(p, y), np.mean((p - y) ** 2), p.mean()
        if g.n_suspect.notna().all() and s != "papila":
            row["rho_" + k] = ent_rho(p, g.n_suspect.values.astype(float))
    rows.append(row)
pc = pd.DataFrame(rows)
pc.to_csv(os.path.join(S, "prior_corr.csv"), index=False)


def rb(d):
    d = d[d != 0]
    rr = rankdata(np.abs(d))
    return (rr[d > 0].sum() - rr[d < 0].sum()) / rr.sum() if len(d) else 0.0


pairs = []
for s in ["chaksu_test", "held_out_camera", "papila", "rimone"]:
    for k in ["raw", "corr"]:
        pv = pc[(pc.set == s) & (pc.config != "all_n271")].pivot_table(index=["config", "model", "seed"], columns="target", values="ece_" + k)
        d = (pv.soft - pv.hard).dropna().values
        pairs.append(dict(set=s, version=k, n_pairs=len(d), ece_hard=pv.hard.mean(), ece_soft=pv.soft.mean(), mean_diff=d.mean(),
                          soft_better=int((d < 0).sum()), rb=rb(d), p=wilcoxon(d).pvalue))
pairs = pd.DataFrame(pairs)
for k in ["raw", "corr"]:
    q = pairs[pairs.version == k].sort_values("p")
    adj, run = [], 0.0
    for i, v in enumerate(q.p.values):
        run = max(run, min(1.0, (len(q) - i) * v))
        adj.append(run)
    pairs.loc[q.index, "p_holm"] = adj
pairs.to_csv(os.path.join(S, "prior_corr_pairs.csv"), index=False)

# ---------------------------------------------------------------- 3. probability by number of experts
q = pr[pr.set.isin(["chaksu_test", "held_out_camera"]) & (pr.config != "all_n271")].copy()
q["p_raw"] = 1 / (1 + np.exp(-q.z))
q["p_corr"] = 1 / (1 + np.exp(-q.z_corr))
q["w"] = q.exp_id.map(wpos)
pv = q.groupby(["set", "target", "n_suspect"]).agg(p_raw=("p_raw", "mean"), p_corr=("p_corr", "mean"), w_pos=("w", "mean"), n=("p_raw", "size")).reset_index()
a = pv.n_suspect / 5.0
pv["p_star"] = pv.w_pos * a / (pv.w_pos * a + 1 - a)
pv.to_csv(os.path.join(S, "prob_by_votes.csv"), index=False)

# ---------------------------------------------------------------- 4. size-matched control
ctl = []
if (pr.config == "all_n271").any():
    for tg in ["hard", "soft"]:
        M = {}
        for cfg, st in [("all", "chaksu_test"), ("all_n271", "chaksu_test"), ("loco_Remidio", "held_out_camera")]:
            q = pr[(pr.config == cfg) & (pr.target == tg) & (pr.set == st) & (pr.camera == "Remidio") & (pr.split == "test")]
            M[cfg] = q.pivot_table(index="array_index", columns=["model", "seed"], values="z")
        cols = M["all"].columns.intersection(M["all_n271"].columns).intersection(M["loco_Remidio"].columns)
        idx = M["all"].index
        Z = {k: v.reindex(index=idx, columns=cols).values.T for k, v in M.items()}
        y = pr[pr.array_index.isin(idx)].drop_duplicates("array_index").set_index("array_index").reindex(idx).label.values.astype(int)
        A = {k: fast_auc(y, v) for k, v in Z.items()}
        rng = np.random.default_rng(0)
        bs_size, bs_cam = [], []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(y):
                a = {k: fast_auc(y[i], v[:, i]) for k, v in Z.items()}
                bs_size.append((a["all"] - a["all_n271"]).mean())
                bs_cam.append((a["all_n271"] - a["loco_Remidio"]).mean())
        ctl.append(dict(target=tg, n_pairs=len(cols), n=len(y), n_pos=int(y.sum()), auc_all=A["all"].mean(), auc_n271=A["all_n271"].mean(),
                        auc_loco=A["loco_Remidio"].mean(), size_effect=(A["all"] - A["all_n271"]).mean(), size_lo=np.quantile(bs_size, 0.025),
                        size_hi=np.quantile(bs_size, 0.975), camera_effect=(A["all_n271"] - A["loco_Remidio"]).mean(),
                        cam_lo=np.quantile(bs_cam, 0.025), cam_hi=np.quantile(bs_cam, 0.975), cam_pairs_pos=int(((A["all_n271"] - A["loco_Remidio"]) > 0).sum())))
pd.DataFrame(ctl).to_csv(os.path.join(S, "control.csv"), index=False)
print(pd.DataFrame(ctl).round(3).to_string())
print(ci.round(3).to_string())
print(cg.round(3).to_string())
print(pairs.round(4).to_string())
print(pc.groupby(["set", "target"])[["ece_raw", "ece_corr", "rho_raw", "rho_corr", "w_pos"]].mean().round(3).to_string())
print(pv.round(3).to_string())
print("EXTRA_DONE")
