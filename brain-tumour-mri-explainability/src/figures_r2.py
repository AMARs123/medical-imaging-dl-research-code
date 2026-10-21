"""Figures added in the Research_Expert round (2026-10-16). Reads only stored result files; CPU only.
Usage: python figures_r2.py <brain-tumour-mri-explainability dir>   ->  figures/final/*.pdf|png

  fig_rank_cd       rank view + statistical comparison: mean rank of macro-F1 over the (fold, seed) blocks with a 95%
                    bootstrap interval; open markers = significantly worse than the best model (Holm-corrected Wilcoxon).
  fig_distribution  distribution view: every run's macro-F1 per architecture (points) with the median and quartiles.
  fig_pareto        trade-off view: macro-F1 against measured GPU latency per image (batch 32), Pareto front marked.
  fig_profile       multi-metric profile: rank of each architecture on eight measured quantities (1 = best).
  fig_failure       failure analysis: the six most confident errors of the best architecture (seed 0, pooled test folds),
                    selected by the stated rule, with the expert outline and the predicted-class evidence map.
  fig_overlap       distribution of the best correlation of each BRISC tumour slice with any Figshare slice.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

R = sys.argv[1]
OUT = os.path.join(R, "figures", "final")
ST = os.path.join(R, "results", "statistics")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 2, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})
NICE = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "efficientnet_b0": "EfficientNet-B0",
        "convnext_tiny": "ConvNeXt-T", "mobilevit_s": "MobileViT-S", "vit_small_patch16_224": "ViT-S/16",
        "deit_small_patch16_224": "DeiT-S/16", "swin_tiny_patch4_window7_224": "Swin-T"}
COL = {"resnet50": "#2a78d6", "densenet121": "#eb6834", "efficientnet_b0": "#1baf7a", "convnext_tiny": "#eda100",
       "mobilevit_s": "#a8509a", "vit_small_patch16_224": "#e87ba4", "deit_small_patch16_224": "#008300",
       "swin_tiny_patch4_window7_224": "#6250d6"}
MARK = {"resnet50": "o", "densenet121": "s", "efficientnet_b0": "D", "convnext_tiny": "P", "mobilevit_s": "X",
        "vit_small_patch16_224": "^", "deit_small_patch16_224": "v", "swin_tiny_patch4_window7_224": "*"}
CLS = ["meningioma", "glioma", "pituitary"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


runs = pd.read_csv(os.path.join(R, "results", "processed", "all_runs.csv"))
B = runs[(runs.protocol == "root") & (runs.lambda_sal == 0) & runs.main.astype(bool)]
wide = B.pivot_table(index=["fold", "seed"], columns="model", values="macro_f1").dropna()
models = [m for m in NICE if m in wide.columns]
wide = wide[models]

# ---------------------------------------------------------------- rank view with the significance result
fr = pd.read_csv(os.path.join(ST, "B_friedman.csv"))
best = fr.vs.iloc[0]
worse = set(fr[fr.p_holm < 0.05].model)
rk = pd.DataFrame(np.vstack([rankdata(-r) for r in wide.values]), columns=models)  # 1 = best macro-F1 in the block
rng = np.random.default_rng(2026)
bs = np.stack([rk.values[rng.integers(0, len(rk), len(rk))].mean(0) for _ in range(4000)])
mean_rank = rk.mean()
order = mean_rank.sort_values().index
fig, ax = plt.subplots(figsize=(3.46, 2.6))
for i, m in enumerate(order):
    j = models.index(m)
    lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
    ax.plot([lo, hi], [i, i], color=COL[m], lw=1.5)
    sig = m in worse
    ax.scatter(mean_rank[m], i, s=34, marker=MARK[m], color="white" if sig else COL[m], edgecolor=COL[m], linewidth=1.2, zorder=3)
ax.set_yticks(range(len(order)), [NICE[m] + (" (best)" if m == best else "") for m in order])
ax.invert_yaxis()
ax.set_xlabel("Mean rank of macro-F1 (1 = best), %d blocks" % len(wide))
ax.set_xlim(0.5, len(models) + 0.5)
ax.set_title("filled: not separable from the best; open: worse ($p_\\mathrm{Holm}<0.05$)", fontsize=8, color=MUTED, loc="left")
save(fig, "fig_rank_cd")

# ---------------------------------------------------------------- distribution of run-level macro-F1
fig, ax = plt.subplots(figsize=(7.09, 2.4))
for i, m in enumerate(models):
    v = B[B.model == m].macro_f1.values
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.4, zorder=4)
    ax.add_patch(plt.Rectangle((i - 0.28, q1), 0.56, q3 - q1, facecolor="none", edgecolor=MUTED, lw=0.8, zorder=3))
    x = i + rng.uniform(-0.18, 0.18, len(v))
    ax.scatter(x, v, s=9, marker=MARK[m], color=COL[m], alpha=0.75, lw=0, zorder=2)
ax.set_xticks(range(len(models)), [NICE[m] for m in models], rotation=20)
ax.set_ylabel("Macro-F1 of one run")
ax.set_title("Each point is one test fold of one seed (n = %d per model); box: quartiles, line: median" % B.groupby("model").size().min(),
             fontsize=8, color=MUTED, loc="left")
save(fig, "fig_distribution")

# ---------------------------------------------------------------- accuracy against latency
EF = os.path.join(R, "results", "processed", "efficiency.csv")
if os.path.exists(EF):
    ef = pd.read_csv(EF).set_index("model")
    pts = [(m, ef.loc[m, "gpu_ms_b32_per"], B[B.model == m].macro_f1.mean(), ef.loc[m, "cpu_ms_b1_t4"]) for m in models if m in ef.index]
    fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.6), sharey=True)
    for ax, col, lab in [(axs[0], 1, "GPU latency per image, batch 32 (ms, log)"), (axs[1], 3, "CPU latency, batch 1, 4 threads (ms, log)")]:
        srt = sorted(pts, key=lambda t: t[col])
        front, bestf = [], -1
        for t in srt:  # Pareto front: no other model is both faster and more accurate
            if t[2] > bestf:
                front.append(t)
                bestf = t[2]
        ax.plot([t[col] for t in front], [t[2] for t in front], color=MUTED, lw=1, ls="--", zorder=1)
        for m, g, f1, c in pts:
            x = g if col == 1 else c
            ax.scatter(x, f1, s=40, marker=MARK[m], color=COL[m], edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(NICE[m], (x, f1), xytext=(4, 2), textcoords="offset points", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
    axs[0].set_ylabel("Macro-F1 (mean over runs)")
    axs[0].text(0.02, 0.02, "dashed: Pareto front", transform=axs[0].transAxes, fontsize=8, color=MUTED)
    save(fig, "fig_pareto")
else:  # clearly marked placeholder until the efficiency measurements exist
    fig, ax = plt.subplots(figsize=(7.09, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, "PENDING: efficiency measurements not yet run", ha="center", va="center", fontsize=10, color=INK)
    save(fig, "fig_pareto")

# ---------------------------------------------------------------- multi-metric profile
prof = pd.DataFrame(index=models)
prof["Macro-F1"] = B.groupby("model").macro_f1.mean()
prof["Macro-AUC"] = B.groupby("model").macro_auc.mean()
prof["ECE"] = B.groupby("model").ece.mean()
dg = pd.read_csv(os.path.join(ST, "evidence_diagnostics.csv"))
dg = dg[(dg.protocol == "root") & (dg.lambda_sal == 0) & (dg.split_seed.fillna(0) == 0) & (dg.dilate.fillna(1) == 1) & (dg.train_frac.fillna(1) == 1)]
prof["Pointing lift"] = (dg.pointing_cell / dg.pointing_cell_chance).groupby(dg.model).mean()
fsum = os.path.join(ST, "faithfulness_runs.csv")
if os.path.exists(fsum):
    F = pd.read_csv(fsum)
    F = F[F.lambda_sal == 0]
    prof["Deletion gain"] = (F.del_auc_random - F.del_auc_signed).groupby(F.model).mean()
    prof["Occlusion drop"] = F.drop_tumour.groupby(F.model).mean()
EXr = os.path.join(R, "results", "processed", "external_runs.csv")
if os.path.exists(EXr):
    X = pd.read_csv(EXr)
    X = X[(X.lambda_sal == 0) & X.exp_id.isin(B.exp_id)]
    prof["External acc."] = X.groupby("model").accuracy.mean()
if os.path.exists(EF):
    prof["CPU latency"] = pd.read_csv(EF).set_index("model").cpu_ms_b1_t4
lower_better = {"ECE", "CPU latency"}
prof = prof.dropna(axis=1, how="any")
ranks = prof.apply(lambda c: rankdata(c if c.name in lower_better else -c, method="min"))
fig, ax = plt.subplots(figsize=(6.4, 2.9))
im = ax.imshow(ranks.values, cmap="Blues_r", vmin=1, vmax=len(models) + 2, aspect="auto")
for i in range(ranks.shape[0]):
    for j in range(ranks.shape[1]):
        v = prof.values[i, j]
        cname = ranks.columns[j]
        txt = "%.1f" % v if cname == "Pointing lift" else ("%.0f" % v if cname == "CPU latency" else "%.3f" % v)
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.2, color="white" if ranks.values[i, j] <= 2 else INK)
ax.set_xticks(range(ranks.shape[1]), [c + (" (low)" if c in lower_better else "") for c in ranks.columns], rotation=25, ha="right")
ax.set_yticks(range(len(models)), [NICE[m] for m in models])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=range(1, len(models) + 1))
cb.set_label("Rank (1 = best)", fontsize=8)
save(fig, "fig_profile")

# ---------------------------------------------------------------- failure analysis
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
for c, v in [("split_seed", 0), ("dilate", 1), ("train_frac", 1.0)]:
    reg[c] = reg[c].fillna(v)
bestm = best
sel = reg[(reg.model == bestm) & (reg.protocol == "root") & (reg.lambda_sal == 0) & (reg.seed == 0) & (reg.split_seed == 0)
          & (reg.dilate == 1) & (reg.train_frac == 1.0)]
P, EV = [], []
for e in sel.exp_id:
    p = pd.read_csv(os.path.join(R, "experiments", e, "predictions.csv"))
    ev = np.load(os.path.join(R, "experiments", e, "evidence_test.npy")).astype(np.float32)
    p["row"] = np.arange(len(p))
    p["exp_id"] = e
    P.append(p)
    EV.append(ev)
P = pd.concat(P, ignore_index=True)
P["conf"] = P[["p0", "p1", "p2"]].max(1)
err = P[P.label != P.pred].sort_values("conf", ascending=False)
ct = pd.crosstab(P.label.map(dict(enumerate(CLS))), P.pred.map(dict(enumerate(CLS))))
ct.to_csv(os.path.join(ST, "failure_confusion_%s_seed0.csv" % bestm))
pick = err.head(6)
images = np.load(os.path.join(R, "data", "processed", "images.npy"), mmap_mode="r")
masks = np.load(os.path.join(R, "data", "processed", "masks.npy"), mmap_mode="r")
evd = {e: ev for e, ev in zip(sel.exp_id, EV)}
fig, axs = plt.subplots(2, 6, figsize=(7.09, 2.7))
import torch  # noqa: E402
import torch.nn.functional as Fnn  # noqa: E402
for k, r in enumerate(pick.itertuples()):
    img, msk = images[r.array_index], masks[r.array_index]
    e = evd[r.exp_id][r.row, r.pred]
    heat = Fnn.interpolate(torch.from_numpy(np.maximum(e, 0))[None, None], size=img.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    axs[0, k].imshow(img, cmap="gray")
    axs[0, k].contour(msk, levels=[0.5], colors=["#eb6834"], linewidths=0.8)
    axs[0, k].set_title("%s $\\to$ %s\np = %.2f" % (CLS[r.label][0].upper(), CLS[r.pred][0].upper(), r.conf), fontsize=8, color=INK)
    axs[1, k].imshow(heat, cmap="viridis")
    axs[1, k].contour(msk, levels=[0.5], colors=["white"], linewidths=0.7)
    for a in axs[:, k]:
        a.set_xticks([]), a.set_yticks([])
        a.grid(False)
axs[0, 0].set_ylabel("MRI + outline", fontsize=8)
axs[1, 0].set_ylabel("evidence", fontsize=8)
save(fig, "fig_failure")
json.dump(dict(model=bestm, n_test=len(P), n_errors=int((P.label != P.pred).sum()), shown=pick[["array_index", "label", "pred", "conf",
          "mass_inside", "pointing"]].to_dict("records"), errors_mass_inside_mean=float(err.mass_inside.mean()),
          correct_mass_inside_mean=float(P[P.label == P.pred].mass_inside.mean()), confusion=ct.to_dict()),
          open(os.path.join(ST, "failure_analysis.json"), "w"), indent=1)

# ---------------------------------------------------------------- BRISC overlap distribution
OV = os.path.join(R, "data", "external", "processed", "meta.csv")
if os.path.exists(OV):
    bm = pd.read_csv(OV)
    fig, ax = plt.subplots(figsize=(3.46, 2.3))
    bins = np.linspace(min(0, bm.best_corr.min()), 1.0, 51)
    ax.hist(bm.best_corr[~bm.near_duplicate], bins=bins, color="#2a78d6", label="independent (%d)" % (~bm.near_duplicate).sum())
    ax.hist(bm.best_corr[bm.near_duplicate], bins=bins, color="#eb6834", label="copy of a Figshare slice (%d)" % bm.near_duplicate.sum())
    ax.axvline(0.95, color=MUTED, ls="--", lw=1)
    ax.text(0.94, ax.get_ylim()[1] * 0.5, "threshold 0.95", rotation=90, ha="right", va="center", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("Best correlation with any Figshare slice")
    ax.set_ylabel("BRISC tumour slices (log)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig_overlap")
print("FIGURES_R2_DONE")
