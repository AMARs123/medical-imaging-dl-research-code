"""All result figures of Paper 02, drawn only from results/ (all_runs.csv, post_hoc.csv, statistics/*.csv).

Usage: python figures.py <Paper_02 dir>   ->  figures/final/*.pdf and *.png
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
from common import CLASSES, MODELS, SHORT  # noqa: E402
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
F = os.path.join(R, "figures", "final")
os.makedirs(F, exist_ok=True)
V.setup()
FAM_COL = {"CNN": V.CAT[0], "Transformer": V.CAT[1], "Hybrid": V.CAT[2]}
runs = pd.read_csv(os.path.join(R, "results", "all_runs.csv")).merge(pd.read_csv(os.path.join(R, "results", "post_hoc.csv")), on="exp_id")
CE = runs[runs.loss == "ce"]

# ---------------------------------------------------------------- Fig. inflation per class (dumbbell)
A = runs[runs.phase == "A"]
fig, ax = plt.subplots(figsize=(6.6, 2.9))
for i, c in enumerate(CLASSES):
    im = A[A.protocol == "image"].groupby("model")["test_recall_" + c].mean()
    le = A[A.protocol == "lesion"].groupby("model")["test_recall_" + c].mean()
    ax.plot([i - 0.12, i + 0.12], [le.mean(), im.mean()], color=V.GRID, lw=2, zorder=1)
    ax.scatter(np.full(len(le), i - 0.12), le.values, s=12, color=V.CAT[0], alpha=0.35, lw=0)
    ax.scatter(np.full(len(im), i + 0.12), im.values, s=12, color=V.CAT[1], alpha=0.35, lw=0)
    ax.scatter([i - 0.12], [le.mean()], s=40, color=V.CAT[0], edgecolor="white", lw=1, zorder=3, label="Lesion split" if i == 0 else None)
    ax.scatter([i + 0.12], [im.mean()], s=40, color=V.CAT[1], edgecolor="white", lw=1, zorder=3, label="Image split" if i == 0 else None)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
ax.set_ylabel("Recall on test folds"); ax.set_ylim(0, 1.02)
ax.legend(loc="lower left", ncol=2)
ax.set_title("Per-class recall: large dots = mean of nine backbones, small dots = single backbones", loc="left")
V.save(fig, os.path.join(F, "fig_inflation_class"))

# ---------------------------------------------------------------- Fig. backbone scores in three settings
sc = pd.read_csv(os.path.join(S, "rq2_scores.csv"))
b = sc[sc.metric == "balanced_accuracy"].pivot_table(index="model", columns="setting", values="mean")
bsd = sc[sc.metric == "balanced_accuracy"].pivot_table(index="model", columns="setting", values="sd")
order = b.sort_values("lesion_cv").index
fig, ax = plt.subplots(figsize=(6.6, 3.4))
y = np.arange(len(order))
for setting, mk, lab in [("image_cv", "o", "Image-split CV"), ("lesion_cv", "s", "Lesion-split CV"), ("isic_lesion_trained", "D", "ISIC 2018 test"),
                         ("isic_external", "^", "ISIC 2018 external part")]:
    if setting not in b:
        continue
    for k, m in enumerate(order):
        ax.errorbar(b.loc[m, setting], k, xerr=(bsd[setting].get(m, np.nan) if setting in bsd else np.nan), fmt=mk, ms=5, color=FAM_COL[MODELS[m]],
                    mfc="white" if setting in ("image_cv", "isic_external") else FAM_COL[MODELS[m]], elinewidth=0.8, capsize=0,
                    label=lab if k == 0 else None)
ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in order])
ax.set_xlabel("Balanced accuracy (mean $\\pm$ SD over runs)")
h, l = ax.get_legend_handles_labels()
for hh in h:
    try:
        hh[0].set_color(V.INK2)
    except Exception:
        pass
ax.legend(h, l, loc="lower right", fontsize=7)
for fam, col in FAM_COL.items():
    ax.plot([], [], "s", color=col, label=fam)
ax.set_title("Backbones by family (colour): CNN blue, transformer orange, hybrid aqua", loc="left")
V.save(fig, os.path.join(F, "fig_backbone_settings"))

# ---------------------------------------------------------------- Fig. rank bump chart
settings = [s for s in ["image_cv", "lesion_cv", "isic_internal", "isic_external"] if s in b]
names = {"image_cv": "Image-split\nCV", "lesion_cv": "Lesion-split\nCV", "isic_internal": "ISIC test\ninternal", "isic_external": "ISIC test\nexternal"}
rk = b[settings].rank(ascending=False)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for m in rk.index:
    ax.plot(range(len(settings)), rk.loc[m].values, "-o", color=FAM_COL[MODELS[m]], lw=1.6, ms=5, mec="white")
    ax.text(len(settings) - 1 + 0.08, rk.loc[m].values[-1], SHORT[m], va="center", fontsize=7.5, color=V.INK)
    ax.text(-0.08, rk.loc[m].values[0], SHORT[m], va="center", ha="right", fontsize=7.5, color=V.INK)
ax.set_xticks(range(len(settings))); ax.set_xticklabels([names[s] for s in settings])
ax.set_yticks(range(1, 10)); ax.invert_yaxis(); ax.set_ylabel("Rank by balanced accuracy (1 = best)")
ax.set_xlim(-1.2, len(settings) - 1 + 1.2); ax.grid(axis="x", visible=False)
V.save(fig, os.path.join(F, "fig_rank_bump"))

# ---------------------------------------------------------------- Fig. remedies: change of per-class recall vs CE
rm = pd.read_csv(os.path.join(S, "rq3_remedies_by_model.csv"))
REM = ["wce", "focal", "la", "sampler", "posthoc_la"]
REMN = {"wce": "Weighted CE", "focal": "Focal", "la": "Logit-adj. loss", "sampler": "Balanced sampling", "posthoc_la": "Post-hoc adj."}
for setn in ["test", "isic"]:
    q = rm[(rm.set == setn) & rm.metric.str.startswith("recall_")]
    if not len(q):
        continue
    mat = q.groupby(["remedy", "metric"])["diff"].mean().unstack().reindex(index=REM, columns=["recall_" + c for c in CLASSES])
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    lim = np.nanmax(np.abs(mat.values))
    imh = ax.imshow(mat.values, cmap=V.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, "%+.2f" % v, ha="center", va="center", fontsize=7, color=V.INK)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(REM))); ax.set_yticklabels([REMN[r] for r in REM]); ax.grid(False)
    cb = fig.colorbar(imh, ax=ax, fraction=0.03); cb.set_label("Recall change vs CE")
    ax.set_title("Mean over ConvNeXt-T, Swin-T, CoAtNet-0 (%s)" % ("lesion-split test folds" if setn == "test" else "ISIC 2018 test"), loc="left")
    V.save(fig, os.path.join(F, "fig_remedy_recall_%s" % setn))

# ---------------------------------------------------------------- Fig. remedies: BACC vs accuracy trade-off
pooled = pd.read_csv(os.path.join(S, "rq3_remedies_by_model.csv"))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
for ax, setn, ttl in [(axs[0], "test", "Lesion-split test folds"), (axs[1], "isic", "ISIC 2018 test")]:
    for k, m in enumerate(["convnext_tiny.fb_in1k", "swin_tiny_patch4_window7_224.ms_in1k", "coatnet_0_rw_224.sw_in1k"]):
        q = pooled[(pooled.set == setn) & (pooled.model == m)]
        ba = q[q.metric == "balanced_accuracy"].set_index("remedy")
        ac = q[q.metric == "accuracy"].set_index("remedy")
        if not len(ba):
            continue
        r0 = [r for r in REM if r in ba.index][0]  # the CE value is the same in every remedy row
        ax.scatter(ac.loc[r0, "ce"], ba.loc[r0, "ce"], marker="*", s=90, color=FAM_COL[MODELS[m]], edgecolor="white", zorder=3)
        for r in REM:
            if r in ba.index:
                ax.annotate("", xy=(ac.loc[r, "remedy_value"], ba.loc[r, "remedy_value"]), xytext=(ac.loc[r, "ce"], ba.loc[r, "ce"]),
                            arrowprops=dict(arrowstyle="->", color=FAM_COL[MODELS[m]], lw=0.8, alpha=0.7))
                ax.text(ac.loc[r, "remedy_value"], ba.loc[r, "remedy_value"], " " + {"wce": "W", "focal": "F", "la": "L", "sampler": "S", "posthoc_la": "P"}[r],
                        fontsize=7, color=V.INK, va="center")
    ax.set_title(ttl, loc="left"); ax.set_xlabel("Accuracy")
axs[0].set_ylabel("Balanced accuracy")
# marker legend is in the LaTeX caption
V.save(fig, os.path.join(F, "fig_remedy_tradeoff"))

# ---------------------------------------------------------------- Fig. seen vs unseen lesions (image split)
su = os.path.join(S, "rq1_seen_unseen.csv")
if os.path.exists(su):
    g = pd.read_csv(su)
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    xs = np.arange(len(CLASSES) + 1)
    labs = CLASSES + ["all"]
    for off, flag, col, lab in [(-0.18, True, V.CAT[1], "Lesion has another image in training"), (0.18, False, V.CAT[0], "Lesion not in training")]:
        v = [g[(g.dx == c) & (g.lesion_in_train == flag)].accuracy.values for c in labs]
        ax.bar(xs + off, [x[0] if len(x) else np.nan for x in v], width=0.34, color=col, label=lab, edgecolor="white", linewidth=1)
    ax.set_xticks(xs); ax.set_xticklabels(labs); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy (image-split test folds)")
    ax.legend(loc="lower left", ncol=2)
    V.save(fig, os.path.join(F, "fig_seen_unseen"))
print("FIGURES_DONE")
