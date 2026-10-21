"""Additional paper figures from stored predictions/metrics (CPU). Usage: python figures_extra.py <Paper_01 dir>
Each figure is skipped when its inputs are missing. Output: figures/final/*.pdf|png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

R = sys.argv[1]
OUT = os.path.join(R, "figures", "final")
ST = os.path.join(R, "results", "statistics")
os.makedirs(OUT, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
for c, v in [("split_seed", 0), ("dilate", 1), ("train_frac", 1.0)]:
    reg[c] = reg[c].fillna(v) if c in reg else v
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"), dtype={"pid": str, "root": str})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 2,
                     "legend.frameon": False, "pdf.fonttype": 42})
NICE = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "efficientnet_b0": "EfficientNet-B0",
        "convnext_tiny": "ConvNeXt-T", "mobilevit_s": "MobileViT-S", "vit_small_patch16_224": "ViT-S/16", "deit_small_patch16_224": "DeiT-S/16",
        "swin_tiny_patch4_window7_224": "Swin-T"}
# one fixed colour per model in every figure (validated palette; MobileViT-S added 2026-10-16)
COL = {"resnet50": "#2a78d6", "densenet121": "#eb6834", "efficientnet_b0": "#1baf7a", "convnext_tiny": "#eda100",
       "mobilevit_s": "#a8509a", "vit_small_patch16_224": "#e87ba4", "deit_small_patch16_224": "#008300",
       "swin_tiny_patch4_window7_224": "#6250d6"}

# one marker shape per model, so that models stay distinguishable without colour (rule VS-03)
MARK = {"resnet50": "o", "densenet121": "s", "efficientnet_b0": "D", "convnext_tiny": "P", "mobilevit_s": "X",
        "vit_small_patch16_224": "^", "deit_small_patch16_224": "v", "swin_tiny_patch4_window7_224": "*"}
MC = ["resnet50", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]
CLS = ["Meningioma", "Glioma", "Pituitary"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def main_runs(model, lam=0.0, seed=None, phase=None):
    r = reg[(reg.model == model) & (reg.protocol == "root") & (reg.lambda_sal == lam) & (reg.split_seed == 0)
            & (reg.dilate == 1) & (reg.train_frac == 1.0)]
    if seed is not None:
        r = r[r.seed == seed]
    return [e for e in r.exp_id if os.path.exists(os.path.join(R, "experiments", e, "results.json"))]


def pooled(model, lam, seed=0):
    ex = main_runs(model, lam, seed)
    if len(ex) != 5:
        return None
    return pd.concat([pd.read_csv(os.path.join(R, "experiments", e, "predictions.csv")) for e in ex])


# ---------------------------------------------------------------- dataset overview
fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.2))
cnt = meta.label.value_counts().sort_index()
axs[0].bar(range(3), cnt.values, color=C[:3], width=0.6)
for i, v in enumerate(cnt.values):
    axs[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8, color=INK)
axs[0].set_xticks(range(3), CLS)
axs[0].set_ylabel("Slices")
axs[0].set_title("Slices per class", fontsize=8, color=INK)
per = meta.groupby("pid").size()
axs[1].hist(per.values, bins=range(0, per.max() + 3, 2), color=C[0], edgecolor="white")
axs[1].set_xlabel("Slices per identifier string")
axs[1].set_ylabel("Identifiers")
axs[1].set_title("Slices per identifier (233)", fontsize=8, color=INK)
var = meta.groupby("root").pid.nunique()
multi = var[var > 1]
axs[2].bar(["1 variant", "2", "3", "4"], [int((var == k).sum()) for k in [1, 2, 3, 4]], color=C[1], width=0.6)
for i, k in enumerate([1, 2, 3, 4]):
    axs[2].text(i, int((var == k).sum()), str(int((var == k).sum())), ha="center", va="bottom", fontsize=8, color=INK)
axs[2].set_ylabel("Identifier roots")
axs[2].set_title("Identifier strings per root (209 roots)", fontsize=8, color=INK)
fig.tight_layout()
save(fig, "fig_dataset")

# ---------------------------------------------------------------- confusion matrices, ROC and PR for the best model
best = "convnext_tiny"
P = {lam: pooled(best, lam) for lam in (0.0, 0.25)}
if all(v is not None for v in P.values()):
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.4))
    for ax, (lam, d) in zip(axs, P.items()):
        cm = confusion_matrix(d.label, d.pred, labels=[0, 1, 2])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "%d\n%.1f%%" % (cm[i, j], 100 * cmn[i, j]), ha="center", va="center", fontsize=8.5,
                        color="white" if cmn[i, j] > 0.6 else INK)
        ax.set_xticks(range(3), CLS, rotation=20)
        ax.set_yticks(range(3), CLS)
        ax.grid(False)
        ax.set_xlabel("Predicted")
        ax.set_title(r"%s, $\lambda$=%g" % (NICE[best], lam), fontsize=8, color=INK)
    axs[0].set_ylabel("True")
    fig.tight_layout()
    save(fig, "fig_confusion")

    fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.6))
    for k in range(3):
        for lam, ls in [(0.0, "-"), (0.25, "--")]:
            d = P[lam]
            yk, pk = (d.label.values == k).astype(int), d["p%d" % k].values
            fpr, tpr, _ = roc_curve(yk, pk)
            pr, rc, _ = precision_recall_curve(yk, pk)
            lab = r"%s $\lambda$=%g" % (CLS[k], lam)
            axs[0].plot(fpr, tpr, ls, color=C[k], lw=1.4, label="%s (AUC %.3f)" % (lab, auc(fpr, tpr)))
            axs[1].plot(rc, pr, ls, color=C[k], lw=1.4, label="%s (AP %.3f)" % (lab, auc(rc, pr)))
    axs[0].plot([0, 1], [0, 1], ":", color=MUTED, lw=1)
    axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[0].set_title("ROC, one class vs rest (pooled, seed 0)", fontsize=8, color=INK)
    axs[1].set_title("Precision-recall (pooled, seed 0)", fontsize=8, color=INK)
    for ax in axs:
        ax.legend(fontsize=8, loc="lower right" if ax is axs[0] else "lower left")
    axs[0].set_xlim(-0.01, 0.4); axs[0].set_ylim(0.6, 1.005)
    axs[1].set_xlim(0.6, 1.005); axs[1].set_ylim(0.6, 1.005)
    fig.tight_layout()
    save(fig, "fig_roc_pr")

# ---------------------------------------------------------------- reliability diagrams
fig, axs = plt.subplots(1, 4, figsize=(7.09, 2.0), sharey=True)
ok = False
for ax, m in zip(axs, MC):
    for lam, col in [(0.0, C[0]), (0.25, C[1])]:
        d = pooled(m, lam)
        if d is None:
            continue
        ok = True
        pr = d[["p0", "p1", "p2"]].values
        conf, corr = pr.max(1), (pr.argmax(1) == d.label.values)
        bins = np.linspace(1 / 3, 1, 9)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
        xs = [conf[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).sum() >= 10]
        ys = [corr[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).sum() >= 10]
        ax.plot(xs, ys, "-o", ms=3, color=col, lw=1.3, label=r"$\lambda$=%g" % lam)
    ax.plot([1 / 3, 1], [1 / 3, 1], ":", color=MUTED, lw=1)
    ax.set_title(NICE[m], fontsize=8, color=INK)
    ax.set_xlabel("Confidence")
axs[0].set_ylabel("Accuracy")
axs[0].legend(fontsize=8.5)
if ok:
    fig.tight_layout()
    save(fig, "fig_reliability")
else:
    plt.close(fig)

# ---------------------------------------------------------------- training curves
fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.3))
for k, m in enumerate(MC):
    for lam, ls in [(0.0, "-"), (0.25, "--")]:
        ex = main_runs(m, lam)
        if not ex:
            continue
        mt = pd.concat([pd.read_csv(os.path.join(R, "experiments", e, "metrics.csv")).assign(exp=e) for e in ex])
        g = mt.groupby("epoch")
        lab = r"%s $\lambda$=%g" % (NICE[m], lam)
        axs[0].plot(g.train_ce.mean(), ls, color=COL[m], lw=1.2, marker=MARK[m], ms=3, markevery=5, label=lab)
        axs[1].plot(g.val_macro_f1.mean(), ls, color=COL[m], lw=1.2, marker=MARK[m], ms=3, markevery=5)
        axs[2].plot(g.train_outside.mean(), ls, color=COL[m], lw=1.2, marker=MARK[m], ms=3, markevery=5)
axs[0].set_yscale("log")
for ax, t in zip(axs, ["Training cross-entropy", "Validation macro-F1", "Training outside share $S$"]):
    ax.set_xlabel("Epoch")
    ax.set_title(t, fontsize=8, color=INK)
axs[1].set_ylim(0.75, 1.0)
axs[0].legend(fontsize=8, ncol=1)
fig.tight_layout()
save(fig, "fig_training_curves")

# ---------------------------------------------------------------- localisation by tumour size and class
fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.4))
q = np.quantile(meta.mask_fraction_resized, [0, 0.25, 0.5, 0.75, 1])
for k, m in enumerate(MC):
    for lam, ls in [(0.0, "-"), (0.25, "--")]:
        ex = main_runs(m, lam)
        if not ex:
            continue
        d = pd.concat([pd.read_csv(os.path.join(R, "experiments", e, "predictions.csv")) for e in ex])
        d["size"] = meta.mask_fraction_resized.values[d.array_index.values]
        d["bin"] = np.clip(np.digitize(d["size"], q[1:-1]), 0, 3)
        axs[0].plot(range(4), d.groupby("bin").mass_inside.mean().reindex(range(4)), ls, marker=MARK[m], ms=3.5, color=COL[m], lw=1.2,
                    label=r"%s $\lambda$=%g" % (NICE[m], lam))
        if lam == 0.25:
            continue
        pc = d.groupby("label").mass_inside.mean()
        axs[1].bar(np.arange(3) + (k - 1.5) * 0.2, pc.values, 0.19, color=COL[m], hatch=["", "///", "...", "xx"][k], edgecolor="white", label=NICE[m])
axs[0].set_xticks(range(4), ["Q1\n(smallest)", "Q2", "Q3", "Q4\n(largest)"])
axs[0].set_xlabel("Tumour-area quartile")
axs[0].set_ylabel("Evidence inside the tumour")
axs[0].legend(fontsize=8, ncol=2)
axs[1].set_xticks(range(3), CLS)
axs[1].set_ylabel(r"Evidence inside the tumour ($\lambda$=0)")
axs[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "fig_loc_size_class")

# ---------------------------------------------------------------- per-class F1 by architecture
runs = pd.read_csv(os.path.join(R, "results", "processed", "all_runs.csv"))
rows = []
for m in NICE:
    for e in main_runs(m, 0.0):
        r = json.load(open(os.path.join(R, "experiments", e, "results.json")))["test"]["per_class_f1"]
        rows.append(dict(model=m, **{CLS[i]: r[i] for i in range(3)}))
if rows:
    pf = pd.DataFrame(rows).groupby("model")[CLS].agg(["mean", "std"]).reindex(list(NICE))
    fig, ax = plt.subplots(figsize=(7.09, 2.3))
    x = np.arange(len(NICE))
    for i, c in enumerate(CLS):
        ax.bar(x + (i - 1) * 0.27, pf[(c, "mean")], 0.25, yerr=pf[(c, "std")], color=C[i], hatch=["", "///", "..."][i], edgecolor="white", label=c,
               error_kw=dict(lw=0.8, ecolor=MUTED))
    ax.set_xticks(x, [NICE[m] for m in NICE], rotation=20, ha="right")
    ax.set_ylim(0.7, 1.06)
    ax.set_ylabel(r"F1 (mean $\pm$ SD over runs)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "fig_per_class_f1")

# ---------------------------------------------------------------- lift heatmap over lambda
dg = os.path.join(ST, "evidence_diagnostics.csv")
if os.path.exists(dg):
    g = pd.read_csv(dg)
    for c, v in [("split_seed", 0), ("dilate", 1), ("train_frac", 1.0)]:
        g[c] = g[c].fillna(v) if c in g else v
    g = g[(g.protocol == "root") & (g.split_seed == 0) & (g.dilate == 1) & (g.train_frac == 1.0) & g.model.isin(MC)]
    g["lift"] = g.pointing_cell / g.pointing_cell_chance
    h = g.groupby(["model", "lambda_sal"]).lift.mean().unstack().reindex(MC)
    fig, ax = plt.subplots(figsize=(3.46, 1.9))
    im = ax.imshow(h.values, cmap="Blues", aspect="auto", vmin=0)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if not np.isnan(h.values[i, j]):
                ax.text(j, i, "%.1f" % h.values[i, j], ha="center", va="center", fontsize=8,
                        color="white" if h.values[i, j] > 0.6 * np.nanmax(h.values) else INK)
    ax.set_xticks(range(h.shape[1]), ["%g" % v for v in h.columns])
    ax.set_yticks(range(h.shape[0]), [NICE[m] for m in h.index])
    ax.set_xlabel(r"Penalty weight $\lambda$")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Pointing / chance", shrink=0.9)
    save(fig, "fig_lift_heatmap")

# ---------------------------------------------------------------- ablations E (dilation) and F (data fraction)
if len(runs) and "dilate" in reg:
    rr = runs.merge(reg[["exp_id", "dilate", "train_frac"]], on="exp_id", how="left", suffixes=("", "_reg"))
    dcol = "dilate_reg" if "dilate_reg" in rr else "dilate"
    fcol = "train_frac_reg" if "train_frac_reg" in rr else "train_frac"
    rr[dcol] = rr[dcol].fillna(1); rr[fcol] = rr[fcol].fillna(1.0)
    rr = rr[(rr.protocol == "root") & (rr.split_seed.fillna(0) == 0)]
    E = rr[(rr[fcol] == 1.0) & (rr.lambda_sal == 0.25) & rr.model.isin(["convnext_tiny", "vit_small_patch16_224"])]
    if E[dcol].nunique() >= 3:
        fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.2))
        for k, m in enumerate(["convnext_tiny", "vit_small_patch16_224"]):
            g = E[E.model == m].groupby(dcol)
            for ax, met in zip(axs, ["macro_f1", "loc_mass_inside"]):
                ax.errorbar(g[met].mean().index + (k - 0.5) * 0.08, g[met].mean(), yerr=g[met].std(), fmt="-", marker=MARK[m], ms=4.5, color=COL[m],
                            capsize=0, lw=1.5, label=NICE[m])
        for ax, t in zip(axs, ["Macro-F1", "Evidence inside the tumour"]):
            ax.set_xticks([0, 1, 2]); ax.set_xlabel("Mask widening $r$ (cells)"); ax.set_title(t, fontsize=8, color=INK)
        axs[0].legend(fontsize=8.5)
        fig.tight_layout()
        save(fig, "fig_ablation_dilation")
    Fd = rr[(rr[dcol] == 1) & rr.model.isin(["convnext_tiny", "vit_small_patch16_224"]) & rr.lambda_sal.isin([0.0, 0.25])]
    if Fd[fcol].nunique() >= 3:
        fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.3))
        for k, m in enumerate(["convnext_tiny", "vit_small_patch16_224"]):
            for lam, ls in [(0.0, "-"), (0.25, "--")]:
                g = Fd[(Fd.model == m) & (Fd.lambda_sal == lam)].groupby(fcol)
                for ax, met in zip(axs, ["macro_f1", "loc_mass_inside"]):
                    ax.errorbar(100 * g[met].mean().index, g[met].mean(), yerr=g[met].std(), fmt=ls, marker=MARK[m], ms=4.5, color=COL[m],
                                capsize=0, lw=1.3, label=r"%s $\lambda$=%g" % (NICE[m], lam))
        for ax, t in zip(axs, ["Macro-F1", "Evidence inside the tumour"]):
            ax.set_xticks([25, 50, 100]); ax.set_xlabel("Training identifier roots used (%)"); ax.set_title(t, fontsize=8, color=INK)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        save(fig, "fig_ablation_datafrac")

# ---------------------------------------------------------------- robustness
rows = []
for r in reg.itertuples():
    p = os.path.join(R, "experiments", r.exp_id, "robust.json")
    if os.path.exists(p):
        for c, v in json.load(open(p)).items():
            rows.append(dict(model=r.model, lambda_sal=r.lambda_sal, corruption=c, **v))
if rows:
    Rb = pd.DataFrame(rows)
    Rb.to_csv(os.path.join(ST, "robustness_runs.csv"), index=False)
    corr = [c for c in Rb.corruption.unique()]
    fig, ax = plt.subplots(figsize=(7.09, 2.5))
    x = np.arange(len(corr))
    for k, m in enumerate(MC):
        for lam in (0.0, 0.25):
            mk = MARK[m]
            g = Rb[(Rb.model == m) & (Rb.lambda_sal == lam)].groupby("corruption").macro_f1.mean().reindex(corr)
            ax.plot(x + (k - 1.5) * 0.12, g.values, mk, ms=4.5, color=COL[m], ls="none",
                    label=r"%s $\lambda$=%g" % (NICE[m], lam), markerfacecolor=COL[m] if lam == 0 else "white")
    ax.set_xticks(x, [c.replace("_", " ") for c in corr], rotation=20)
    ax.set_ylabel("Macro-F1 (mean of runs)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "fig_robustness")
