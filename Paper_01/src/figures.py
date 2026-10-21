"""Paper figures from stored results only. Usage: python figures.py <Paper_01 dir>
Writes PDF (vector) and PNG (300 dpi) to figures/final/. A figure whose inputs do not exist yet is skipped.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

R = sys.argv[1]
OUT = os.path.join(R, "figures", "final")
os.makedirs(OUT, exist_ok=True)
ST = os.path.join(R, "results", "statistics")
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
runs = pd.read_csv(os.path.join(R, "results", "processed", "all_runs.csv"))
if "split_seed" not in runs.columns:
    runs["split_seed"] = 0
runs["split_seed"] = runs["split_seed"].fillna(0).astype(int)

# categorical slots (validated reference palette, light mode), fixed order, never cycled
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
RULE = {"slice": "Image-level\n(no grouping)", "authors": "Authors'\nfolds", "pid": "Grouped by\nID string", "root": "Grouped by\nID root"}
MODELS_C = ["resnet50", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


# ---------------------------------------------------------------- Fig: fold-formation rules
p = os.path.join(ST, "A_pooled.csv")
if os.path.exists(p):
    a = pd.read_csv(p)
    rules = ["slice", "authors", "pid", "root"]
    fig, ax = plt.subplots(figsize=(3.46, 2.6))
    for k, m in enumerate(["resnet50", "vit_small_patch16_224"]):
        d = a[a.model == m].set_index("protocol").loc[rules]
        x = np.arange(4) + (k - 0.5) * 0.22
        ax.errorbar(x, d.macro_f1, yerr=[d.macro_f1 - d.ci_lo, d.ci_hi - d.macro_f1], fmt="o", ms=5, color=COL[m],
                    capsize=0, elinewidth=1.5, label=NICE[m])
        dd = os.path.join(ST, "D_partition.csv")
        if os.path.exists(dd):  # spread over re-drawn grouped partitions
            sp = pd.read_csv(dd)
            for j, pr in enumerate(rules):
                v = sp[(sp.model == m) & (sp.protocol == pr)]["macro_f1"]
                if len(v) > 1:
                    ax.scatter(np.full(len(v), j + (k - 0.5) * 0.22 + 0.08), v, s=8, color=COL[m], alpha=0.5, lw=0)
    ax.set_xticks(range(4), [RULE[r] for r in rules])
    ax.set_ylabel("Macro-F1 (pooled out-of-fold)")
    ax.legend(loc="upper right")
    save(fig, "fig_fold_rules")

# ---------------------------------------------------------------- Fig: architectures, accuracy vs localisation
dg = os.path.join(ST, "evidence_diagnostics.csv")
if os.path.exists(dg):
    dgn = pd.read_csv(dg)
    dgn = dgn[(dgn.protocol == "root") & ((dgn.split_seed.fillna(0) == 0) & (dgn.dilate.fillna(1) == 1) & (dgn.train_frac.fillna(1) == 1))]
    dgn["pointing_lift"] = dgn.pointing_cell / dgn.pointing_cell_chance
    base = runs[(runs.protocol == "root") & runs.main]
    fig, ax = plt.subplots(figsize=(3.46, 2.8))
    order = list(NICE)
    for k, m in enumerate(order):
        for lam, mk in [(0.0, "o"), (0.25, "^")]:
            f1 = base[(base.model == m) & (base.lambda_sal == lam)].macro_f1
            pl = dgn[(dgn.model == m) & (dgn.lambda_sal == lam)].pointing_lift
            if len(f1) == 0 or len(pl) == 0:
                continue
            ax.scatter(pl.mean(), f1.mean(), s=36, marker=mk, color=COL[m], edgecolor="white", linewidth=0.8, zorder=3,
                       label=NICE[m] if lam == 0 else None)
            if lam == 0:  # direct label, so colour is not the only way to tell the models apart
                ax.annotate(NICE[m], (pl.mean(), f1.mean()), xytext=(4, -2 if k % 2 else 3), textcoords="offset points",
                            fontsize=8, color=INK)
            if lam == 0.25:
                f0 = base[(base.model == m) & (base.lambda_sal == 0)].macro_f1.mean()
                p0 = dgn[(dgn.model == m) & (dgn.lambda_sal == 0)].pointing_lift.mean()
                ax.annotate("", (pl.mean(), f1.mean()), (p0, f0), arrowprops=dict(arrowstyle="->", color=COL[m], lw=1))
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.text(1.05, ax.get_ylim()[0], " chance", color=MUTED, va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Pointing / chance (evidence-map resolution, log)")
    ax.set_ylabel("Macro-F1 (mean over runs)")
    ax.scatter([], [], marker="o", color=MUTED, label=r"$\lambda=0$")
    ax.scatter([], [], marker="^", color=MUTED, label=r"$\lambda=0.25$")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[-2:], l[-2:], fontsize=8.5, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig_arch_localisation")

# ---------------------------------------------------------------- Fig: saliency weight ablation
cs = runs[(runs.protocol == "root") & runs.main & runs.model.isin(MODELS_C)]
if (cs.lambda_sal > 0).any():
    fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.3))
    for k, m in enumerate(MODELS_C):
        g = cs[cs.model == m].groupby("lambda_sal")
        lam = np.array(sorted(g.groups))
        xs = np.arange(len(lam))
        for ax, met in zip(axs, ["macro_f1", "loc_mass_inside", "ece"]):
            mu, sd = g[met].mean().loc[lam], g[met].std().loc[lam]
            ax.plot(xs, mu, "-", marker=MARK[m], ms=4.5, color=COL[m], label=NICE[m])
            ax.fill_between(xs, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
    for ax, t in zip(axs, ["Macro-F1", "Evidence inside the tumour", "Expected calibration error"]):
        ax.set_xticks(range(5), ["0", "0.25", "0.5", "1", "2"])
        ax.set_xlabel(r"Penalty weight $\lambda$")
        ax.set_title(t, fontsize=8, color=INK)
    axs[0].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_lambda_ablation")

# ---------------------------------------------------------------- Fig: example evidence maps
images = np.load(os.path.join(R, "data", "processed", "images.npy"), mmap_mode="r")
masks = np.load(os.path.join(R, "data", "processed", "masks.npy"), mmap_mode="r")
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"), dtype={"pid": str, "root": str})


def exp_of(model, lam):
    r = reg[(reg.model == model) & (reg.protocol == "root") & (reg.fold == 0) & (reg.seed == 0) & (reg.lambda_sal == lam)]
    if "split_seed" in reg:
        r = r[r.split_seed.fillna(0) == 0]
    e = r.exp_id.values[0] if len(r) else None
    return e if e and os.path.exists(os.path.join(R, "experiments", e, "results.json")) else None


cols = [(m, lam) for m in ["resnet50", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"] for lam in (0.0, 0.25)]
if all(exp_of(*c) for c in cols):
    preds = {c: pd.read_csv(os.path.join(R, "experiments", exp_of(*c), "predictions.csv")) for c in cols}
    evs = {c: np.load(os.path.join(R, "experiments", exp_of(*c), "evidence_test.npy")) for c in cols}
    p0 = preds[cols[0]]
    ok = np.all([(preds[c].label == preds[c].pred).values for c in cols], 0)  # correctly classified by all eight
    pick = []
    for lab in range(3):  # one image per class, fixed rule: correct for all models, median tumour size of its class
        cand = np.where(ok & (p0.label.values == lab))[0]
        frac = meta.mask_fraction.values[p0.array_index.values[cand]]
        pick.append(cand[np.argsort(frac)[len(frac) // 2]])
    fig, axs = plt.subplots(3, 1 + len(cols), figsize=(7.09, 2.9))
    for r, i in enumerate(pick):
        j = p0.array_index.values[i]
        img, msk = images[j].astype(float), masks[j]
        axs[r, 0].imshow(img, cmap="gray")
        axs[r, 0].contour(msk, levels=[0.5], colors=["#e34948"], linewidths=0.8)
        axs[r, 0].set_ylabel(["Meningioma", "Glioma", "Pituitary"][p0.label.values[i]], fontsize=8)
        for c, key in enumerate(cols):
            pr = preds[key]
            row = np.where(pr.array_index.values == j)[0][0]
            e = evs[key][row, pr.pred.values[row]].astype(np.float32)
            h = F.interpolate(torch.from_numpy(np.maximum(e, 0))[None, None], size=img.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
            ax = axs[r, c + 1]
            ax.imshow(img, cmap="gray")
            ax.imshow(h / (h.max() + 1e-9), cmap="magma", alpha=0.55, vmin=0, vmax=1)
            ax.contour(msk, levels=[0.5], colors=["#1baf7a"], linewidths=0.6)
            if r == 0:
                SHORT = {"resnet50": "ResNet", "convnext_tiny": "ConvNeXt", "vit_small_patch16_224": "ViT-S", "swin_tiny_patch4_window7_224": "Swin-T"}
                ax.set_title("%s\n%s" % (SHORT.get(key[0], NICE[key[0]]), r"$\lambda$=0" if key[1] == 0 else r"$\lambda$=0.25"), fontsize=8, color=INK)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.03, hspace=0.04)
    save(fig, "fig_examples")

# ---------------------------------------------------------------- Fig: perturbation tests
ex = [(r, os.path.join(R, "experiments", r.exp_id, "explain.json")) for r in reg.itertuples()]
ex = [(r, json.load(open(p))) for r, p in ex if os.path.exists(p)]
for r, s in ex:  # signed-evidence deletion curve, when computed (explain_signed.py)
    ps = os.path.join(R, "experiments", r.exp_id, "explain_signed.json")
    s["curve_signed"] = json.load(open(ps))["curve_signed"] if os.path.exists(ps) else None
if ex:
    rows = [dict(model=r.model, protocol=r.protocol, lambda_sal=r.lambda_sal, split_seed=getattr(r, "split_seed", 0), dilate=getattr(r, "dilate", 1), train_frac=getattr(r, "train_frac", 1.0), **{k: v for k, v in s.items()})
            for r, s in ex]
    E = pd.DataFrame(rows)
    E = E[(E.protocol == "root") & ((E.split_seed.fillna(0) == 0) & (E.dilate.fillna(1) == 1) & (E.train_frac.fillna(1) == 1))]
    E.drop(columns=["fractions", "curve_evidence", "curve_random", "curve_signed"]).to_csv(os.path.join(ST, "explain_runs.csv"), index=False)
    use = "curve_signed" if E.curve_signed.notna().all() else "curve_evidence"
    fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.5))
    fr = np.array(ex[0][1]["fractions"]) * 100
    for k, m in enumerate(MODELS_C):
        for lam, ls in [(0.0, "-"), (0.25, "--")]:
            d = E[(E.model == m) & (E.lambda_sal == lam)]
            if len(d):
                axs[0].plot(fr, np.mean(np.stack(d[use].values), 0), ls, color=COL[m], lw=1.5, marker=MARK[m], ms=3.5, markevery=2,
                            label="%s %s" % (NICE[m], r"$\lambda$=0" if lam == 0 else r"$\lambda$=0.25"))
        d = E[(E.model == m) & (E.lambda_sal == 0)]
        if len(d):
            axs[0].plot(fr, np.mean(np.stack(d.curve_random.values), 0), ":", color=COL[m], lw=1)
    axs[0].set_xlabel("Patches removed (%)")
    axs[0].set_ylabel("Probability of predicted class")
    axs[0].set_title("Deletion by %s evidence (dotted: random)" % ("signed" if use == "curve_signed" else "positive"), fontsize=8, color=INK)
    axs[0].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    allm = [m for m in NICE if m in set(E.model)]
    x = np.arange(len(allm))
    b = E[E.lambda_sal == 0].groupby("model")[["drop_tumour", "drop_control"]].mean().loc[allm]
    axs[1].bar(x - 0.2, b.drop_tumour, 0.38, color=C[0], label="Tumour removed")
    axs[1].bar(x + 0.2, b.drop_control, 0.38, color=C[1], hatch="///", edgecolor="white", label="Same shape elsewhere")
    axs[1].set_xticks(x, [NICE[m] for m in allm], rotation=35, ha="right")
    axs[1].set_ylabel("Mean drop in probability")
    axs[1].set_title(r"Occlusion test ($\lambda$=0)", fontsize=8, color=INK)
    axs[1].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_perturbation")
