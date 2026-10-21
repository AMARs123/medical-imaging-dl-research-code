"""Result tables and figures of leaf-disease-lab-to-field from results/processed/runs.csv and results/statistics (analyze.py).
Tables: tab_split (RQ1: PlantVillage accuracy under both splits, inflation, PlantDoc accuracy and gap, per backbone),
tab_conditions (RQ2, RQ3: every condition of groups B and C against the reference of the same backbone).
Figures: fig_split (PlantVillage and PlantDoc accuracy by split and backbone), fig_conditions (accuracy difference to the
reference with 95% bootstrap intervals), fig_fewshot (PlantDoc accuracy against field images per class).
Usage: python make_results.py <leaf-disease-lab-to-field dir>"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

V.setup()
R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
P = os.path.join(R, "results", "processed")
T = os.path.join(R, "paper", "tables")
F = os.path.join(R, "figures", "final")
os.makedirs(F, exist_ok=True)
MN = {"resnet50.a1_in1k": "ResNet-50", "efficientnet_b0.ra_in1k": "EfficientNet-B0", "convnext_tiny.fb_in1k": "ConvNeXt-T",
      "vit_small_patch14_dinov2.lvd142m": "DINOv2 ViT-S"}
SRC = {"color": "colour", "segmented": "segmented", "mixed": "mixed"}
runs = pd.read_csv(os.path.join(P, "runs.csv"), keep_default_na=False, na_values=[""])


def w(name, body):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(body)


def ms(v, f="%.1f"):
    v = 100 * pd.Series(v).dropna()  # all accuracies and differences in percent
    return (f + " (" + f + ")") % (v.mean(), v.std()) if len(v) > 1 else ((f % v.mean()) if len(v) else "--")


def cond_name(r):
    s = "%s images" % SRC[r["source"]]
    if r["aug"] == "strong":
        s += ", strong aug."
    if int(r["fewshot"]):
        s += ", $k=%d$" % int(r["fewshot"])
    return s


# ---------------------------------------------------------------- RQ1: split inflation and gap
A = runs[runs.phase == "A"]
rows = []
for m in MN:
    g = A[A.model == m]
    if not len(g):
        continue
    c = []
    for sp in ["random", "leaf"]:
        c.append(ms(g[(g.split == sp) & (g.set == "pv_test")].acc, "%.2f"))
    pvq = g[g.set == "pv_test"].pivot_table(index="seed", columns="split", values=["acc", "acc_leafclasses"])
    if ("acc", "random") in pvq and ("acc", "leaf") in pvq:
        c.append(ms((pvq[("acc", "random")] - pvq[("acc", "leaf")]).dropna(), "%.2f"))
        c.append(ms((pvq[("acc_leafclasses", "random")] - pvq[("acc_leafclasses", "leaf")]).dropna(), "%.2f"))
    else:
        c += ["--", "--"]
    for sp in ["random", "leaf"]:
        c.append(ms(g[(g.split == sp) & (g.set == "pd_all")].acc))
    q = g[g.split == "leaf"].pivot_table(index="seed", columns="set", values="acc")
    c.append(ms((q.pv_test - q.pd_all).dropna()) if {"pv_test", "pd_all"} <= set(q.columns) else "--")
    rows.append("%s & %s \\\\" % (MN[m], " & ".join(c)))
w("tab_split", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{PlantVillage test accuracy under the random and the leaf-grouped split, the inflation $\iota$ (random minus leaf, paired by seed; also on the classes with leaf identity only), PlantDoc accuracy (Eq.~\ref{eq:acc}) and the laboratory-to-field gap $\Delta$ (Eq.~\ref{eq:gap}) under the leaf split, in percent (inflation and gap in percentage points); mean (standard deviation) over three seeds. All models fine-tuned on colour images with base augmentation.}
\label{tab:split}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{2}{c}{PlantVillage Acc $\uparrow$} & \multicolumn{2}{c}{Inflation $\iota$} & \multicolumn{2}{c}{PlantDoc Acc $\uparrow$} & Gap $\Delta$ \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Backbone & random & leaf & all classes & leaf classes & random & leaf & leaf split \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- RQ2, RQ3: conditions
cf = os.path.join(S, "conditions.csv")
cond = pd.read_csv(cf) if os.path.exists(cf) and os.path.getsize(cf) > 5 else pd.DataFrame()
pdr = runs[(runs.set == "pd_all") & (runs.split == "leaf")]
rows = []
for m in ["convnext_tiny.fb_in1k", "vit_small_patch14_dinov2.lvd142m"]:
    ref = pdr[(pdr.model == m) & (pdr.source == "color") & (pdr.aug == "base") & (pdr.fewshot == 0) & (pdr.freeze == 0)]
    if not len(ref):
        continue
    rows.append("\\multicolumn{7}{l}{\\textit{%s}} \\\\" % MN[m])
    rows.append("Reference: colour images, fine-tuned & %s & %s & %s & %s & -- & -- \\\\" % (ms(ref.acc), ms(ref.bacc), ms(ref.retention), ms(ref.acc_within_crop)))
    if len(cond):
        cm = cond[cond.model == m].copy()
        cm["order"] = cm.freeze * 100 + cm.fewshot + cm.source.map({"color": 0, "segmented": 1, "mixed": 2}) * 0.1 + (cm.aug == "strong") * 0.01
        for r in cm.sort_values("order").to_dict("records"):
            g = pdr[(pdr.model == m) & (pdr.source == r["source"]) & (pdr.aug == r["aug"]) & (pdr.fewshot == r["fewshot"]) & (pdr.freeze == r["freeze"])]
            lab = ("Linear probe: " if r["freeze"] else "") + cond_name(r)
            rows.append("%s & %s & %s & %s & %s & $%+.1f$ & [$%+.1f$, $%+.1f$] \\\\" % (lab, ms(g.acc), ms(g.bacc), ms(g.retention), ms(g.acc_within_crop),
                                                                                      100 * r["d_acc_common"], 100 * r["ci_low"], 100 * r["ci_high"]))
    rows.append("\\midrule")
if rows and rows[-1] == "\\midrule":
    rows = rows[:-1]
w("tab_conditions", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{PlantDoc results of every condition under the leaf split: accuracy, balanced accuracy, crop retention and within-crop accuracy (Eq.~\ref{eq:retention}) in percent; mean (standard deviation) over three seeds. $\delta$ (percentage points): accuracy difference to the reference of the same backbone on the PlantDoc images evaluated by both (images used as field training examples excluded), with the 95\% paired bootstrap interval (Eq.~\ref{eq:boot}). $k$: labelled PlantDoc images per class added to training.}
\label{tab:conditions}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Condition & Acc $\uparrow$ & BAcc $\uparrow$ & Retention $\uparrow$ & Within-crop $\uparrow$ & $\delta$ & 95\% interval \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- figures
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
ms_ = list(MN)
for k, (s, ttl) in enumerate([("pv_test", "PlantVillage test"), ("pd_all", "PlantDoc")]):
    ax = axs[k]
    for j, sp in enumerate(["random", "leaf"]):
        for i, m in enumerate(ms_):
            v = A[(A.model == m) & (A.split == sp) & (A.set == s)].acc
            if len(v):
                x = i + (j - 0.5) * 0.3
                ax.plot(np.full(len(v), x), v, "o", color=V.CAT[j], ms=4.5, mfc=V.CAT[j] if sp == "random" else "white",
                        label=("%s split" % sp) if i == 0 else None)
    ax.set_xticks(range(len(ms_)), [MN[m] for m in ms_], rotation=15)
    ax.set_title(ttl)
    ax.set_ylabel("Accuracy")
axs[0].legend(fontsize=8)
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_split"))

if len(cond):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    labs, y = [], 0
    for j, m in enumerate(["convnext_tiny.fb_in1k", "vit_small_patch14_dinov2.lvd142m"]):
        cm = cond[cond.model == m].copy()
        cm["order"] = cm.freeze * 100 + cm.fewshot + cm.source.map({"color": 0, "segmented": 1, "mixed": 2}) * 0.1 + (cm.aug == "strong") * 0.01
        for r in cm.sort_values("order").to_dict("records"):
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], "-", color=V.CAT[j], lw=1.6)
            ax.plot(r["d_acc_common"], y, "o", color=V.CAT[j], ms=5, label=MN[m] if not any(l == MN[m] for l in ax.get_legend_handles_labels()[1]) else None)
            labs.append(("probe: " if r["freeze"] else "") + cond_name(r).replace("$", "").replace("k=", "k = "))
            y += 1
        y += 0.6
    ax.axvline(0, color=V.INK2, lw=0.8)
    ticks = [i + 0.6 * (i >= len(cond[cond.model == "convnext_tiny.fb_in1k"])) for i in range(len(labs))]
    ax.set_yticks(ticks, labs, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("PlantDoc accuracy difference to the reference, $\\delta$")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    V.save(fig, os.path.join(F, "fig_conditions"))

fs = pdr[(pdr.source == "color") & (pdr.aug == "base")]
if fs.fewshot.nunique() > 1:
    fig, ax = plt.subplots(figsize=(3.46, 2.8))
    j = 0
    for m in ["convnext_tiny.fb_in1k", "vit_small_patch14_dinov2.lvd142m"]:
        for fr, ls in [(0, "-"), (1, "--")]:
            q = fs[(fs.model == m) & (fs.freeze == fr)].groupby("fewshot").acc.agg(["mean", "std"])
            if len(q) > 1:
                ax.errorbar(q.index, q["mean"], yerr=q["std"], fmt=ls + "o", color=V.CAT[0 if "convnext" in m else 1], ms=4.5,
                            mfc="white" if fr else None, capsize=2, label="%s, %s" % (MN[m], "probe" if fr else "fine-tuned"))
                j += 1
    ax.set_xticks([0, 5, 20])
    ax.set_xlabel("Labelled field images per class, $k$")
    ax.set_ylabel("PlantDoc accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    V.save(fig, os.path.join(F, "fig_fewshot"))
print("RESULTS_DONE")
