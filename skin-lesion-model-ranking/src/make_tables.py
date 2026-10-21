"""LaTeX tables of skin-lesion-model-ranking, generated only from stored files (nothing typed by hand).

Usage: python make_tables.py <skin-lesion-model-ranking dir> [data|results|all]
data    : tab_data (class composition), tab_models (backbones; parameter counts from results.json)
results : result tables (written once the statistics exist)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, MODELS, NAMES, SHORT  # noqa: E402

R = sys.argv[1]
what = sys.argv[2] if len(sys.argv) > 2 else "all"
T = os.path.join(R, "paper", "tables")
os.makedirs(T, exist_ok=True)
INS = os.path.join(R, "data", "inspection")


def pct(x, d=1):
    return "{:.{d}f}\\%".format(100 * x, d=d)


def write(name, s):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(s)
    print("wrote", name)


if what in ("data", "all"):
    ct = pd.read_csv(os.path.join(INS, "class_table.csv")).set_index("dx")
    mult = pd.read_csv(os.path.join(INS, "lesion_multiplicity.csv")).set_index("n")
    leak = pd.read_csv(os.path.join(INS, "image_split_leakage.csv")).groupby("dx")[["test_images", "test_images_with_lesion_in_train"]].sum()
    rows = []
    for c in CLASSES:
        r = ct.loc[c]
        multi = int(mult.loc[mult.index >= 2, c].sum())
        lk = leak.loc[c, "test_images_with_lesion_in_train"] / leak.loc[c, "test_images"]
        rows.append("%s (\\texttt{%s}) & %d (%s) & %d & %d & %.2f & %d (%s) & %s \\\\" % (
            NAMES[c], c, r.ham_images, pct(r.ham_share), r.ham_lesions, multi, r.ham_images_per_lesion,
            r.test_images, pct(r.test_share), pct(lk)))
    tot_lk = leak.test_images_with_lesion_in_train.sum() / leak.test_images.sum()
    tot_multi = int(mult.loc[mult.index >= 2].values.sum())
    rows.append("\\midrule\nAll & %d (100\\%%) & %d & %d & %.2f & %d (100\\%%) & %s \\\\" % (
        ct.ham_images.sum(), ct.ham_lesions.sum(), tot_multi, ct.ham_images.sum() / ct.ham_lesions.sum(), ct.test_images.sum(), pct(tot_lk)))
    write("tab_data", r"""\begin{table}[!htbp]
\centering\small
\caption{Class composition of HAM10000 (development set) and of the 1,511 available ISIC 2018 test images. ``Lesions $\geq$2'' counts lesions with two or more images. The last column gives, for a random stratified image split (Section~\ref{sec:protocols}), the share of test-fold images whose lesion also has an image in the training folds (five folds pooled).}
\label{tab:data}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{4}{c}{HAM10000} & ISIC 2018 test & Image split \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}
Class & Images (share) & Lesions & Lesions $\geq$2 & Img/lesion & Images (share) & lesion seen \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # backbones: parameter counts from the first finished run of each model
    reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
    rows = []
    for m, fam in MODELS.items():
        n = None
        for e in reg[reg.model == m].exp_id:
            f = os.path.join(R, "experiments", e, "results.json")
            if os.path.exists(f):
                n = json.load(open(f))["n_params"]
                break
        rows.append("%s & %s & \\texttt{%s} & %s \\\\" % (fam, SHORT[m], m.replace("_", "\\_"), ("%.1f" % (n / 1e6)) if n else "PENDING"))
    write("tab_models", r"""\begin{table}[!htbp]
\centering\small
\caption{Backbones. All use ImageNet-1k pretrained weights from \texttt{timm}; parameters include the new 7-class layer.}
\label{tab:models}
\begin{tabular}{lllr}
\toprule
Family & Model & \texttt{timm} identifier & Parameters (M) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # run plan: counts from the registry (fixed before training)
    GROUP = {"A": "Backbone comparison", "B": "Imbalance remedies", "C": "Seed repeats"}
    LOSS = {"ce": "CE", "wce": "WCE", "focal": "focal", "la": "LA", "sampler": "sampler"}
    rows = []
    for ph, g in reg.groupby("phase"):
        rows.append("%s & %s & %d & %s & %s & %s & %d \\\\" % (
            ph, GROUP.get(ph, ph), g.model.nunique(), ", ".join(sorted(g.protocol.unique())),
            ", ".join(LOSS.get(x, x) for x in sorted(g.loss.unique())), ", ".join(str(s) for s in sorted(g.seed.unique())), len(g)))
    write("tab_plan", r"""\begin{table}[!htbp]
\centering\small
\caption{Registered runs (five folds each), fixed before training. CE: cross-entropy; WCE: weighted cross-entropy; LA: logit-adjusted loss.}
\label{tab:plan}
\begin{tabular}{llrlllr}
\toprule
Group & Purpose & Backbones & Protocols & Losses & Seeds & Runs \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
All & & & & & & %d \\
\bottomrule
\end{tabular}
\end{table}
""" % len(reg))
    # measured cost: training time and peak GPU memory per backbone (cross-entropy runs that have finished)
    rows = []
    for m, fam in MODELS.items():
        sec, mb, n = [], [], 0
        for e in reg[(reg.model == m) & (reg.loss == "ce")].exp_id:
            f = os.path.join(R, "experiments", e, "results.json")
            if os.path.exists(f):
                d = json.load(open(f))
                sec.append(d["train_seconds"])
                mb.append(d.get("peak_gpu_mb") or np.nan)
                n += 1
        if n:
            rows.append("%s & %s & %d & %.1f $\\pm$ %.1f & %.0f \\\\" % (fam, SHORT[m], n, np.mean(sec) / 60, np.std(sec, ddof=1) / 60 if n > 1 else 0,
                                                                        np.nanmax(mb)))
        else:
            rows.append("%s & %s & 0 & PENDING & PENDING \\\\" % (fam, SHORT[m]))
    write("tab_budget", r"""\begin{table}[!htbp]
\centering\small
\caption{Measured training cost per run (25 epochs, batch 32, mixed precision, one GPU share of an NVIDIA RTX 5000 Ada), over all finished cross-entropy runs. Times include validation after every epoch but not the final inference on the test fold and the ISIC test set.}
\label{tab:budget}
\begin{tabular}{llrrr}
\toprule
Family & Model & Runs & Time per run (min) $\downarrow$ & Peak GPU memory (MB) $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
