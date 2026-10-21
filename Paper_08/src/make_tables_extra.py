"""Tables of the analyses added after the main results (extra.py, extra_q2.py) and the appendix interval table (analyze.py).
Writes paper/tables/tab_quant.tex, tab_domain.tex, tab_ci.tex and figures/final/fig_reliability.pdf/png.
Usage: python make_tables_extra.py <Paper_08 dir>"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

V.setup()
R = sys.argv[1]
T = os.path.join(R, "paper", "tables")
F = os.path.join(R, "figures", "final")
MN = {"mobilenetv3_small_100.lamb_in1k": "MobileNetV3-S", "mobilenetv3_large_100.ra_in1k": "MobileNetV3-L", "efficientnet_b0.ra_in1k": "EfficientNet-B0",
      "resnet18.a1_in1k": "ResNet-18", "resnet50.a1_in1k": "ResNet-50", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
ORDER = list(MN)


def w(name, body):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(body)


# ---------------------------------------------------------------- quantisation variants
q = pd.read_csv(os.path.join(R, "results", "quant_ablation.csv"))
ef = pd.read_csv(os.path.join(R, "results", "efficiency.csv"))
fp = ef[ef.seed == 0].set_index("model")
VAR = [("default_256", "Default"), ("default_1024", "1,024 calib."), ("conv_linear_only_256", "Conv/linear only"), ("no_depthwise_256", "No depthwise")]
rows = []
for m in ORDER:
    cells = ["%.3f" % fp.loc[m, "fp32_bacc_taco"]]
    for v, _ in VAR:
        r = q[(q.model == m) & (q.variant == v)]
        if not len(r):
            cells.append("n/a")
        elif r.status.iloc[0] != "ok":
            cells.append("failed$^{a}$")
        else:
            cells.append("%.3f" % r.bacc_taco.iloc[0])
    rows.append("%s & %s \\\\" % (MN[m], " & ".join(cells)))
w("tab_quant", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Balanced accuracy on the TACO test set of the pooled seed-0 models in full precision and after 8-bit quantisation with four recipes (Section~\ref{sec:extra}): the default static recipe with 256 or 1,024 calibration images, only convolutions and linear layers quantised, and all layers except depthwise convolutions quantised. n/a: the backbone has no depthwise convolution. $^{a}$The converter rejects mixed precision inside a fused convolution--activation pattern. Chance level is 0.20. $\uparrow$: higher is better.}
\label{tab:quant}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Backbone & fp32 $\uparrow$ & """ + " & ".join("%s $\\uparrow$" % lab for _, lab in VAR) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- domain separability
d = pd.read_csv(os.path.join(R, "results", "domain_auc.csv")).drop_duplicates(["model", "train_on"])
pv = d.pivot(index="model", columns="train_on", values="domain_auc")
rows = ["%s & %s \\\\" % (MN[m], " & ".join("%.3f" % pv.loc[m, c] if c in pv.columns and pd.notna(pv.loc[m, c]) else "--" for c in ["trashnet", "taco", "both"]))
        for m in ORDER if m in pv.index]
w("tab_domain", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Domain separability of the penultimate features (Eq.~\ref{eq:auc}): cross-validated AUC of a logistic-regression classifier that tells TrashNet from TACO test images, seed-0 models. 1: the sources are perfectly separated in feature space; 0.5: not separated.}
\label{tab:domain}
\begin{tabular}{lccc}
\toprule
Backbone & Trained on TrashNet & Trained on TACO & Trained on both \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- bootstrap intervals (appendix)
ci = pd.read_csv(os.path.join(R, "results", "processed", "ci.csv"))
rows = []
for m in ORDER:
    cells = []
    for tr, te in [("trashnet", "TN"), ("trashnet", "TA"), ("taco", "TA"), ("both", "TN"), ("both", "TA")]:
        r = ci[(ci.model == m) & (ci.train_on == tr) & (ci.test == te)]
        cells.append("[%.2f, %.2f]" % (r.ci_low.iloc[0], r.ci_high.iloc[0]) if len(r) else "--")
    rows.append("%s & %s \\\\" % (MN[m], " & ".join(cells)))
w("tab_ci", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{95\%% bootstrap intervals (Eq.~\ref{eq:boot}; 2,000 resamples of test photographs) of the seed-mean balanced accuracy. Training source $\to$ test set. TACO intervals are wide because its test set holds %d photographs.}
\label{tab:ci}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Backbone & TN $\to$ TN & TN $\to$ TA & TA $\to$ TA & Both $\to$ TN & Both $\to$ TA \\
\midrule
""" % int(ci[ci.test == "TA"].n_photos.iloc[0]) + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- reliability diagram on TACO
rel = pd.read_csv(os.path.join(R, "results", "reliability.csv"))
fig, ax = plt.subplots(figsize=(3.46, 3.0))
ax.plot([0, 1], [0, 1], "--", color=V.INK2, lw=0.8)
for (tr, lab), col, mk in zip([("trashnet", "TrashNet only"), ("taco", "TACO only"), ("both", "Both")], V.CAT[:3], ["o", "s", "^"]):
    g = rel[(rel.train_on == tr) & (rel.n >= 20)]
    ax.plot(g.conf, g.acc, "-", marker=mk, color=col, ms=4.5, lw=1.5, label=lab)
ax.set_xlabel("Mean confidence in bin")
ax.set_ylabel("Accuracy in bin")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(title="Trained on", fontsize=8, title_fontsize=8, loc="upper left")
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_reliability"))
# ---------------------------------------------------------------- calibration repair and selective prediction
cf = os.path.join(R, "results", "calibration.csv")
sf = os.path.join(R, "results", "selective.csv")
if os.path.exists(cf) and os.path.exists(sf):
    c = pd.read_csv(cf)
    rows = []
    for tr, lab in [("trashnet", "TrashNet only"), ("taco", "TACO only"), ("both", "Both")]:
        cells = []
        for te in ["TN", "TA"]:
            g = c[(c.train_on == tr) & (c.test == te)]
            cells += ["%.2f" % g.ece_before.mean(), "%.2f" % g.ece_after.mean()]
        rows.append("%s & %.2f & %s \\\\" % (lab, c[c.train_on == tr].groupby("exp_id").T.first().median(), " & ".join(cells)))
    w("tab_calib", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Calibration before and after temperature scaling fitted on the validation images of the training source(s) (Eq.~\ref{eq:temp}): median fitted temperature and expected calibration error (Eq.~\ref{eq:ece}) on both test sets, mean over backbones and seeds. $\downarrow$: lower is better.}
\label{tab:calib}
\begin{tabular}{lccccc}
\toprule
 & & \multicolumn{2}{c}{TrashNet test} & \multicolumn{2}{c}{TACO test} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Trained on & Median $T$ & ECE $\downarrow$ & ECE scaled $\downarrow$ & ECE $\downarrow$ & ECE scaled $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    s = pd.read_csv(sf)
    s = s[s.test == "TA"].groupby(["train_on", "coverage"]).sel_acc.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(3.46, 2.9))
    for (tr, lab), col, mk in zip([("trashnet", "TrashNet only"), ("taco", "TACO only"), ("both", "Both")], V.CAT[:3], ["o", "s", "^"]):
        g = s[s.train_on == tr].sort_values("coverage")
        ax.errorbar(g.coverage, g["mean"], yerr=g["std"], marker=mk, color=col, ms=4.5, lw=1.5, capsize=2, label=lab)
    ax.set_xlabel("Coverage (share of objects accepted)")
    ax.set_ylabel("Accuracy on accepted objects")
    ax.invert_xaxis()
    ax.legend(title="Trained on", fontsize=8, title_fontsize=8, loc="lower left")
    fig.tight_layout()
    V.save(fig, os.path.join(F, "fig_selective"))
print("EXTRA_TABLES_DONE")
