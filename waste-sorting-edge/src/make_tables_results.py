"""Result tables of waste-sorting-edge from results/statistics and results/processed (stats.py, analyze.py).
Writes paper/tables/tab_main.tex, tab_pool.tex, tab_recall.tex and, when efficiency results exist, tab_eff.tex.
Usage: python make_tables_results.py <waste-sorting-edge dir>"""
import os
import sys

import pandas as pd

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
T = os.path.join(R, "paper", "tables")
MN = {"mobilenetv3_small_100.lamb_in1k": "MobileNetV3-S", "mobilenetv3_large_100.ra_in1k": "MobileNetV3-L", "efficientnet_b0.ra_in1k": "EfficientNet-B0",
      "resnet18.a1_in1k": "ResNet-18", "resnet50.a1_in1k": "ResNet-50", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
ORDER = list(MN)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]


def w(name, body):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(body)


su = pd.read_csv(os.path.join(S, "summary.csv")).set_index(["model", "train_on", "test"])
cols = [("trashnet", "TN"), ("trashnet", "TA"), ("taco", "TA"), ("taco", "TN"), ("both", "TN"), ("both", "TA")]
rows = []
for m in ORDER:
    cells = []
    for tr, te in cols:
        if (m, tr, te) in su.index:
            r = su.loc[(m, tr, te)]
            cells.append("%.3f (%.3f)" % (r.bacc_mean, r.bacc_sd))
        else:
            cells.append("--")
    rows.append("%s & %s \\\\" % (MN[m], " & ".join(cells)))
w("tab_main", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Balanced accuracy (Eq.~\ref{eq:bacc}) on the two test sets for every backbone and training source: mean (standard deviation) over three seeds. TN: TrashNet studio photographs; TA: TACO litter crops. $\uparrow$: higher is better.}
\label{tab:main}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Trained on TN} & \multicolumn{2}{c}{Trained on TA} & \multicolumn{2}{c}{Trained on both} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Backbone & TN test $\uparrow$ & TA test $\uparrow$ & TA test $\uparrow$ & TN test $\uparrow$ & TN test $\uparrow$ & TA test $\uparrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

gap = pd.read_csv(os.path.join(S, "gap.csv")).set_index("model")
pool = pd.read_csv(os.path.join(S, "pool.csv"))
pt = pd.read_csv(os.path.join(S, "pool_tests.csv")).set_index("test")
pm = pool.groupby(["model", "test"]).pool_effect.agg(["mean", "std"])
rows = []
for m in ORDER:
    if m not in gap.index:
        continue
    g = gap.loc[m]
    c = ["%.3f (%.3f)" % (g.gap_mean, g.gap_sd), "%.3f (%.3f)" % (g.reverse_gap_mean, g.reverse_gap_sd)]
    for t in ["TN", "TA"]:
        c.append("$%+.3f$ (%.3f)" % tuple(pm.loc[(m, t)]) if (m, t) in pm.index and pd.notna(pm.loc[(m, t)]["std"]) else "--")
    rows.append("%s & %s \\\\" % (MN[m], " & ".join(c)))
foot = []
for t in ["TN", "TA"]:
    if t in pt.index:
        r = pt.loc[t]
        foot.append("%s: mean $%+.3f$, %d of %d pairs above zero, $r_{\\mathrm{rb}}=%.2f$, Wilcoxon $p=%.3g$" % (
            {"TN": "TrashNet test", "TA": "TACO test"}[t], r.mean_diff, r.pairs_positive, r.n_pairs, r.rank_biserial, r.p_wilcoxon))
w("tab_pool", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Wild gap and pooling effect: mean (standard deviation) over three seeds. Wild gap $\Delta$ (Eq.~\ref{eq:gap}): balanced accuracy of the TrashNet-trained model on TrashNet minus on TACO; reverse gap: TACO-trained model on TACO minus on TrashNet. Pooling effect $\pi_t$ (Eq.~\ref{eq:pool}): pooled training minus training on the test source alone. Paired over backbones and seeds: """ + "; ".join(foot) + r""".}
\label{tab:pool}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
Backbone & Wild gap $\Delta$ & Reverse gap & $\pi_{\mathrm{TN}}$ & $\pi_{\mathrm{TA}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

runs = pd.read_csv(os.path.join(R, "results", "processed", "runs.csv"))
q = runs[runs.test == "TA"].groupby("train_on")[["recall_" + c for c in CLASSES]].mean()
rows = []
for tr, lab in [("trashnet", "TrashNet only"), ("taco", "TACO only"), ("both", "Both")]:
    if tr in q.index:
        rows.append("%s & %s \\\\" % (lab, " & ".join("%.2f" % q.loc[tr, "recall_" + c] for c in CLASSES)))
w("tab_recall", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Recall per material on the TACO test set by training source, mean over the six backbones and three seeds. $\uparrow$: higher is better.}
\label{tab:recall}
\begin{tabular}{lccccc}
\toprule
Trained on & Cardboard $\uparrow$ & Glass $\uparrow$ & Metal $\uparrow$ & Paper $\uparrow$ & Plastic $\uparrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

ef = os.path.join(S, "efficiency.csv")
if os.path.exists(ef):
    e = pd.read_csv(ef).set_index("model")
    rows = []
    for m in ORDER:
        if m not in e.index:
            continue
        r = e.loc[m]
        i8 = ("%.1f" % r.int8_ms_1t if r.int8_ok and "int8_ms_1t" in r and pd.notna(r.int8_ms_1t) else "failed")
        dq = ("$%+.3f$" % r.dq_TA if r.int8_ok and "dq_TA" in r and pd.notna(r.dq_TA) else "--")
        rows.append("%s & %.1f & %.2f & %.1f & %.1f & %s & %s & %.3f & %s \\\\" % (MN[m], r.params_m, r.gmacs, r.fp32_ms_1t, r.fp32_ms_4t, i8, dq,
                                                                              r.fp32_bacc_taco, "yes" if r.pareto_fp32 else ""))
    w("tab_eff", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Cost of the pooled models on the server CPU (Eq.~\ref{eq:lat}): parameters, multiply--accumulate operations (GMACs) for one $224\times224$ image, median latency at one and four threads in full precision, latency at one thread after 8-bit quantisation, accuracy cost of quantisation on TACO $\Delta_q$ (Eq.~\ref{eq:quant}), TACO balanced accuracy in full precision, and membership of the Pareto front (Eq.~\ref{eq:pareto}) of one-thread latency against TACO balanced accuracy. Means over three seeds. $\downarrow$: lower is better.}
\label{tab:eff}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrc}
\toprule
Backbone & Params (M) & GMACs & ms, 1 thr.\ $\downarrow$ & ms, 4 thr.\ $\downarrow$ & int8 ms $\downarrow$ & $\Delta_q$ (TA) $\downarrow$ & BAcc TA $\uparrow$ & Pareto \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print("TABLES_DONE")
