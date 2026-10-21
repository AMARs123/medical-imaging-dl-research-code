"""Tables of the four descriptive analyses of Paper 09 (from extra.py outputs). Usage: python make_tables_extra.py <Paper_09 dir>
  tab_dilation  AURC x100 of single-pass foreground entropy for each widening d, clean vs corrupted/shifted sets
  tab_enssize   ensemble Dice and AURC x100 of ensemble agreement for K = 2..5 members (mean over all subsets)
  tab_types     failure types and the median percentile each score gives them (100 = most doubtful)
  tab_ops       share of all failures that remain among the accepted images at 80% and 90% coverage
"""
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
O = os.path.join(R, "results", "processed")
T = os.path.join(R, "paper", "tables")
DATA = {"kvasir": "Kvasir-SEG", "ham": "HAM10000"}
NAME = {"ent_det": "Entropy, image", "ent_det_fg": "Entropy, foreground", "ent_det_fg_e": "Entropy, foreground + empty flag$^\\dagger$",
        "unc_pdice_mc": "MC agreement", "unc_pdice_tta": "TTA agreement", "ref_irreg": "Irregularity (reference)"}


def write(name, s):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(s)
    print("wrote", name)


# ---------------------------------------------------------------- widening ablation
d = pd.read_csv(os.path.join(O, "ablation_dilation.csv"))
d["clean"] = np.where(d.set == "test", "clean", "other")
g = d.groupby(["d", "data", "clean"]).aurc.mean().unstack(["data", "clean"]) * 100
order = ["image", "0", "1", "2", "5", "10", "20"]
rows = []
for k in order:
    lab = "Whole image" if k == "image" else ("%s px%s" % (k, " (used)" if k == "5" else ""))
    rows.append("%s & %s \\\\" % (lab, " & ".join("%.2f" % g.loc[k, (dd, c)] for dd in ["kvasir", "ham"] for c in ["clean", "other"])))
rc = d[d.d == "5"].recomputed_vs_stored.min()
write("tab_dilation", r"""\begin{table}[!htbp]
\centering\small
\caption{Ablation of the foreground widening $d$ (Eq.~\ref{eq:ufg}): AURC$\times$100 of single-pass foreground entropy, recomputed from the stored 8-bit probability maps (correlation with the stored scores at $d=5$: at least %.3f per run and set), mean over 10 runs and over the clean test set or the corrupted and other-camera sets. ``Whole image'' is $U_{\mathrm{img}}$ from the same maps; $d=0$ averages over the predicted mask only.}
\label{tab:dilation}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Kvasir-SEG} & \multicolumn{2}{c}{HAM10000} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Averaged over & Clean $\downarrow$ & Corrupted $\downarrow$ & Clean $\downarrow$ & Corrupted, camera $\downarrow$ \\
\midrule
""" % rc + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- ensemble size
k = pd.read_csv(os.path.join(O, "ensemble_size.csv"))
q = pd.read_csv(os.path.join(O, "quality.csv"))
f = pd.read_csv(os.path.join(O, "failure.csv"))
f = f[(f.set != "val") & (f.member != "ens")]
q = q[(q.set != "val") & (q.member != "ens")]
rows = ["1 (single network) & %s \\\\" % " & ".join("%.3f & --" % q[q.data == dd].dice_det.mean() for dd in ["kvasir", "ham"])]
for K in range(2, 6):
    h = k[k.K == K]
    rows.append("%d & %s \\\\" % (K, " & ".join("%.3f & %.2f" % (h[h.data == dd].dice.mean(), 100 * h[h.data == dd].aurc.mean()) for dd in ["kvasir", "ham"])))
rows.append("\\midrule\n1, TTA agreement (8 views) & %s \\\\" % " & ".join(
    "%.3f & %.2f" % (q[q.data == dd].dice_tta.mean(), 100 * f[(f.data == dd) & (f.score == "unc_pdice_tta")].aurc.mean()) for dd in ["kvasir", "ham"]))
write("tab_enssize", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Ensemble size: Dice of the ensemble mean and AURC$\times$100 of ensemble agreement (Eq.~\ref{eq:agree}) for every subset of $K$ of the five seeds, averaged over subsets, encoders and evaluation sets. The last row gives the single network with TTA agreement for comparison.}
\label{tab:enssize}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Kvasir-SEG} & \multicolumn{2}{c}{HAM10000} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Members $K$ & Dice $\uparrow$ & AURC $\downarrow$ & Dice $\uparrow$ & AURC $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- failure types
t = pd.read_csv(os.path.join(O, "failure_types.csv"))
TYPES = {"empty": "Empty mask", "misplaced": "Misplaced", "under": "Under-segmented", "over": "Over-segmented"}
SC = ["ent_det", "ent_det_fg_e", "unc_pdice_mc", "unc_pdice_tta", "ref_irreg"]
rows = []
for dd in ["kvasir", "ham"]:
    for ty in TYPES:
        h = t[(t.data == dd) & (t.type == ty)]
        if len(h) == 0:
            continue
        rows.append("%s & %s & %d & %s \\\\" % (DATA[dd] if ty == "empty" else "", TYPES[ty], len(h), " & ".join("%.0f" % h["pct_" + s].median() for s in SC)))
    rows.append("\\addlinespace")
write("tab_types", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Failure types (Dice $<$ 0.5 of the single-pass prediction, all runs and evaluation sets) and the median percentile that each score gives them within their evaluation set (100 = most doubtful, 50 = no better than chance). Empty: no foreground predicted; misplaced: precision and recall both below 0.5; under-segmented: precision $\geq$ 0.5, recall $<$ 0.5; over-segmented: precision $<$ 0.5, recall $\geq$ 0.5. $^\dagger$Post-hoc variant.}
\label{tab:types}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrccccc}
\toprule
Data set & Type & $n$ & Entropy, image & Fg entropy + flag$^\dagger$ & MC agreement & TTA agreement & Irregularity \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- operating points
op = pd.read_csv(os.path.join(O, "operating_points.csv"))
rows = []
for s in ["ref_irreg", "ent_det", "ent_det_fg", "ent_det_fg_e", "unc_pdice_mc", "unc_pdice_tta"]:
    vals = []
    for dd in ["kvasir", "ham"]:
        for cov in [0.8, 0.9]:
            h = op[(op.data == dd) & (op.score == s) & (op.coverage == cov)]
            vals.append("%.0f" % (100 * h.failures_accepted.sum() / h.failures_total.sum()))
    rows.append("%s & %s \\\\" % (NAME[s], " & ".join(vals)))
write("tab_ops", r"""\begin{table}[!htbp]
\centering\small
\caption{Operating points: share (\%) of all failures (Dice $<$ 0.5) that remain among the accepted images when the most trusted 80\% or 90\% of the images of each evaluation set are accepted, pooled over runs and evaluation sets. An uninformative score would leave about 80\% and 90\%. $^\dagger$Post-hoc variant.}
\label{tab:ops}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Kvasir-SEG} & \multicolumn{2}{c}{HAM10000} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Score & 80\% $\downarrow$ & 90\% $\downarrow$ & 80\% $\downarrow$ & 90\% $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
