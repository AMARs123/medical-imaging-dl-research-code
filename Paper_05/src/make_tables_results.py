"""Result tables of Paper 05 from results/statistics (stats.py). Writes paper/tables/tab_main.tex (single models: QWK, SER,
clinical measures on DDR test and DeepDRiD), tab_sources.tex (SER and AUSER of single, TTA and ensemble), tab_paired.tex
(paired comparison with CE and bootstrap intervals).
Usage: python make_tables_results.py <Paper_05 dir>"""
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
T = os.path.join(R, "paper", "tables")
FN = {"ce": "CE", "sord": "SORD", "reg": "REG", "coral": "CORAL", "corn": "CORN"}
FORMS = ["ce", "sord", "reg", "coral", "corn"]
MN = {"resnet50.a1_in1k": "ResNet-50", "convnext_tiny.fb_in1k": "ConvNeXt-T"}


def w(name, body):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(body)


def ms(r, k, d=3):
    m, s = r.get(k + "_mean", np.nan), r.get(k + "_std", np.nan)
    if not np.isfinite(m):
        return "--"
    return ("%.*f (%.*f)" % (d, m, d, s)) if np.isfinite(s) else "%.*f" % (d, m)


po = pd.read_csv(os.path.join(S, "pooled.csv"))
rows = []
for s, lab in [("test", "DDR test"), ("ext", "DeepDRiD")]:
    rows.append(r"\multicolumn{6}{l}{\textit{%s}} \\" % lab)
    for f in FORMS:
        q = po[(po.form == f) & (po.source == "single") & (po.set == s)]
        if not len(q):
            continue
        r = q.iloc[0].to_dict()
        rows.append("%s & %s & %s & %s & %s & %s \\\\" % (FN[f], ms(r, "qwk"), ms(r, "severe_error_rate"), ms(r, "referable_sens"), ms(r, "referable_spec"), ms(r, "vt_sens")))
w("tab_main", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Single models: quadratic weighted kappa (Eq.~\ref{eq:qwk}), severe-error rate (Eq.~\ref{eq:ser}) and clinical measures (Eq.~\ref{eq:clin}) on the DDR test set and on DeepDRiD, mean (standard deviation) over two backbones and three seeds. $\uparrow$/$\downarrow$: higher/lower is better.}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Formulation & QWK $\uparrow$ & SER $\downarrow$ & Sens.\ referable $\uparrow$ & Spec.\ referable $\uparrow$ & Sens.\ vision-threat.\ $\uparrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

rows = []
for s, lab in [("test", "DDR test"), ("ext", "DeepDRiD")]:
    rows.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % lab)
    for f in FORMS:
        cells = []
        for src in ["single", "tta", "ens"]:
            q = po[(po.form == f) & (po.source == src) & (po.set == s)]
            r = q.iloc[0].to_dict() if len(q) else {}
            cells.append("%.3f" % r["severe_error_rate_mean"] if r else "--")
            cells.append("%.3f" % r["auser_mean"] if r and np.isfinite(r.get("auser_mean", np.nan)) else "--")
        rows.append("%s & %s \\\\" % (FN[f], " & ".join(cells)))
w("tab_sources", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Uncertainty sources: severe-error rate (SER, Eq.~\ref{eq:ser}) of the combined prediction and area under the severe-error abstention curve (AUSER, Eq.~\ref{eq:auser}, ordering by the ordinal variance $U$ of Eq.~\ref{eq:ovar}) for single models, test-time augmentation (TTA, Eq.~\ref{eq:views}) and the three-seed ensemble (Eq.~\ref{eq:ens}); means over backbones (and seeds for single and TTA). REG has no single-model uncertainty. $\downarrow$: lower is better.}
\label{tab:sources}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Single} & \multicolumn{2}{c}{TTA} & \multicolumn{2}{c}{Ensemble} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Formulation & SER $\downarrow$ & AUSER $\downarrow$ & SER $\downarrow$ & AUSER $\downarrow$ & SER $\downarrow$ & AUSER $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

pc = pd.read_csv(os.path.join(S, "paired_ce.csv"))
bt = pd.read_csv(os.path.join(S, "boot_ser.csv"))
rows = []
for s, lab in [("test", "DDR test"), ("ext", "DeepDRiD")]:
    rows.append(r"\multicolumn{5}{l}{\textit{%s}} \\" % lab)
    for f in FORMS[1:]:
        q = pc[(pc.set == s) & (pc.form == f)]
        if not len(q):
            continue
        r = q.iloc[0]
        cis = []
        for m in MN:
            b = bt[(bt.set == s) & (bt.form == f) & (bt.model == m)]
            cis.append("[$%+.3f$, $%+.3f$]" % (b.ci_low.iloc[0], b.ci_high.iloc[0]) if len(b) else "--")
        rows.append("%s & $%+.3f$ (%d/%d) & $%+.3f$ (%d/%d) & %s \\\\" % (FN[f], r.d_ser, r.ser_better, r.n_pairs, r.d_qwk, r.qwk_better, r.n_pairs, " & ".join(cis)))
w("tab_paired", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Ordinal formulations against cross-entropy, single models. Mean paired difference over the six (backbone, seed) pairs with the number of pairs in which the formulation is better, and 95\% paired bootstrap intervals of the SER difference for each backbone (DDR: images resampled; DeepDRiD: patients resampled) (Eq.~\ref{eq:boot}; per-image error averaged over seeds). Negative SER differences favour the formulation.}
\label{tab:paired}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Formulation & $\Delta$SER (better) & $\Delta$QWK (better) & SER CI ResNet-50 & SER CI ConvNeXt-T \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print("TABLES_DONE")
