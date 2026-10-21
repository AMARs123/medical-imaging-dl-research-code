"""Result tables and figures of chest-xray-pneumonia-shift from results/processed/runs.csv and results/statistics (analyze.py).
Read every CSV with keep_default_na=False ("NA" = NIH adults).
Tables: tab_transfers (RQ1: AUC at source and target and the transfer gap per backbone, pooled over folds and seeds),
tab_contrast (source contrast and pooling on the same NP images, Holm-adjusted), tab_threshold (RQ2: sensitivity,
specificity and PPV at the source threshold on every group), tab_remedies (RQ2: remedies against the base condition, when
group B / AdaBN exist), tab_robust (RQ3: relative AUC per corruption, when robust.json files exist).
Figures: fig_transfers (AUC of every run on each group, by training source), fig_remedies, fig_robust (when data exist).
Usage: python make_results.py <chest-xray-pneumonia-shift dir>"""
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
rd = lambda f: pd.read_csv(os.path.join(S, f), keep_default_na=False, na_values=[""]) if os.path.exists(os.path.join(S, f)) and os.path.getsize(os.path.join(S, f)) > 5 else None  # noqa: E731
MN = {"resnet50.a1_in1k": "ResNet-50", "densenet121.tv_in1k": "DenseNet-121", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
GN = {"G": "Guangzhou children (G)", "NA": "NIH adults (NA)", "NP": "NIH children (NP)"}
SN = {"G": "G", "NA": "NA", "G,NA": "G + NA"}
runs = pd.read_csv(os.path.join(P, "runs.csv"), keep_default_na=False, na_values=[""])
base = runs[(runs.aug == "base") & (runs.prep == "raw")]


def w(name, body):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(body)


def ms(v, f="%.3f"):
    v = pd.Series(v).dropna()
    out = (f + " (" + f + ")") % (v.mean(), v.std()) if len(v) > 1 else ((f % v.mean()) if len(v) else "--")
    return out.replace("-", "$-$") if out != "--" else out


def pv(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else "%.3f" % p)


def sg(x):
    return "--" if pd.isna(x) else "$%+.3f$" % x


# ---------------------------------------------------------------- RQ1: transfers
gap = rd("gaps.csv")
tr = rd("transfers.csv")
TR = [("G", "NP", "Hospital only"), ("NA", "NP", "Age only"), ("G", "NA", "Both"), ("NA", "G", "Both (reverse)")]
rows = []
for s, t, name in TR:
    q = gap[(gap.source == s) & (gap.target == t)] if gap is not None else pd.DataFrame()
    if not len(q):
        continue
    c = [("%.3f" % q[q.model == m].gap.mean()).replace("-", "$-$") for m in MN]
    tq = tr[(tr.source == s) & (tr.target == t)].iloc[0]
    rows.append("%s & %s$\\to$%s & %s & %s & %s & %s & %s \\\\" % (name, s, t, ms(q.auc_source), ms(q.auc_target), ms(q.gap), pv(tq.p_vs_zero),
                                                                    " & ".join(c)))
w("tab_transfers", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Transfers of the base condition (Eq.~\ref{eq:transfers}): AUC on held-out patients of the source and on all images of the target, and the transfer gap $\Delta$ (Eq.~\ref{eq:gap}), pooled (mean and standard deviation over backbones, five folds and the available seeds) and per backbone (mean). $p$: Wilcoxon signed-rank test of the gaps against zero over all runs (runs share training data, so the test is descriptive).}
\label{tab:transfers}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llccccccc}
\toprule
 & & & & & & \multicolumn{3}{c}{Gap $\Delta$ per backbone} \\
\cmidrule(lr){7-9}
Change & Transfer & AUC source $\uparrow$ & AUC target $\uparrow$ & Gap $\Delta$ & $p$ & """ + " & ".join(MN.values()) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- contrast on NP
ct = rd("contrast.csv")
CN = {"delta_NP: NA minus G": "NA $-$ G ($\\delta_{\\mathrm{NP}}$, Eq.~\\ref{eq:contrast})", "pooled minus G": "G + NA $-$ G",
      "pooled minus NA": "G + NA $-$ NA", "pooled minus G (in-domain G)": "G + NA $-$ G", "pooled minus NA (in-domain NA)": "G + NA $-$ NA"}
rows = []
if ct is not None:
    for _, r in ct.iterrows():
        rows.append("%s & %s & %s & %d/%d & %s & %s \\\\" % (r.target, CN.get(r.comparison, r.comparison), sg(r.mean_diff), r.better, r.n_pairs,
                                                         "--" if pd.isna(r.rb) else ("%.2f" % r.rb).replace("-", "$-$"), pv(r.p_holm)))
w("tab_contrast", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Training sources compared on the same test images, base condition: mean paired AUC difference (pairs: backbone, fold and seed), number of pairs in which the first model is better, matched-pairs rank-biserial correlation (Eq.~\ref{eq:rb}) and Holm-adjusted Wilcoxon $p$ (Eq.~\ref{eq:holm}; one family). Target NP: all NIH children; targets G and NA: held-out patients.}
\label{tab:contrast}
\begin{tabular}{llcccc}
\toprule
Target & Comparison & $\bar d$ (AUC) & Better & $r_{\mathrm{rb}}$ & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- RQ2: thresholds and PPV
A = base[base.phase.isin(["A", "C"])]
rows = []
for src in ["G", "NA", "G,NA"]:
    for g in ["G", "NA", "NP"]:
        q = A[(A.train_groups == src) & (A.group == g)]
        if not len(q):
            continue
        rows.append("%s & %s%s & %.1f & %s & %s & %s & %s \\\\" % (SN[src], g, "$^\\ast$" if q.in_train.iloc[0] else "", 100 * q.prev.mean(),
                                                               ms(q.auc), "%.2f" % q.sens.mean(), "%.2f" % q.spec.mean(), "%.2f" % q.ppv.mean()))
    rows.append("\\addlinespace")
w("tab_threshold", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Decisions at the threshold chosen on the source validation patients (Eq.~\ref{eq:youden}), carried unchanged to every group: AUC, sensitivity, specificity (Eq.~\ref{eq:sens}) and positive predictive value at the group's prevalence (Eq.~\ref{eq:ppv}); means over backbones, folds and seeds (AUC with standard deviation), base condition. $^\ast$: training group (held-out patients). Prev.: share of images with pneumonia, in percent.}
\label{tab:threshold}
\begin{tabular}{llccccc}
\toprule
Source & Group & Prev. (\%) & AUC $\uparrow$ & Sens. $\uparrow$ & Spec. $\uparrow$ & PPV $\uparrow$ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- RQ2: remedies
rm = rd("remedies.csv")
if rm is not None and "p_holm" in rm:
    rows = []
    for rem_, q in rm.groupby("remedy", sort=False):
        for _, r in q.iterrows():
            rows.append("%s & %s$\\to$%s%s & %s & %s & %s & %d/%d & %s \\\\" % (rem_, SN.get(r.source, r.source), r.target, "$^\\ast$" if str(r.in_train) == "True" else "",
                                                                     "%.3f" % r.auc_base, "%.3f" % r.auc, sg(r.mean_diff), r.better, r.n_pairs, pv(r.p_holm)))
        rows.append("\\addlinespace")
    w("tab_remedies", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Label-free remedies against the base condition (Eq.~\ref{eq:diff}): mean AUC without and with the remedy, mean paired difference (pairs: backbone and fold; seed 0), number of pairs improved and Holm-adjusted Wilcoxon $p$ within each remedy. AdaBN only for the BatchNorm backbones (DenseNet-121, ResNet-50). $^\ast$: training group (held-out patients).}
\label{tab:remedies}
\begin{tabular}{llccccc}
\toprule
Remedy & Transfer & AUC base & AUC remedy & $\bar d$ & Better & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- RQ3: corruptions
rob = rd("robust.csv")
CK = [("noise", "Noise"), ("blur", "Blur"), ("contrast", "Contrast"), ("gamma", "Gamma"), ("jpeg", "JPEG"), ("lowres", "Low res.")]
if rob is not None:
    rows = []
    for src in ["G", "NA", "G,NA"]:
        for g in ["G", "NA", "NP"]:
            q = rob[(rob.train_groups == src) & (rob.group == g)]
            if len(q):
                rows.append("%s & %s & %s \\\\" % (SN[src], g, " & ".join("%.2f" % q["rauc_" + k].mean() if "rauc_" + k in q else "--" for k, _ in CK)))
    w("tab_robust", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Relative AUC under six corruptions averaged over five severities (Eq.~\ref{eq:rauc}; 1 = no effect, 0 = chance), mean over backbones and folds, group A.}
\label{tab:robust}
\begin{tabular}{llcccccc}
\toprule
Source & Group & """ + " & ".join(n for _, n in CK) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- figures
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
MK = {"resnet50.a1_in1k": "o", "densenet121.tv_in1k": "s", "convnext_tiny.fb_in1k": "^"}
for ax, src in zip(axs, ["G", "NA", "G,NA"]):
    for i, g in enumerate(["G", "NA", "NP"]):
        for j, (m, mk) in enumerate(MK.items()):
            q = A[(A.train_groups == src) & (A.group == g) & (A.model == m)]
            x = i + (j - 1) * 0.22 + np.linspace(-0.06, 0.06, max(1, len(q)))
            ax.scatter(x[:len(q)], q.auc, s=12, marker=mk, color=V.CAT[j], label=MN[m] if (src == "G" and i == 0) else None,
                       facecolor=V.CAT[j] if q.in_train.any() else "none", linewidths=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["G", "NA", "NP"])
    ax.set_title("Trained on %s" % SN[src]); ax.set_xlabel("Test group")
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
axs[0].set_ylabel("AUC"); axs[0].legend(loc="lower left", fontsize=8, frameon=False)
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_transfers"))
plt.close(fig)
print("RESULTS_DONE")
