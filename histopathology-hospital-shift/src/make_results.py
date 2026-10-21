"""Result tables and figures of histopathology-hospital-shift from results/processed/runs.csv and results/statistics (analyze.py).
Tables: tab_conditions (accuracy on the three evaluation sets per condition), tab_gains (RQ1/RQ2: gain over the reference under the
three selection rules, Eq. gain), tab_spread (RQ3: seed spread, Brown-Forsythe test, probit slope and Spearman correlation).
Figures: fig_conditions (test accuracy of every run by condition), fig_idtest (ID against test accuracy of every run).
Usage: python make_results.py <histopathology-hospital-shift dir>"""
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
os.makedirs(T, exist_ok=True)
runs = pd.read_csv(os.path.join(P, "runs.csv"))
cond = pd.read_csv(os.path.join(S, "conditions.csv"))
gains = pd.read_csv(os.path.join(S, "gains.csv"))
spread = pd.read_csv(os.path.join(S, "spread.csv"))
_pf = os.path.join(S, "probit.csv")
probit = pd.read_csv(_pf) if os.path.getsize(_pf) > 5 else pd.DataFrame(columns=["arch", "weights", "aug", "mode", "slope", "spearman"])
AN = {"r50": "ResNet-50", "vits": "ViT-S/16"}
# conditions in display order: (weights, aug, mode, label)
CO = [("imagenet", "none", "finetune", "ImageNet, fine-tuned (reference)"), ("imagenet", "hed", "finetune", "ImageNet + HED, fine-tuned"),
      ("pathology", "none", "finetune", "Pathology, fine-tuned"), ("pathology", "hed", "finetune", "Pathology + HED, fine-tuned"),
      ("imagenet", "none", "linear", "ImageNet, linear probe"), ("pathology", "none", "linear", "Pathology, linear probe")]


def w(name, body):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(body)


def pc(x, f="%.1f"):
    return "--" if pd.isna(x) else f % (100 * x)


def spc(x):
    return "--" if pd.isna(x) else ("$%+.1f$" % (100 * x))


def pv(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else "%.3f" % p)


def sel(df, a, wt, au, mo):
    return df[(df.arch == a) & (df.weights == wt) & (df.aug == au) & (df["mode"] == mo)]


# ---------------------------------------------------------------- conditions
rows = []
for a in ["r50", "vits"]:
    rows.append("\\multicolumn{6}{l}{\\textit{%s}} \\\\" % AN[a])
    for wt, au, mo, lab in CO:
        q = sel(cond, a, wt, au, mo)
        if not len(q):
            continue
        q = q.iloc[0]
        rows.append("%s & %d & %s & %s & %s (%s) & %s--%s \\\\" % (lab, q.n, pc(q.acc_id), pc(q.acc_ood), pc(q.acc_test), pc(q.sd_test), pc(q.min_test), pc(q.max_test)))
w("tab_conditions", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Accuracy (\%) of the final model of every condition on the ID validation set, the OOD validation hospital and the test hospital (all patches of each set): mean over seeds, with standard deviation and range for the test hospital. $S$: number of seeds. HED: stain augmentation (Section~\ref{sec:hed}).}
\label{tab:conditions}
\begin{tabular}{lccccc}
\toprule
Condition & $S$ & ID val. $\uparrow$ & OOD val. $\uparrow$ & Test (SD) $\uparrow$ & Test range \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- gains under the three rules
rows = []
for a in ["r50", "vits"]:
    rows.append("\\multicolumn{8}{l}{\\textit{%s}} \\\\" % AN[a])
    for wt, au, mo, lab in CO[1:]:
        q = sel(gains, a, wt, au, mo).set_index("rule")
        if not len(q):
            continue
        cells = []
        for r_ in ["ood", "id", "final"]:
            g = q.loc[r_]
            cells.append("%s [%s, %s]" % (spc(g.gain), spc(g.lo), spc(g.hi)))
        share = q.loc["id", "id_share_of_ood"] if q.loc["ood", "lo"] > 0 else np.nan
        rows.append("%s & %d & %s & %s & %s \\\\" % (lab, q.loc["id", "n"], " & ".join(cells), "--" if pd.isna(share) else "%.2f" % share,
                                                    pv(q.loc["ood", "p_holm"]) if mo == "finetune" else "--"))
w("tab_gains", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Gain in test-hospital accuracy over the reference of the same architecture (ImageNet weights, no stain augmentation, fine-tuned; Eq.~\ref{eq:gain}), in percentage points, with 95\% bootstrap intervals over seeds, when the evaluation point is chosen on the OOD validation hospital, on the ID validation set, or not chosen (final point; Eq.~\ref{eq:select}). Share: ID-selected gain divided by OOD-selected gain, when the OOD-selected gain is clearly positive. $p_{\mathrm{Holm}}$: Mann--Whitney test of the OOD-selected accuracies, Holm-adjusted over the planned fine-tuned comparisons. Selection is made on fixed subsets of 10,000 patches per set.}
\label{tab:gains}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccrr}
\toprule
Condition & $S$ & $G_{\mathrm{ood}}$ & $G_{\mathrm{id}}$ & $G_{\mathrm{final}}$ & Share & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows).replace("\\multicolumn{8}", "\\multicolumn{7}") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- spread and ID-test relation
rows = []
for a in ["r50", "vits"]:
    rows.append("\\multicolumn{6}{l}{\\textit{%s}} \\\\" % AN[a])
    for wt, au, mo, lab in CO:
        s = sel(spread, a, wt, au, mo)
        p = sel(probit, a, wt, au, mo)
        if not len(s):
            continue
        s = s.iloc[0]
        slope = "--" if not len(p) or pd.isna(p.iloc[0].slope) else ("%.2f" % p.iloc[0].slope).replace("-", "$-$")
        rho = "--" if not len(p) or pd.isna(p.iloc[0].spearman) else ("%.2f" % p.iloc[0].spearman).replace("-", "$-$")
        rows.append("%s & %s & %s & %s & %s & %s \\\\" % (lab, pc(s.sd, "%.2f"), pc(s.range, "%.1f"), pv(s.p_bf), slope, rho))
w("tab_spread", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Spread of test-hospital accuracy over seeds (standard deviation and range, percentage points), Brown--Forsythe test of equal spread against the reference ($p_{\mathrm{BF}}$, unadjusted), and the relation between ID and test-hospital accuracy over the seeds of each condition: probit slope $\gamma_1$ (Eq.~\ref{eq:probit}) and Spearman correlation $\rho$ (shown for at least four seeds).}
\label{tab:spread}
\begin{tabular}{lccccc}
\toprule
Condition & SD & Range & $p_{\mathrm{BF}}$ & $\gamma_1$ & $\rho$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- figures
SH = {("imagenet", "none", "finetune"): "Ref.", ("imagenet", "hed", "finetune"): "+HED", ("pathology", "none", "finetune"): "Path.",
      ("pathology", "hed", "finetune"): "Path.+HED", ("imagenet", "none", "linear"): "Probe IN", ("pathology", "none", "linear"): "Probe path."}
COL = {"imagenet": V.CAT[0], "pathology": V.CAT[1]}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, a in zip(axs, ["r50", "vits"]):
    for i, (wt, au, mo, lab) in enumerate(CO):
        q = sel(runs, a, wt, au, mo)
        x = i + np.linspace(-0.12, 0.12, max(1, len(q)))
        ax.scatter(x[:len(q)], 100 * q.acc_test, s=14, color=COL[wt], marker="o" if au == "none" else "s",
                   facecolor=COL[wt] if mo == "finetune" else "none", linewidths=0.9)
        if len(q):
            ax.plot([i - 0.25, i + 0.25], [100 * q.acc_test.mean()] * 2, color="0.2", lw=1.2)
    ax.set_xticks(range(len(CO))); ax.set_xticklabels([SH[c[:3]] for c in CO], rotation=35, ha="right")
    ax.set_title(AN[a])
axs[0].set_ylabel("Test-hospital accuracy (%)")
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_conditions"))
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, a in zip(axs, ["r50", "vits"]):
    for wt, au, mo, lab in CO:
        q = sel(runs, a, wt, au, mo)
        ax.scatter(100 * q.acc_id, 100 * q.acc_test, s=14, color=COL[wt], marker="o" if au == "none" else "s",
                   facecolor=COL[wt] if mo == "finetune" else "none", linewidths=0.9, label=SH[(wt, au, mo)] if a == "r50" else None)
    ax.set_title(AN[a]); ax.set_xlabel("ID validation accuracy (%)")
axs[0].set_ylabel("Test-hospital accuracy (%)")
axs[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_idtest"))
plt.close(fig)
print("RESULTS_DONE")
