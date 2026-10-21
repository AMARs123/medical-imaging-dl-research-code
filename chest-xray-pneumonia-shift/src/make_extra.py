"""Supplementary tables of chest-xray-pneumonia-shift and their numbers (paper/numbers_extra.tex): tab_extra (base condition, every source and group:
AUC, AP, balanced accuracy, Brier score and ECE), tab_ci (patient-bootstrap 95% intervals of the AUC per backbone and transfer,
group A) and the cost of the runs (train_seconds and peak_gpu_mb of results.json); with CHECK lines for the statements of the text.
All CSVs are read with keep_default_na=False ("NA" is a group). Usage: python make_extra.py <chest-xray-pneumonia-shift dir>"""
import json
import os
import sys

import pandas as pd

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
T = os.path.join(R, "paper", "tables")
rd = lambda f: pd.read_csv(os.path.join(S, f) if not os.path.isabs(f) else f, keep_default_na=False, na_values=[""])  # noqa: E731
checks, nums = [], {}


def check(ok, msg):
    if not ok:
        checks.append(msg)
        print("CHECK:", msg)


MN = {"densenet121.tv_in1k": "DenseNet-121", "resnet50.a1_in1k": "ResNet-50", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
SRC = {"G": "G", "NA": "NA", "G,NA": "G + NA"}
runs = rd(os.path.join(R, "results", "processed", "runs.csv"))
b = runs[runs.phase.isin(["A", "C"]) & (runs.aug == "base") & (runs.prep == "raw")]

# ---------------------------------------------------------------- tab_extra: probability quality per source and group
rows, ece_ok = [], True
for s in ["G", "NA", "G,NA"]:
    q = b[b.train_groups == s]
    for g in ["G", "NA", "NP"]:
        x = q[q.group == g]
        star = "$^\\ast$" if x.in_train.astype(str).iloc[0] == "True" else ""
        rows.append("%s & %s%s & %d & %.3f (%.3f) & %.3f & %.2f & %.3f & %.3f \\\\" % (
            SRC[s], g, star, len(x), x.auc.mean(), x.auc.std(), x.ap.mean(), x.bacc.mean(), x.brier.mean(), x.ece.mean()))
    rows.append("\\addlinespace")
    if s != "G,NA":
        e_in = q[q.group == s].ece.mean()
        e_out = q[(q.group != s)].groupby("group").ece.mean()
        ece_ok &= bool(e_out.min() > e_in)
open(os.path.join(T, "tab_extra.tex"), "w", encoding="utf8", newline="\n").write(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Base condition, groups A and C (three backbones, five folds, two seeds): mean AUC with its standard deviation over runs, average precision (AP), balanced accuracy at the source threshold (Eq.~\ref{eq:sens}), Brier score and expected calibration error (Eq.~\ref{eq:calib}). $^\ast$: training group (held-out patients). $R$: number of runs.}
\label{tab:extra}
\begin{tabular}{llcccccc}
\toprule
Source & Group & $R$ & AUC (SD) $\uparrow$ & AP $\uparrow$ & BAcc $\uparrow$ & Brier $\downarrow$ & ECE $\downarrow$ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
check(ece_ok, "for a single source the ECE outside the training group is no longer higher than the in-domain ECE")
e = b.groupby(["train_groups", "group"]).ece.mean()
nums["nEceGG"] = "%.3f" % e.loc[("G", "G")]; nums["nEceGNp"] = "%.3f" % e.loc[("G", "NP")]; nums["nEceGNa"] = "%.3f" % e.loc[("G", "NA")]
nums["nEceNaNa"] = "%.3f" % e.loc[("NA", "NA")]; nums["nEceNaNp"] = "%.3f" % e.loc[("NA", "NP")]
ap = b.groupby(["train_groups", "group"]).ap.mean()
nums["nApNpMax"] = "%.2f" % ap.xs("NP", level="group").max()
check(ap.xs("NP", level="group").max() < 0.2, "AP on NIH children is no longer below 0.2 for every source (text: low AP at 3.4% prevalence)")

# ---------------------------------------------------------------- tab_ci: bootstrap intervals per backbone (group A)
ci = rd("ci.csv")
ci = ci[(ci.phase == "A") & (ci.aug == "base") & (ci.prep == "raw")]
TR = [("G", "NA"), ("G", "NP"), ("NA", "G"), ("NA", "NP"), ("G,NA", "NP")]
rows = []
for m in MN:
    rows.append("\\multicolumn{4}{l}{\\textit{%s}} \\\\" % MN[m])
    for s, t in TR:
        r = ci[(ci.model == m) & (ci.source == s) & (ci.target == t)].iloc[0]
        rows.append("%s$\\to$%s & %.3f & [%.3f, %.3f] & %.3f \\\\" % (SRC[s], t, r.auc_mean, r.lo, r.hi, r.hi - r.lo))
open(os.path.join(T, "tab_ci.tex"), "w", encoding="utf8", newline="\n").write(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Mean AUC of the five fold models of group A at the groups not used for training, with patient-level bootstrap 95\% intervals (Eq.~\ref{eq:boot}; 2,000 resamples of patients, the fold models' mean AUC recomputed on each resample) and their width.}
\label{tab:ci}
\begin{tabular}{lccc}
\toprule
Transfer & AUC & 95\% interval & Width \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
ci["w"] = ci.hi - ci.lo
w = ci.set_index(["model", "source", "target"]).w
ratio = [w.loc[(m, "G", "NP")] / w.loc[(m, "G", "NA")] for m in MN]
nums["nCiWidthNp"] = "%.2f" % ci[ci.target == "NP"].w.mean(); nums["nCiWidthNa"] = "%.2f" % ci[ci.target == "NA"].w.mean()
nums["nCiRatioMin"] = "%.1f" % min(ratio); nums["nCiRatioMax"] = "%.1f" % max(ratio)
check(min(ratio) > 2, "bootstrap intervals on NIH children are no longer more than twice as wide as on NIH adults (G source)")
lo_hi = ci.set_index(["model", "source", "target"])
overlap = [lo_hi.loc[(m, "NA", "NP")].lo < lo_hi.loc[(m, "G", "NP")].hi for m in MN]
check(all(overlap), "for some backbone the G->NP and NA->NP intervals no longer overlap (text: single-backbone intervals overlap)")

# ---------------------------------------------------------------- cost of the runs
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"), keep_default_na=False)
cost = []
for eid in reg.exp_id:
    j = json.load(open(os.path.join(R, "experiments", eid, "results.json")))
    cost.append(dict(model=j["config"]["model"], src=j["config"]["train_groups"], sec=j["train_seconds"], mb=j.get("peak_gpu_mb", float("nan"))))
cost = pd.DataFrame(cost)
med = cost.groupby("model").sec.median() / 60
nums["nCostMinLo"] = "%.0f" % med.min(); nums["nCostMinHi"] = "%.0f" % med.max()
nums["nCostTotalH"] = "%.0f" % (cost.sec.sum() / 3600)
nums["nPeakGb"] = "%.1f" % (cost.mb.max() / 1024)
nums["nCostFastest"] = MN[med.idxmin()]; nums["nCostSlowest"] = MN[med.idxmax()]
check(len(cost) == len(reg), "a registered run has no results.json")
with open(os.path.join(R, "paper", "numbers_extra.tex"), "w", encoding="utf8", newline="\n") as fh:
    fh.write("% generated by src/make_extra.py (do not edit)\n")
    for k in sorted(nums):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, nums[k]))
print("EXTRA_DONE", nums, "checks failing", len(checks))
