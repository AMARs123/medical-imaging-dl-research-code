"""LaTeX tables of Paper 09, generated from stored files only. Usage: python make_tables.py <Paper_09 dir>
Every table has at most 8 columns and no resizebox; AURC values are multiplied by 100 for readability."""
import json
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
O = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
T = os.path.join(R, "paper", "tables")
os.makedirs(T, exist_ok=True)
NAME = {"ent_det": "Entropy, image", "ent_det_fg": "Entropy, foreground", "ent_det_fg_e": "Entropy, foreground + empty flag$^\\dagger$",
        "ent_mc": "MC entropy", "mi_mc": "MC mutual information", "unc_pdice_mc": "MC agreement", "ent_tta": "TTA entropy",
        "unc_pdice_tta": "TTA agreement", "ref_area": "Area (reference)", "ref_parts": "Parts (reference)", "ref_irreg": "Irregularity (reference)",
        "ens_ent": "Ensemble entropy, image", "ens_ent_fg": "Ensemble entropy, foreground", "ens_ent_fg_e": "Ensemble fg entropy + empty flag$^\\dagger$",
        "unc_pdice_ens": "Ensemble agreement"}
PASSES = {"ent_det": "1", "ent_det_fg": "1", "ent_det_fg_e": "1", "ref_area": "1", "ref_parts": "1", "ref_irreg": "1", "ent_mc": "10", "mi_mc": "10",
          "unc_pdice_mc": "10", "ent_tta": "8", "unc_pdice_tta": "8", "ens_ent": "5$^\\ast$", "ens_ent_fg": "5$^\\ast$", "ens_ent_fg_e": "5$^\\ast$",
          "unc_pdice_ens": "5$^\\ast$"}
ORDER = ["ref_area", "ref_parts", "ref_irreg", "ent_det", "ent_det_fg", "ent_det_fg_e", "ent_mc", "mi_mc", "unc_pdice_mc", "ent_tta", "unc_pdice_tta",
         "ens_ent", "ens_ent_fg", "ens_ent_fg_e", "unc_pdice_ens"]
SETS = {"kvasir": ["test", "test_blur", "test_noise", "test_contrast"], "ham": ["test", "test_blur", "test_noise", "test_contrast", "shift"]}
SETN = {"test": "Clean", "test_blur": "Blur", "test_noise": "Noise", "test_contrast": "Low contr.", "shift": "Other camera", "val": "Validation"}
ENC = {"resnet34.a1_in1k": "ResNet-34", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
DATA = {"kvasir": "Kvasir-SEG", "ham": "HAM10000"}


def write(name, s):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(s)
    print("wrote", name)


def pv(p):
    return "$<$0.001" if p < 0.001 else "%.3f" % p


fa = pd.read_csv(os.path.join(O, "failure.csv"))
fa = fa[fa.set != "val"]
fa["kind"] = np.where(fa.member == "ens", "ens", "single")

# ---------------------------------------------------------------- data
rows = []
for data, f in [("kvasir", "kvasir_meta.csv"), ("ham", "ham_meta.csv")]:
    m = pd.read_csv(os.path.join(R, "data", "processed", f))
    if "has_mask" in m:
        m = m[m.has_mask]
    for sp in ["train", "val", "test", "shift"]:
        q = m[m.split == sp]
        if len(q):
            les = q.lesion_id.nunique() if "lesion_id" in q else None
            src = ", ".join(sorted(q.dataset.unique())) if "dataset" in q else "Kvasir-SEG"
            rows.append("%s & %s & %d & %s & %.1f & %s \\\\" % (DATA[data], {"train": "Training", "val": "Validation", "test": "Test",
                                                                             "shift": "Other camera (test only)"}[sp], len(q),
                                                              "{:,}".format(les) if les else "n/a", 100 * q.mask_frac.mean(), src.replace("_", "\\_")))
write("tab_data", r"""\begin{table}[!htbp]
\centering\small
\caption{Data. Images were resized to 256$\times$256 pixels. Lesion identifiers are available for HAM10000 only; its development splits are grouped by lesion, and all images from the \texttt{vidir\_molemax} collection form a separate test set from another acquisition system. Kvasir-SEG has no patient identifiers, so its split is random by image.}
\label{tab:data}
\begin{tabular}{llrrr>{\raggedright\arraybackslash}p{3.6cm}}
\toprule
Data set & Split & Images & Lesions & Mean mask area (\%) & Source collections \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- segmentation quality
q = pd.read_csv(os.path.join(O, "quality.csv"))
q = q[q.set != "val"]
rows = []
for data in ["kvasir", "ham"]:
    for enc in ENC:
        for st in SETS[data]:
            s = q[(q.data == data) & (q.encoder == enc) & (q.set == st) & (q.member != "ens")]
            e = q[(q.data == data) & (q.encoder == enc) & (q.set == st) & (q.member == "ens")]
            rows.append("%s & %s & %s & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %.3f \\\\" % (
                DATA[data], ENC[enc], SETN[st], s.dice_det.mean(), s.dice_det.std(ddof=1), s.dice_mc.mean(), s.dice_mc.std(ddof=1),
                s.dice_tta.mean(), s.dice_tta.std(ddof=1), e.dice_ens.iloc[0]))
        rows.append("\\addlinespace")
write("tab_quality", r"""\begin{table}[!htbp]
\centering\small
\caption{Segmentation quality: mean Dice over images, mean $\pm$ SD over the five seeds for the single models, and the Dice of the five-seed ensemble. MC: mean of 10 dropout samples; TTA: mean of 8 flipped and rotated views.}
\label{tab:quality}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllcccc}
\toprule
Data set & Encoder & Images & Single pass $\uparrow$ & MC dropout $\uparrow$ & TTA $\uparrow$ & Ensemble $\uparrow$ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- failure detection (AURC x 100), one table per data set
for data in ["kvasir", "ham"]:
    sets = SETS[data]
    rows = []
    for sc in ORDER:
        d = fa[(fa.data == data) & (fa.score == sc) & (fa.kind == ("ens" if sc.startswith(("ens", "unc_pdice_ens")) else "single"))]
        vals = []
        for st in sets:
            v = d[d.set == st].aurc
            vals.append("%.2f" % (100 * v.mean()) if len(v) else "--")
        allm = 100 * d.groupby("set").aurc.mean().mean()
        rows.append("%s & %s & %s & %.2f \\\\" % (NAME[sc], PASSES[sc], " & ".join(vals), allm))
        if sc in ("ref_irreg", "ent_det_fg_e", "unc_pdice_tta"):
            rows.append("\\addlinespace")
    opt = []
    for st in sets:
        o = fa[(fa.data == data) & (fa.set == st) & (fa.score == "ent_det") & (fa.kind == "single")]
        opt.append("%.2f" % (100 * (o.aurc - o.eaurc).mean()))
    write("tab_fail_%s" % data, r"""\begin{table}[!htbp]
\centering\small
\caption{Failure detection on %s: area under the risk--coverage curve (AURC $\times$100, lower is better; risk = 1 $-$ Dice of the accepted images). Single-model scores: mean over 2 encoders $\times$ 5 seeds; ensemble scores: mean over the 2 encoders. Passes: forward passes per image ($^\ast$five trained networks). $^\dagger$Post-hoc variant, defined after inspecting the worst cases (Section~\ref{sec:posthoc}). The last row gives the best possible AURC of the single-pass prediction (images sorted by their true Dice).}
\label{tab:fail_%s}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lc%s}
\toprule
Score & Passes & %s & Mean $\downarrow$ \\
\midrule
""" % (DATA[data], data, "c" * (len(sets) + 1), " & ".join(SETN[s] + r" $\downarrow$" for s in sets)) + "\n".join(rows[:-1] if rows[-1] == "\\addlinespace" else rows)
          + r"""
\midrule
Oracle (single pass) & -- & %s & %.2f \\
\bottomrule
\end{tabular}
\end{table}
""" % (" & ".join(opt), np.mean([float(x) for x in opt])))

# ---------------------------------------------------------------- planned and post-hoc paired tests
pt = pd.read_csv(os.path.join(S, "paired_tests.csv"))
ph = pd.read_csv(os.path.join(S, "posthoc_tests.csv"))
rows = []
for r in pt.itertuples():
    rows.append("%s vs.\\ %s & %+.2f $\\pm$ %.2f & %d/%d & %.2f & %s \\\\" % (NAME[r.a], NAME[r.b], 100 * r.mean_diff, 100 * r.sd_diff, r.a_better, r.n,
                                                                             r.r_rb, pv(r.p_holm)))
rows.append("\\midrule\n\\multicolumn{5}{l}{\\emph{Post hoc (exploratory, unadjusted $p$)}} \\\\")
for r in ph.itertuples():
    rows.append("%s vs.\\ %s & %+.2f $\\pm$ %.2f & %d/%d & %.2f & %s \\\\" % (NAME[r.a], NAME[r.b], 100 * r.mean_diff, 100 * r.sd_diff, r.a_better, r.n,
                                                                             r.r_rb, pv(r.p)))
write("tab_tests", r"""\begin{table}[!htbp]
\centering\small
\caption{Paired comparisons of failure-detection scores. Unit: one run (2 data sets $\times$ 2 encoders $\times$ 5 seeds = 20); value: the run's AURC averaged over its evaluation sets. Difference = first minus second score (AURC $\times$100; negative: the first score is better), mean $\pm$ SD over runs; runs in which the first score is better; matched-pairs rank-biserial correlation $r_{\mathrm{rb}}$; Wilcoxon signed-rank test, Holm-adjusted over the nine planned comparisons.}
\label{tab:tests}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{7.4cm}rrrr}
\toprule
Comparison & Difference & First better & $r_{\mathrm{rb}}$ & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- cost
ef = json.load(open(os.path.join(S, "efficiency.json")))
rows = []
for enc, e in ef["encoders"].items():
    tr = np.mean([v for k, v in ef["train_minutes"].items() if k.endswith(enc)])
    rows.append("%s & %.1f & %.0f & %.0f (%.1f$\\times$) & %.0f (%.1f$\\times$) & %.0f (%.1f$\\times$) & %.0f \\\\" % (
        ENC[enc], e["params_m"], e["single_ms"], e["mc10_ms"], e["mc10_ms"] / e["single_ms"], e["tta8_ms"], e["tta8_ms"] / e["single_ms"],
        e["ens5_ms"], e["ens5_ms"] / e["single_ms"], tr))
write("tab_cost", r"""\begin{table}[!htbp]
\centering\small
\caption{Measured cost per image on the CPU (median over %d Kvasir-SEG test images, batch 1, %d threads, shared server; times include computing the uncertainty score), with the factor relative to one pass. The ensemble also needs five trained networks; the last column gives the median training time of one network (one GPU share), averaged over the two data sets.}
\label{tab:cost}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrrrrr}
\toprule
Encoder & Params (M) & 1 pass (ms) $\downarrow$ & MC, 10 (ms) $\downarrow$ & TTA, 8 (ms) $\downarrow$ & Ensemble, 5 (ms) $\downarrow$ & Training (min) \\
\midrule
""" % (ef["images_timed"], ef["threads"]) + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- pixel-level error detection
px = pd.read_csv(os.path.join(O, "pixel.csv"))
px = px[px.set != "val"]
rows = []
for data in ["kvasir", "ham"]:
    for st in SETS[data]:
        d = px[(px.data == data) & (px.set == st)].groupby("score").auroc_pixel
        m, s = d.mean(), d.std()
        rows.append("%s & %s & %s & %.1f \\\\" % (DATA[data], SETN[st], " & ".join("%.3f $\\pm$ %.3f" % (m[k], s[k]) for k in ["ent_det", "ent_mc", "mi_mc", "ent_tta"]),
                                                100 * px[(px.data == data) & (px.set == st)].error_rate.mean()))
write("tab_pixel", r"""\begin{table}[!htbp]
\centering\small
\caption{Pixel-level error detection: AUROC of each pixel uncertainty map for the errors of the single-pass prediction (512 random pixels per image), mean $\pm$ SD over 2 encoders $\times$ 5 seeds. The last column gives the share of wrongly labelled pixels.}
\label{tab:pixel}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llccccr}
\toprule
Data set & Images & Entropy, 1 pass $\uparrow$ & MC entropy $\uparrow$ & MC mutual inf. $\uparrow$ & TTA entropy $\uparrow$ & Errors (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- appendix: AUROC for failure and threshold sensitivity, calibration
rows = []
for data in ["kvasir", "ham"]:
    for sc in ORDER:
        d = fa[(fa.data == data) & (fa.score == sc) & (fa.kind == ("ens" if sc.startswith(("ens", "unc_pdice_ens")) else "single"))]
        vals = ["%.2f" % d[d.set == st].auroc_fail.mean() for st in SETS[data]]
        vals += [""] * (5 - len(vals))
        rows.append("%s & %s & %s \\\\" % (DATA[data] if sc == ORDER[0] else "", NAME[sc], " & ".join(vals)))
    rows.append("\\midrule")
fails = fa[(fa.score == "ent_det") & (fa.kind == "single")].groupby(["data", "set"]).failures.mean()
write("tab_auroc", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{AUROC for detecting failed segmentations (Dice $<$ 0.5), mean over runs. Failures per run (single pass): %s. With so few failures per set, AUROC values are noisy; the main text uses AURC, which uses every image.}
\label{tab:auroc}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}
\toprule
Data set & Score & Clean & Blur & Noise & Low contr. & Other camera \\
\midrule
""" % "; ".join("%s %s %.1f" % (DATA[d], SETN[s].lower(), v) for (d, s), v in fails.items()) + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
th = pd.read_csv(os.path.join(S, "threshold.csv"))
g = th.groupby(["data", "threshold", "score"]).auroc.mean().unstack("threshold")
rows = []
for data in ["kvasir", "ham"]:
    for sc in [s for s in ORDER if not s.startswith(("ens", "unc_pdice_ens"))]:
        if (data, sc) in g.index:
            v = g.loc[(data, sc)]
            rows.append("%s & %s & %.2f & %.2f & %.2f \\\\" % (DATA[data] if sc == "ref_area" else "", NAME[sc], v[0.5], v[0.7], v[0.8]))
    rows.append("\\midrule")
write("tab_threshold", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Sensitivity to the failure definition: AUROC for detecting images with Dice below 0.5, 0.7 and 0.8 (single models, mean over runs and evaluation sets).}
\label{tab:threshold}
\begin{tabular}{llccc}
\toprule
Data set & Score & Dice $<$ 0.5 & Dice $<$ 0.7 & Dice $<$ 0.8 \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
ca = pd.read_csv(os.path.join(O, "calibration.csv"))
ca = ca[ca.set != "val"]
rows = []
for data in ["kvasir", "ham"]:
    for st in SETS[data]:
        s = ca[(ca.data == data) & (ca.set == st) & (ca.member != "ens")].ece
        e = ca[(ca.data == data) & (ca.set == st) & (ca.member == "ens")].ece
        rows.append("%s & %s & %.2f $\\pm$ %.2f & %.2f \\\\" % (DATA[data], SETN[st], 100 * s.mean(), 100 * s.std(), 100 * e.mean()))
write("tab_calib", r"""\begin{table}[!htbp]
\centering\small
\caption{Pixel calibration: expected calibration error (ECE $\times$100, 15 bins, 4,096 random pixels per image) of the single-pass foreground probability (mean $\pm$ SD over 10 runs) and of the five-seed ensemble mean (mean over 2 encoders).}
\label{tab:calib}
\begin{tabular}{llcc}
\toprule
Data set & Images & Single pass $\downarrow$ & Ensemble $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
