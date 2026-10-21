"""LaTeX tables of ecg-attribution-testing from results/statistics, results/external and the data files (nothing typed by hand).
Research_Expert round (2026-10-16): at most 8 columns per table in the body, no \\resizebox, wide matrices in the
appendix, eight attribution methods, external and cost tables. Tables whose inputs do not exist yet are written as
clearly marked PENDING placeholders.
Usage: python make_tables.py <ecg-attribution-testing dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
X = os.path.join(R, "results", "external")
T = os.path.join(R, "paper", "tables")
os.makedirs(T, exist_ok=True)
AN = {"xresnet1d50": "XResNet1d-50", "inception1d": "InceptionTime-1d", "convtran1d": "Conv-Transformer-1d"}
MN = {"saliency": "Saliency", "gxi": "Gradient$\\times$input", "ig": "Integrated gradients", "smoothgrad": "SmoothGrad",
      "gradshap": "GradientSHAP", "occlusion": "Occlusion", "cam": "CAM", "gradcam_mid": "Grad-CAM (mid layer)",
      "amplitude": "Amplitude (no model)"}
FAMILY = {"saliency": "gradient", "gxi": "gradient", "ig": "gradient", "smoothgrad": "gradient", "gradshap": "gradient",
          "occlusion": "perturbation", "cam": "activation", "gradcam_mid": "activation"}
ARCH = list(AN)
ALL = ["saliency", "gxi", "ig", "smoothgrad", "gradshap", "occlusion", "cam", "gradcam_mid"]
METH = [m for m in ALL if os.path.exists(os.path.join(R, "results", "attributions", "profiles_%s.npy" % m))]
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]


def w(name, s):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(s)
    print("table", name)


def pending(name, label, what):
    w(name, "\\begin{table}\n\\centering\n\\caption{%s. PENDING: the analysis that produces this table has not finished.}\\label{%s}\n"
            "\\begin{tabular}{l}\\toprule PENDING\\\\ \\bottomrule\\end{tabular}\n\\end{table}\n" % (what, label))


def fam_rules(rows_by_method):
    out, prev = [], None
    for m, rows in rows_by_method:
        if prev is not None and FAMILY.get(m, "ref") != FAMILY.get(prev, "ref"):
            out.append("\\midrule")
        out += rows
        prev = m
    return out


# ---------------------------------------------------------------- data
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
te = meta[meta.strat_fold == 10]
ext_meta = os.path.join(R, "data", "external", "processed", "meta.csv")
em = pd.read_csv(ext_meta) if os.path.exists(ext_meta) else None
rows = ["Recordings & %d & %d & %s \\\\" % (len(meta), len(te), "%d" % len(em) if em is not None else "pending"),
        "Patients & %d & %d & n/a \\\\" % (meta.patient_id.nunique(), te.patient_id.nunique())]
for c in SUPER:
    rows.append("%s labels & %d & %d & %s \\\\" % (c, int(meta[c].sum()), int(te[c].sum()), "%d" % int(em[c].sum()) if em is not None else "pending"))
rows.append("Items (recording $\\times$ label) & n/a & %d & %s \\\\" % (int(te[SUPER].values.sum()), "%d" % int(em[SUPER].values.sum()) if em is not None else "pending"))
w("tab_data", r"""\begin{table}
\centering
\caption{Data used in the study. PTB-XL: recordings with at least one diagnostic superclass; test: recommended fold 10. External: label-balanced subset of the Chapman--Shaoxing--Ningbo database (Section~\ref{sec:ext}), used only for testing. An item is one recording with one of its labels.}\label{tab:data}
\small
\begin{tabular}{lrrr}
\toprule
 & PTB-XL (all) & PTB-XL test & External\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))

# ---------------------------------------------------------------- performance (PTB-XL test fold)
pr = pd.read_csv(os.path.join(S, "performance_runs.csv"))
prt = pr[pr.random_init == 0]
rows = []
for a in ARCH:
    q = prt[prt.model == a]
    rows.append("%s & %s \\\\" % (AN[a], " & ".join("%.3f (%.3f)" % (q[c].mean(), q[c].std()) for c in ["macro_auc"] + ["auc_" + s for s in SUPER] + ["macro_ap"])))
w("tab_perf", r"""\begin{table}
\centering
\caption{Classification on the PTB-XL test fold: mean (standard deviation) over five seeds of the area under the ROC curve (AUC, Eq.~\ref{eq:auc}) and of the macro-averaged precision (AP). $\uparrow$: higher is better.}\label{tab:perf}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lccccccc}
\toprule
Model & Macro AUC $\uparrow$ & NORM & MI & STTC & CD & HYP & Macro AP $\uparrow$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))

# ---------------------------------------------------------------- stability
st = pd.read_csv(os.path.join(S, "rq1_stability.csv"))
st = st[st.label == "all"]
blocks = []
for m in METH:
    rr = []
    for a in ARCH:
        g = lambda t: st[(st.method == m) & (st.type == t) & (st.arch == a)]  # noqa: E731
        sd, rn = g("seed"), g("random")
        ar = st[(st.method == m) & (st.type == "arch") & st.arch.str.contains(a)]
        rr.append("%s & %s & %.2f [%.2f, %.2f] & %.2f [%.2f, %.2f] & %.2f & %.2f & %d \\\\" % (
            MN[m] if a == ARCH[0] else "", AN[a], sd.spearman.iloc[0], sd.ci_low.iloc[0], sd.ci_high.iloc[0],
            rn.spearman.iloc[0], rn.ci_low.iloc[0], rn.ci_high.iloc[0], ar.spearman.mean(), sd.jaccard_top10.iloc[0], sd.n.iloc[0]))
    blocks.append((m, rr + ["\\addlinespace[2pt]"]))
lines = fam_rules(blocks)
w("tab_stability", r"""\begin{table}
\centering
\caption{Agreement of relevance profiles (Spearman $\rho$, Eq.~\ref{eq:spearman}; mean over test items, 95\%% bootstrap CI over patients). Seed: two trained models of the same architecture; untrained: trained versus untrained model of the same architecture (reference level); cross-arch.: mean over pairs with the other two architectures; top-10\%%: Jaccard overlap (Eq.~\ref{eq:jaccard}) of the 100 highest-ranked steps for seed pairs; $n$: items with non-constant profiles. Rules separate the gradient, perturbation and activation families. $\uparrow$: higher is more stable.}\label{tab:stability}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccr}
\toprule
Method & Model & Seed $\rho\uparrow$ & Untrained $\rho$ & Cross-arch.\ $\rho$ & Top-10\%% $\uparrow$ & $n$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines[:-1] if lines[-1].startswith("\\addlinespace") else lines))

# ---------------------------------------------------------------- localisation: lift and learned share in the expected segments
lo = pd.read_csv(os.path.join(S, "rq2_localisation.csv"))
le = pd.read_csv(os.path.join(S, "rq2_learned_excess.csv"))
EXP = [("MI", "QRS"), ("MI", "STT"), ("STTC", "STT"), ("CD", "QRS"), ("HYP", "QRS")]
blocks = []
for m in METH + ["amplitude"]:
    cells = []
    for lab, sg in EXP:
        t_ = lo[(lo.method == m) & (lo.random_init == 0) & (lo.label == lab)]["lift_" + sg].mean()
        if m == "amplitude":
            cells.append("%.2f" % t_)
            continue
        e_ = le[(le.method == m) & (le.label == lab) & (le.segment == sg)]
        sig = int((e_.p_holm < 0.05).sum())
        cells.append(r"\makecell{%.2f\\ {\scriptsize %+.3f$^{%d}$}}" % (t_, e_.learned_excess.mean(), sig))
    blocks.append((m, ["%s & %s \\\\" % (MN[m], " & ".join(cells))]))
w("tab_localisation", r"""\begin{table}
\centering
\caption{Relevance in the segments each class is expected to affect. Upper number: lift (relevance share divided by time share, Eq.~\ref{eq:lift}; 1 = no concentration), mean over the trained models of the three architectures. Lower number: learned share $\Delta^{\mathrm{learn}}$ (Eq.~\ref{eq:learned}), trained minus untrained model of the same architecture on the same item, mean over architectures; superscript: number of the three architectures for which it differs from zero after Holm correction ($p<0.05$). Amplitude: model-free profile, for which no learned share exists.}\label{tab:localisation}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Method & MI: QRS & MI: ST-T & STTC: ST-T & CD: QRS & HYP: QRS\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(fam_rules(blocks)))

# ---------------------------------------------------------------- faithfulness: deletion gain per architecture (trained), details in the appendix
fa = pd.read_csv(os.path.join(S, "rq3_faithfulness.csv"))
blocks, app = [], []
for m in METH + ["amplitude"]:
    cells, cells2 = [], []
    for a in ARCH:
        q = fa[(fa.method == m) & (fa.arch == a) & (fa.random_init == 0)].iloc[0]
        r0 = fa[(fa.method == m) & (fa.arch == a) & (fa.random_init == 1)].iloc[0]
        cells.append("%.2f [%.2f, %.2f]%s" % (q.gain, q.ci_low, q.ci_high, "" if q.p_holm < 0.05 else "$^{\\dagger}$"))
        cells2.append("%.0f\\%% & %.2f" % (100 * q.zero_map_rate, r0.gain))
    blocks.append((m, ["%s & %s \\\\" % (MN[m], " & ".join(cells))]))
    app.append((m, ["%s & %s \\\\" % (MN[m], " & ".join(cells2))]))
w("tab_faith", r"""\begin{table}
\centering
\caption{Deletion gain (Eq.~\ref{eq:gain}): drop of the class logit when the 10\%% highest-ranked time steps are set to zero, minus the drop for 10\%% random steps, for the trained models (mean over items with a non-zero profile, 95\%% bootstrap CI over patients). $\dagger$: not different from zero after Holm correction. The amplitude row ranks time steps by signal magnitude without any model; a method that does not exceed it cannot be credited with faithfulness. Zero-profile shares and untrained gains are in Table~\ref{tab:faithmore}. $\uparrow$: higher is better.}\label{tab:faith}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
Method & XResNet1d-50 $\uparrow$ & InceptionTime-1d $\uparrow$ & Conv-Transformer-1d $\uparrow$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(fam_rules(blocks)))
w("tab_faithmore", r"""\begin{table}
\centering
\caption{Supplement to Table~\ref{tab:faith}: share of all-zero profiles of the trained models (no positive evidence; excluded from the deletion test) and mean deletion gain of the untrained model of each architecture.}\label{tab:faithmore}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{2}{c}{XResNet1d-50} & \multicolumn{2}{c}{InceptionTime-1d} & \multicolumn{2}{c}{Conv-Transformer-1d}\\
Method & Zero & Untrained & Zero & Untrained & Zero & Untrained\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(fam_rules(app)))

# ---------------------------------------------------------------- stability vs detection
r4 = pd.read_csv(os.path.join(S, "rq4_stability_vs_detection.csv"))
blocks = []
for m in METH:
    cells = []
    for a in ARCH:
        q = r4[(r4.method == m) & (r4.arch == a)].iloc[0]
        cells.append("%.2f / %.2f & %+.2f" % (q.stab_detected, q.stab_missed, q.spearman_prob))
    blocks.append((m, ["%s & %s \\\\" % (MN[m], " & ".join(cells))]))
w("tab_detection", r"""\begin{table}
\centering
\caption{Seed stability (mean Spearman $\rho$ of seed pairs) for labels detected by the five-seed mean (Eq.~\ref{eq:detect}) versus missed labels, and the Spearman correlation $r_s$ between seed stability and the five-seed mean probability, over all test items. %d of %d detected-versus-missed differences are significant after Holm correction (Mann--Whitney test, $p<0.05$).}\label{tab:detection}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcrcrcr}
\toprule
 & \multicolumn{2}{c}{XResNet1d-50} & \multicolumn{2}{c}{InceptionTime-1d} & \multicolumn{2}{c}{Conv-Transformer-1d}\\
Method & Det./missed & $r_s$ & Det./missed & $r_s$ & Det./missed & $r_s$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % ((r4.p_holm < 0.05).sum(), len(r4), "\n".join(fam_rules(blocks))))

# ---------------------------------------------------------------- budget and plan (configuration and run counts)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
rows = []
for a in ARCH:
    q = prt[prt.model == a]
    rows.append("%s & %.2f & 40 & 64 & $10^{-3}$ & %d & %d \\\\" % (AN[a], q.n_params.iloc[0] / 1e6, (reg[(reg.model == a)].random_init == 0).sum(),
                                                                   (reg[(reg.model == a)].random_init == 1).sum()))
w("tab_budget", r"""\begin{table}
\centering
\caption{Training budget per architecture. All architectures share the recipe of Section~\ref{sec:recipe}: AdamW with weight decay $10^{-2}$, one-cycle schedule, binary cross-entropy, the same augmentation, checkpoint selection on fold 9 and the same hardware.}\label{tab:budget}
\small
\begin{tabular}{lrrrrrr}
\toprule
Model & Params (M) & Epochs & Batch & Learning rate & Trained seeds & Untrained\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))
nitems = int(te[SUPER].values.sum())
plan = [("Classification", "AUC per class, macro AUC, AP", "15 trained", "test fold; external set", "planned"),
        ("RQ1 stability", r"Spearman $\rho$, top-10\% Jaccard", "18", "%d items, 8 methods" % nitems, "planned (5 methods); 3 methods added"),
        ("RQ2 localisation", "segment lift, learned share", "18", "items with landmarks", "planned; learned share added"),
        ("RQ3 faithfulness", "deletion gain", "18", "items with non-zero profile", "planned; amplitude reference added"),
        ("RQ4 detection", "stability of detected vs missed labels", "15 trained", "%d items" % nitems, "planned"),
        ("External replication", "AUC, stability, lift, deletion gain", "18", "external items, 4 methods + amplitude", "added"),
        ("Cost", "parameters, FLOPs, latency, attribution time", "3 architectures", "CPU", "added")]
w("tab_plan", r"""\begin{table}
\centering
\caption{Analysis plan. Every analysis uses the same items and the same 18 models (15 trained, 3 untrained), so methods are compared with paired tests on items. Status: planned before the attributions were computed, or added later (labelled as additional analyses in the text).}\label{tab:plan}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lp{3.6cm}lp{3.4cm}p{3.2cm}}
\toprule
Analysis & Measure & Models & Data & Status\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join("%s & %s & %s & %s & %s \\\\" % r for r in plan))

# ---------------------------------------------------------------- external replication
if os.path.exists(os.path.join(X, "performance.csv")) and os.path.exists(os.path.join(X, "seed_spearman.csv")):
    ep = pd.read_csv(os.path.join(X, "performance.csv"))
    ss = pd.read_csv(os.path.join(X, "seed_spearman.csv"))
    rows = []
    for a in ARCH:
        q = ep[(ep.model == a) & (ep.random_init == 0)]
        sp = ss[(ss.method == "ig") & (ss.arch == a)].iloc[0]
        cells = ["%.3f $\\pm$ %.3f" % (prt[prt.model == a].macro_auc.mean(), prt[prt.model == a].macro_auc.std()),
                 "%.3f $\\pm$ %.3f" % (q.macro_auc.mean(), q.macro_auc.std()), "%.2f / %.2f" % (sp.seed_spearman, sp.untrained_spearman)]
        for mt in ["gxi", "ig", "amplitude"]:
            L = pd.read_csv(os.path.join(X, "local_%s.csv" % mt))
            L["arch"] = reg.model.values[L.model_idx]
            L["ri"] = reg.random_init.values[L.model_idx]
            L = L[L.fiducials_ok & ~L.zero_map & (L.arch == a) & (L.ri == 0)]
            cells.append("%.2f" % float(np.exp(np.log((L.share_QRS / L.time_QRS.clip(lower=1e-6)).clip(lower=1e-6)).mean())))
        D = {}
        for mt in ["ig", "amplitude"]:
            d = pd.read_csv(os.path.join(X, "deletion_%s.csv" % mt))
            d["arch"] = reg.model.values[d.model_idx]
            d["ri"] = reg.random_init.values[d.model_idx]
            d = d[(d.arch == a) & (d.ri == 0)]
            D[mt] = (d.drop_top10 - d.drop_rand10).mean()
        cells.append("%.2f / %.2f" % (D["ig"], D["amplitude"]))
        rows.append("%s & %s \\\\" % (AN[a], " & ".join(cells)))
    w("tab_external", r"""\begin{table}
\centering
\caption{External replication on the Chapman--Shaoxing--Ningbo subset (no setting tuned on these data). AUC: macro AUC, mean $\pm$ SD over five seeds, on PTB-XL and on the external set. Seed / untrained $\rho$: mean Spearman agreement of integrated-gradient profiles between seeds and with the untrained model. QRS lift (Eq.~\ref{eq:lift}) over all items for gradient$\times$input, integrated gradients and the model-free amplitude profile, using NeuroKit2 landmarks. Deletion gain: integrated gradients / amplitude profile.}\label{tab:external}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccc}
\toprule
Model & \makecell{AUC\\PTB-XL} & \makecell{AUC\\external} & \makecell{IG $\rho$ seed /\\untrained} & \makecell{QRS lift\\G$\times$I} & \makecell{QRS lift\\IG} & \makecell{QRS lift\\amplitude} & \makecell{Deletion gain\\IG / ampl.}\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))
else:
    pending("tab_external", "tab:external", "External replication")

# ---------------------------------------------------------------- cost
EF = os.path.join(S, "efficiency.json")
if os.path.exists(EF):
    ef = json.load(open(EF))
    rows = ["%s & %.2f & %.2f & %.1f & %.1f \\\\" % (AN[a], ef["arch"][a]["params_m"], ef["arch"][a]["gflops"], ef["arch"][a]["cpu_ms"],
                                                    prt[prt.model == a].train_seconds.mean() / 60) for a in ARCH]
    rows2 = ["%s & %s & %d & & %.1f \\\\" % (MN[m], FAMILY[m], ef["method"][m]["passes"], ef["method"][m]["cpu_ms_per_item"]) for m in ALL if m in ef["method"]]
    w("tab_cost", r"""\begin{table}
\centering
\caption{Measured cost on one machine. Top: classifiers (parameters, forward floating-point operations for one $12\times1000$ recording, median CPU latency at batch 1 with four threads, mean training time per run on the shared GPU). Bottom: attribution methods for XResNet1d-50 (network passes per item and median CPU time per item at batch 32, eight threads). $\downarrow$: lower is better.}\label{tab:cost}
\footnotesize
\begin{tabular}{lrrrr}
\toprule
Model & Params (M) $\downarrow$ & GFLOPs $\downarrow$ & CPU ms $\downarrow$ & Train min $\downarrow$\\
\midrule
%s
\midrule
Method & Family & Passes & \multicolumn{2}{r}{CPU ms per item $\downarrow$}\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % ("\n".join(rows), "\n".join(rows2)))
else:
    pending("tab_cost", "tab:cost", "Measured cost of classifiers and attribution methods")
print("TABLES_DONE")
