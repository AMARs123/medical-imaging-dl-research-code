"""LaTeX tables generated from results/statistics (no number is typed by hand).
Usage: python make_tables.py <Paper_01 dir>   ->  paper/tables/*.tex
"""
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
ST = os.path.join(R, "results", "statistics")
OUT = os.path.join(R, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
NICE = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "efficientnet_b0": "EfficientNet-B0",
        "convnext_tiny": "ConvNeXt-T", "mobilevit_s": "MobileViT-S", "vit_small_patch16_224": "ViT-S/16", "deit_small_patch16_224": "DeiT-S/16",
        "swin_tiny_patch4_window7_224": "Swin-T"}
ORDER = list(NICE)
RULE = {"slice": "Image level", "authors": "Authors' folds", "pid": "ID string", "root": "ID root"}


def pv(p):
    if pd.isna(p):
        return "n/a"
    return "$<$0.001" if p < 0.001 else "%.3f" % p


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def sci(p):
    if p >= 0.001:
        return "%.3f" % p
    mant, ex = ("%.1e" % p).split("e")
    return "%s\\times10^{%d}" % (mant, int(ex))


def ms(m, s, d=3):
    return ("%.*f $\\pm$ %.*f" % (d, m, d, s)) if not pd.isna(s) else "%.*f" % (d, m)


def write(name, body):
    open(os.path.join(OUT, name + ".tex"), "w", encoding="utf8").write(body)
    print("table", name)


# ---------------------------------------------------------------- fold-formation rules
a = pd.read_csv(os.path.join(ST, "A_pooled.csv"))
pa = pd.read_csv(os.path.join(ST, "A_paired.csv"))
dpath = os.path.join(ST, "D_partition.csv")
D = pd.read_csv(dpath) if os.path.exists(dpath) else None
lines = []
for m in ["resnet50", "vit_small_patch16_224"]:
    first = True
    for pr in ["slice", "authors", "pid", "root"]:
        r = a[(a.model == m) & (a.protocol == pr)].iloc[0]
        q = pa[(pa.model == m) & (pa.protocol == pr)]
        diff = "n/a" if q.empty else "%+.3f [%+.3f, %+.3f]" % (q.diff_macro_f1.iloc[0], q.diff_ci_lo.iloc[0], q.diff_ci_hi.iloc[0])
        pm = "n/a" if q.empty else pv(q.p_holm.iloc[0])
        spread = "n/a"
        if D is not None and pr in ("pid", "root"):
            v = D[(D.model == m) & (D.protocol == pr)].macro_f1
            if len(v) > 1:
                spread = "%.3f--%.3f (%d)" % (v.min(), v.max(), len(v))
        lines.append("%s & %s & %.3f [%.3f, %.3f] & %.3f & %s & %s & %s \\\\" % (
            "\\multirow{4}{*}{%s}" % NICE[m] if first else "", RULE[pr], r.macro_f1, r.ci_lo, r.ci_hi, r.accuracy, diff, pm, spread))
        first = False
    lines.append("\\midrule")
lines = lines[:-1]
write("tab_rules", r"""\begin{table}
\centering
\caption{Splitting rules. Macro-F1 of the pooled test predictions over all 3064 slices with 95\%% cluster-bootstrap CI (resampling identifier roots), accuracy, difference to the ID-root rule on the same images with its bootstrap CI, Holm-corrected exact McNemar $p$, and the range of macro-F1 over repeated splits (number of splits).}\label{tab:rules}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llccccc}
\toprule
Model & Rule & Macro-F1 [95\%% CI] & Acc. & $\Delta$ vs ID root [95\%% CI] & $p_\mathrm{Holm}$ & Range over splits\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines))

# ---------------------------------------------------------------- architectures
b = pd.read_csv(os.path.join(ST, "B_table.csv")).set_index("model")
fr = pd.read_csv(os.path.join(ST, "B_friedman.csv")).set_index("model")
dg = pd.read_csv(os.path.join(ST, "evidence_diagnostics.csv"))
dg = dg[(dg.protocol == "root") & (dg.lambda_sal == 0) & ((dg.split_seed.fillna(0) == 0) & (dg.dilate.fillna(1) == 1) & (dg.train_frac.fillna(1) == 1))]
dg["lift"] = dg.pointing_cell / dg.pointing_cell_chance
lift = dg.groupby("model").lift.agg(["mean", "std"])
runs_all = pd.read_csv(os.path.join(R, "results", "processed", "all_runs.csv"))
Bm = runs_all[(runs_all.protocol == "root") & (runs_all.lambda_sal == 0) & runs_all.main.astype(bool)]
med = Bm.groupby("model").macro_f1.median()
nrun = Bm.groupby("model").macro_f1.count()
nseed = Bm.groupby("model").seed.nunique()
FAM = {"resnet50": "CNN", "densenet121": "CNN", "efficientnet_b0": "CNN", "convnext_tiny": "CNN", "mobilevit_s": "hyb",
       "vit_small_patch16_224": "ViT", "deit_small_patch16_224": "ViT", "swin_tiny_patch4_window7_224": "ViT"}
lines, prev = [], None
for m in [m for m in ORDER if m in b.index]:
    if prev is not None and FAM[m] != FAM[prev]:
        lines.append("\\midrule")
    prev = m
    r = b.loc[m]
    p = "ref." if m not in fr.index else pv(fr.loc[m, "p_holm"])
    lines.append("%s & %s & %.3f & %s & %s & %s & %s & %s \\\\" % (
        NICE[m], ms(r.macro_f1_mean, r.macro_f1_std), med[m], p, ms(r.macro_auc_mean, r.macro_auc_std),
        ms(r.ece_mean, r.ece_std), ms(r.loc_mass_inside_mean, r.loc_mass_inside_std, 2), ms(lift.loc[m, "mean"], lift.loc[m, "std"], 1)))
f0 = pd.read_csv(os.path.join(ST, "B_friedman.csv")).iloc[0]
NB_ = int(nrun.min()) if nrun.min() == nrun.max() else None
write("tab_arch", r"""\begin{table}
\centering
\caption{Architectures under ID-root folds without the evidence penalty: mean $\pm$ SD and median over %s runs per model (5 folds $\times$ %d seeds). $p_\mathrm{Holm}$: Wilcoxon signed-rank test of macro-F1 against the best model, Holm-corrected (Friedman $\chi^2=%.1f$, $p=%s$, %d blocks). Evidence inside: share of positive evidence inside the tumour (pixel level). Lift: pointing at map resolution divided by its chance level (1 = chance). Rules separate CNNs, the CNN--transformer hybrid and the vision transformers. $\uparrow$: higher is better; $\downarrow$: lower is better.}\label{tab:arch}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccc}
\toprule
Model & Macro-F1 $\uparrow$ & Median & $p_\mathrm{Holm}$ & Macro-AUC $\uparrow$ & ECE $\downarrow$ & Evidence inside $\uparrow$ & Lift $\uparrow$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (NB_ if NB_ else "%d--%d" % (nrun.min(), nrun.max()), int(nseed.max()), f0.friedman_chi2, sci(f0.friedman_p), f0.n_blocks, "\n".join(lines)))

# ---------------------------------------------------------------- evidence supervision
from scipy.stats import wilcoxon  # noqa: E402

MC4 = ["resnet50", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]
dgc = pd.read_csv(os.path.join(ST, "evidence_diagnostics.csv"))
dgc = dgc[(dgc.protocol == "root") & ((dgc.split_seed.fillna(0) == 0) & (dgc.dilate.fillna(1) == 1) & (dgc.train_frac.fillna(1) == 1))]
dgc["lift"] = dgc.pointing_cell / dgc.pointing_cell_chance
# the weight sweep is a sensitivity analysis on seeds 0-2 for every weight, so every cell has the same 15 runs
Cs = runs_all[(runs_all.protocol == "root") & runs_all.main.astype(bool) & runs_all.model.isin(MC4) & (runs_all.seed <= 2)]
Cs = Cs.merge(dgc[["exp_id", "lift"]], on="exp_id", how="left")
MET3 = ["macro_f1", "loc_pointing", "loc_mass_inside"]
tests = []
for m in MC4:
    ref = Cs[(Cs.model == m) & (Cs.lambda_sal == 0)].set_index(["fold", "seed"])
    for lam in (0.25, 0.5, 1.0, 2.0):
        cur = Cs[(Cs.model == m) & (Cs.lambda_sal == lam)].set_index(["fold", "seed"])
        j = cur.join(ref, lsuffix="_x", rsuffix="_0", how="inner")
        for met in MET3:
            d = j[met + "_x"] - j[met + "_0"]
            tests.append(dict(model=m, lambda_sal=lam, metric=met, n=len(j), p=wilcoxon(d).pvalue if len(j) >= 5 and not np.allclose(d, 0) else 1.0))
tests = pd.DataFrame(tests)
for met, idx in tests.groupby("metric").groups.items():
    tests.loc[idx, "p_holm"] = holm(tests.loc[idx, "p"].values)
lines = []
for m in MC4:
    first = True
    lams = sorted(Cs[Cs.model == m].lambda_sal.unique())
    for lam in lams:
        g = Cs[(Cs.model == m) & (Cs.lambda_sal == lam)]

        def p_of(met):
            q = tests[(tests.model == m) & (tests.lambda_sal == lam) & (tests.metric == met)]
            return "" if q.empty else "$^{%s}$" % ("*" if q.p_holm.iloc[0] < 0.05 else "")
        lines.append("%s & %s & %s%s & %s & %s%s & %s%s & %.1f & %d \\\\" % (
            "\\multirow{%d}{*}{%s}" % (len(lams), NICE[m]) if first else "", ("%g" % lam),
            ms(g.macro_f1.mean(), g.macro_f1.std()), p_of("macro_f1"), ms(g.ece.mean(), g.ece.std()),
            ms(g.loc_pointing.mean(), g.loc_pointing.std(), 2), p_of("loc_pointing"), ms(g.loc_mass_inside.mean(), g.loc_mass_inside.std(), 2),
            p_of("loc_mass_inside"), g.lift.mean(), len(g)))
        first = False
    lines.append("\\midrule")
tests.to_csv(os.path.join(ST, "C_sweep_tests_seeds012.csv"), index=False)
write("tab_lambda", r"""\begin{table}
\centering
\caption{Sensitivity to the penalty weight $\lambda$ (ID-root folds; mean $\pm$ SD over the same 15 runs per cell: 5 folds $\times$ seeds 0--2). $^{*}$: differs from $\lambda=0$ of the same model (paired Wilcoxon signed-rank test over the 15 runs, Holm-corrected within each metric over the 16 model--weight pairs, $p<0.05$). Pointing: pixel-level pointing game. Lift: pointing at map resolution divided by chance. The main comparison of $\lambda=0.25$ with $\lambda=0$ over five seeds is in Table~\ref{tab:penalty}.}\label{tab:lambda}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccccr}
\toprule
Model & $\lambda$ & Macro-F1 & ECE & Pointing & Evidence inside & Lift & $n$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines[:-1]))

# ---------------------------------------------------------------- appendix: per-fold results
runs = pd.read_csv(os.path.join(R, "results", "processed", "all_runs.csv"))
regx = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
for col, v in [("split_seed", 0), ("dilate", 1), ("train_frac", 1.0)]:
    regx[col] = regx[col].fillna(v) if col in regx else v
runs = runs.drop(columns=[c for c in ("dilate", "train_frac") if c in runs]).merge(regx[["exp_id", "dilate", "train_frac"]], on="exp_id")
base = runs[(runs.protocol == "root") & (runs.split_seed.fillna(0) == 0) & (runs.lambda_sal == 0) & (runs.dilate == 1) & (runs.train_frac == 1.0)]
pf = base.groupby(["model", "fold"]).macro_f1.agg(["mean", "std"])
MB = [m for m in ORDER if m in base.model.unique()]
lines = []
for m in MB:
    cells = " & ".join(ms(pf.loc[(m, f), "mean"], pf.loc[(m, f), "std"]) for f in range(5))
    lines.append("%s & %s \\\\" % (NICE[m], cells))
tf = base.groupby("fold")[["split_n_train", "split_n_val", "split_n_test"]].first()
lines.append("\\midrule")
lines.append("Slices (train/val/test) & " + " & ".join("%d/%d/%d" % tuple(tf.loc[f]) for f in range(5)) + " \\\\")
write("tab_perfold", r"""\begin{table}
\centering
\caption{Macro-F1 per ID-root test fold (mean $\pm$ SD over %d seeds), $\lambda=0$, and split sizes (train/validation/test slices).}\label{tab:perfold}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
Model & Fold 0 & Fold 1 & Fold 2 & Fold 3 & Fold 4\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (base.groupby("model").seed.nunique().max(), "\n".join(lines)))

# ---------------------------------------------------------------- appendix: all scalar metrics per architecture
cols = [("accuracy", "Acc. $\\uparrow$"), ("balanced_accuracy", "Bal. acc. $\\uparrow$"), ("nll", "NLL $\\downarrow$"),
        ("loc_pointing", "Pointing $\\uparrow$"), ("loc_iou_area", "IoU$_\\mathrm{area}$ $\\uparrow$"), ("best_epoch", "Best ep."),
        ("train_seconds", "Time (s)"), ("peak_gpu_mb", "GPU (MB)")]
g = base.groupby("model")
lines = []
for m in MB:
    vals = []
    for c, _ in cols:
        mu, sd = g[c].mean()[m], g[c].std()[m]
        vals.append("%.0f" % mu if c in ("train_seconds", "peak_gpu_mb") else ("%.1f" % mu if c == "best_epoch" else ms(mu, sd)))
    lines.append("%s & %s \\\\" % (NICE[m], " & ".join(vals)))
write("tab_allmetrics", r"""\begin{table}
\centering
\caption{Further test metrics of the eight architectures (ID-root folds, $\lambda=0$; mean $\pm$ SD over %d runs per model; best epoch, training time per run and peak training GPU memory are means). Pointing: pixel-level pointing game (Eq.~\ref{eq:point}). IoU$_\mathrm{area}$: overlap of the strongest map pixels with the tumour (Eq.~\ref{eq:iou}).}\label{tab:allmetrics}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{l%s}
\toprule
Model & %s\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (g.macro_f1.count().max(), "c" * len(cols), " & ".join(h for _, h in cols), "\n".join(lines)))

# ---------------------------------------------------------------- faithfulness
fp = os.path.join(ST, "faithfulness_summary.csv")
if os.path.exists(fp):
    f = pd.read_csv(fp)
    lines, prev = [], None
    star = lambda p: "$^{*}$" if (not pd.isna(p) and p < 0.05) else ""  # noqa: E731
    nr = set()
    for m in ORDER:
        for lam in sorted(f[f.model == m].lambda_sal.unique()):
            if lam not in (0.0, 0.25):
                continue
            if prev is not None and m != prev:
                lines.append("\\midrule" if FAM[m] != FAM[prev] else "\\addlinespace[2pt]")
            prev = m
            r = f[(f.model == m) & (f.lambda_sal == lam)].iloc[0]
            sg = r.get("del_auc_signed_mean", np.nan)
            nr.add(int(r.n_runs))
            lines.append("%s & %g & %.3f%s & %.3f & %.3f & %s%s & %.3f \\\\" % (
                NICE[m], lam, r.drop_tumour_mean, star(r.get("p_tumour_vs_control_holm")), r.drop_control_mean,
                r.del_auc_evidence_mean, "n/a" if pd.isna(sg) else "%.3f" % sg, star(r.get("p_signed_vs_random_holm", np.nan)),
                r.del_auc_random_mean))
    write("tab_faith", r"""\begin{table}
\centering
\caption{Faithfulness tests on the test folds without and with the evidence penalty (ID-root folds; means over %s runs per row). Occlusion: mean drop in the probability of the predicted class when the tumour, or the same shape elsewhere (control), is filled with the image mean ($\uparrow$ for the tumour means more reliance on it). Deletion: area under the probability curve when up to 50\%% of $16\times16$ patches are removed in order of positive evidence, of signed evidence, or at random ($\downarrow$: a lower area means a faster drop, so a more faithful map). $^{*}$: tumour drop larger than control drop, or signed deletion area smaller than random, paired Wilcoxon signed-rank test over runs, Holm-corrected, $p<0.05$.}\label{tab:faith}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
 & & \multicolumn{2}{c}{Occlusion drop} & \multicolumn{3}{c}{Deletion AUC $\downarrow$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
Model & $\lambda$ & Tumour & Control & Positive & Signed & Random\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % ("/".join(str(x) for x in sorted(nr)), "\n".join(lines)))

# ================================================================ Research_Expert round (2026-10-16): new tables


def boot_ci(d, n=4000, seed=2026):
    d = np.asarray(d, float)
    rg = np.random.default_rng(seed)
    bs = d[rg.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (effect size of the Wilcoxon signed-rank test)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    rk = rankdata(np.abs(d))
    return float((rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum())


# ---------------------------------------------------------------- main penalty comparison: lambda 0.25 vs 0, all seeds
FR = os.path.join(ST, "faithfulness_runs.csv")
fr_runs = pd.read_csv(FR) if os.path.exists(FR) else pd.DataFrame()
pen_rows, lines = [], []
for m in MC4:
    a = runs_all[(runs_all.model == m) & (runs_all.protocol == "root") & runs_all.main.astype(bool)]
    a = a.merge(dgc[["exp_id", "lift"]], on="exp_id", how="left")
    if not fr_runs.empty:
        a = a.merge(fr_runs[["exp_id", "drop_tumour", "del_auc_signed"]], on="exp_id", how="left")
    j = a[a.lambda_sal == 0.25].set_index(["fold", "seed"]).join(a[a.lambda_sal == 0].set_index(["fold", "seed"]), lsuffix="_1", rsuffix="_0", how="inner")
    cells = []
    for met in ["macro_f1", "loc_mass_inside", "lift", "drop_tumour", "del_auc_signed"]:
        if met + "_1" not in j:
            cells.append("n/a")
            continue
        d = (j[met + "_1"] - j[met + "_0"]).dropna()
        p = wilcoxon(d).pvalue if len(d) >= 5 and not np.allclose(d, 0) else 1.0
        lo, hi = boot_ci(d)
        pen_rows.append(dict(model=m, metric=met, n_pairs=len(d), mean_0=j[met + "_0"].mean(), mean_025=j[met + "_1"].mean(), diff=d.mean(),
                             ci_lo=lo, ci_hi=hi, p_wilcoxon=p, r_rb=rank_biserial(d)))
PEN = pd.DataFrame(pen_rows)
for met, idx in PEN.groupby("metric").groups.items():
    PEN.loc[idx, "p_holm"] = holm(PEN.loc[idx, "p_wilcoxon"].values)
PEN.to_csv(os.path.join(ST, "penalty_main_tests.csv"), index=False)
DEC = {"macro_f1": 3, "loc_mass_inside": 3, "lift": 1, "drop_tumour": 3, "del_auc_signed": 3}
lines = []
for m in MC4:
    cells = []
    for met in ["macro_f1", "loc_mass_inside", "lift", "drop_tumour", "del_auc_signed"]:
        q = PEN[(PEN.model == m) & (PEN.metric == met)]
        if q.empty:
            cells.append("n/a")
            continue
        q = q.iloc[0]
        k = DEC[met]
        cells.append(r"\makecell{%+.*f%s\\ {\scriptsize [%+.*f, %+.*f]}}" % (k, q["diff"], "$^{*}$" if q.p_holm < 0.05 else "", k, q.ci_lo, k, q.ci_hi))
    npair = int(PEN[PEN.model == m].n_pairs.max())
    lines.append("%s & %s & %d \\\\" % (NICE[m], " & ".join(cells), npair))
write("tab_penalty", r"""\begin{table}
\centering
\caption{Effect of the evidence penalty at $\lambda=0.25$: mean paired difference from $\lambda=0$ on the same (fold, seed) pairs, with a 95\%% bootstrap interval over the pairs ($n$ pairs: 5 folds $\times$ 5 seeds). $^{*}$: paired Wilcoxon signed-rank test, Holm-corrected over the four architectures within each metric, $p<0.05$. Evidence inside: pixel-level share of positive evidence inside the tumour. Lift: pointing at map resolution divided by chance. Occlusion: drop in the predicted-class probability when the tumour is filled with the image mean (a change means a change in reliance on the tumour). Deletion: signed-evidence deletion area (a negative change means a more faithful map).}\label{tab:penalty}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccr}
\toprule
Model & $\Delta$ Macro-F1 & \makecell{$\Delta$ Evidence\\inside} & $\Delta$ Lift & \makecell{$\Delta$ Occlusion\\drop} & \makecell{$\Delta$ Deletion\\AUC} & $n$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines))

# ---------------------------------------------------------------- efficiency (measured on one GPU and one CPU)
EF = os.path.join(R, "results", "processed", "efficiency.csv")
if os.path.exists(EF):
    ef = pd.read_csv(EF).set_index("model")
    tt = Bm.groupby("model")[["train_seconds", "peak_gpu_mb"]].mean()
    lines, prev = [], None
    for m in [m for m in ORDER if m in ef.index]:
        if prev is not None and FAM[m] != FAM[prev]:
            lines.append("\\midrule")
        prev = m
        e = ef.loc[m]
        lines.append("%s & %.1f & %.2f & %.1f & %.2f & %.0f & %.1f & %.1f \\\\" % (
            NICE[m], e.params_m, e.gflops, e.gpu_ms_b1, e.gpu_ms_b32_per, e.cpu_ms_b1_t4,
            tt.loc[m, "train_seconds"] / 60 if m in tt.index else np.nan, tt.loc[m, "peak_gpu_mb"] / 1024 if m in tt.index else np.nan))
    envf = EF.replace(".csv", "_environment.txt")
    write("tab_eff", r"""\begin{table}
\centering
\caption{Measured cost of the eight evidence networks for one $224\times224$ slice. GFLOPs: forward floating-point operations (a multiply--add counts as two). GPU latency: median over 100 timed passes after 20 warm-up passes, mixed precision, batch 1 and per image at batch 32, on the shared NVIDIA RTX 5000 Ada. CPU latency: median over 30 passes, batch 1, four threads, fp32. Training: mean wall-clock minutes per 30-epoch run and peak training GPU memory over the main runs. All measurements on the same machine; GPU times are indicative because the GPU was shared with other jobs. $\downarrow$: lower is better.}\label{tab:eff}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccc}
\toprule
Model & \makecell{Params\\(M) $\downarrow$} & \makecell{GFLOPs\\$\downarrow$} & \makecell{GPU ms\\(b1) $\downarrow$} & \makecell{GPU ms/img\\(b32) $\downarrow$} & \makecell{CPU ms\\$\downarrow$} & \makecell{Train\\min $\downarrow$} & \makecell{Train\\GB $\downarrow$}\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines))

# ---------------------------------------------------------------- external test (independent part of BRISC 2025)
EX = os.path.join(R, "results", "processed", "external_runs.csv")
if os.path.exists(EX):
    ex = pd.read_csv(EX)
    ex = ex[ex.exp_id.isin(runs_all.exp_id)]
    ins = runs_all.set_index("exp_id")
    lines, prev, ns = [], None, set()
    for m in [m for m in ORDER if m in ex.model.unique()]:
        for lam in (0.0, 0.25):
            g = ex[(ex.model == m) & (ex.lambda_sal == lam)]
            if g.empty:
                continue
            if prev is not None and m != prev:
                lines.append("\\midrule" if FAM[m] != FAM[prev] else "\\addlinespace[2pt]")
            prev = m
            ns.add(len(g))
            internal = ins.loc[g.exp_id, "accuracy"].mean()
            lines.append("%s & %g & %.3f & %s & %s & %s & %s & %s \\\\" % (
                NICE[m], lam, internal, ms(g.accuracy.mean(), g.accuracy.std()), ms(g.recall_meningioma.mean(), g.recall_meningioma.std()),
                ms(g.pointing_lift.mean(), g.pointing_lift.std(), 1), ms(g.loc_mass_inside.mean(), g.loc_mass_inside.std(), 2),
                "%.3f / %.3f" % (g.occlusion_tumour.mean(), g.occlusion_control.mean())))
    n_ext = int(ex.n.iloc[0])
    write("tab_external", r"""\begin{table}
\centering
\caption{External test on the %d BRISC slices that are not copies of Figshare slices (mean $\pm$ SD over %s runs per row; no setting was tuned on these data). Internal: accuracy on the Figshare test folds of the same runs. Recall: meningioma recall (meningioma is %d of the %d slices). Lift and evidence inside are computed against the BRISC tumour masks with the same definitions as in Table~\ref{tab:arch}. Occlusion: drop in predicted-class probability when the tumour / a control region of the same shape is filled with the image mean. $\uparrow$: higher is better.}\label{tab:external}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccc}
\toprule
Model & $\lambda$ & \makecell{Internal\\acc.} & \makecell{External\\acc. $\uparrow$} & \makecell{Recall\\$\uparrow$} & \makecell{Lift\\$\uparrow$} & \makecell{Evidence\\inside $\uparrow$} & \makecell{Occlusion\\tumour / control}\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (n_ext, "/".join(str(x) for x in sorted(ns)), int(pd.read_csv(os.path.join(R, "data", "external", "processed", "external_test_set.csv")).label.eq(0).sum()), n_ext, "\n".join(lines)))

# ---------------------------------------------------------------- learning-rate check (validation macro-F1 only)
LRR = os.path.join(R, "experiments", "lr_registry.csv")
if os.path.exists(LRR):
    import json as _json
    lr_reg = pd.read_csv(LRR)
    rows = []
    for r in lr_reg.itertuples():
        p = os.path.join(R, "experiments", r.exp_id, "results.json")
        if os.path.exists(p):
            rows.append(dict(model=r.model, fold=r.fold, lr=r.lr, val=_json.load(open(p))["best_val_macro_f1"]))
    ref = runs_all[(runs_all.protocol == "root") & runs_all.main.astype(bool) & (runs_all.lambda_sal == 0) & (runs_all.seed == 0)]
    rows += [dict(model=r.model, fold=r.fold, lr=1e-4, val=r.best_val_macro_f1) for r in ref.itertuples()]
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(ST, "lr_check.csv"), index=False)
    piv = L.pivot_table(index="model", columns="lr", values="val", aggfunc=["mean", "count"])
    lines, out = [], []
    for m in [m for m in ORDER if m in piv.index]:
        mu = piv["mean"].loc[m]
        cnt = piv["count"].loc[m]
        if (cnt.reindex([3e-5, 1e-4, 3e-4]).fillna(0) < 5).any():
            continue
        best = mu.idxmax()
        w = L[L.model == m].pivot_table(index="fold", columns="lr", values="val")
        d = w[best] - w[1e-4]
        p = wilcoxon(d).pvalue if best != 1e-4 and not np.allclose(d, 0) else np.nan
        out.append(dict(model=m, best_lr=best, gap=mu[best] - mu[1e-4], p=p))
        lines.append("%s & %.3f & %.3f & %.3f & %s & %.3f \\\\" % (NICE[m], mu[3e-5], mu[1e-4], mu[3e-4], "%g" % best, mu[best] - mu[1e-4]))
    pd.DataFrame(out).to_csv(os.path.join(ST, "lr_check_summary.csv"), index=False)
    if lines:
        write("tab_lr", r"""\begin{table}
\centering
\caption{Learning-rate check with the same three-value search for every architecture: mean best validation macro-F1 over the five ID-root folds (seed 0, $\lambda=0$; the test folds were not used). The shared recipe uses $10^{-4}$. Gap: best value minus the value at $10^{-4}$ ($\geq 0$ by construction).}\label{tab:lr}
\footnotesize
\begin{tabular}{lccccc}
\toprule
Model & $3\times10^{-5}$ & $10^{-4}$ (used) & $3\times10^{-4}$ & Best & Gap\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines))

# ---------------------------------------------------------------- equal budget per compared model (configuration, not results)
TAG = {"resnet50": "a1\\_in1k", "densenet121": "tv\\_in1k", "efficientnet_b0": "ra\\_in1k", "convnext_tiny": "in12k\\_ft\\_in1k",
       "mobilevit_s": "cvnets\\_in1k", "vit_small_patch16_224": "augreg\\_in21k\\_ft\\_in1k", "deit_small_patch16_224": "fb\\_in1k",
       "swin_tiny_patch4_window7_224": "ms\\_in1k"}
lines = []
for m in [m for m in ORDER if m in Bm.model.unique()]:
    lines.append("%s & %s & %s & 30 & 32 & 3 & %d $\\times$ 5 \\\\" % (
        NICE[m], {"CNN": "CNN", "hyb": "hybrid", "ViT": "transformer"}[FAM[m]], TAG[m], nseed[m]))
write("tab_budget", r"""\begin{table}
\centering
\caption{Training budget of every compared model. All models share the input pipeline, augmentation, loss, optimiser (AdamW, weight decay 0.05, one-cycle schedule), the learning-rate search over $\{3\times10^{-5}, 10^{-4}, 3\times10^{-4}\}$ judged on validation folds (Table~\ref{tab:lr}), the selection rule (best validation macro-F1), the folds and the hardware. Pre-trained weights are the \texttt{timm} defaults (tag shown).}\label{tab:budget}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllcccc}
\toprule
Model & Family & Weights tag & Epochs & Batch & LR values & Seeds $\times$ folds\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines))

# ---------------------------------------------------------------- BRISC overlap with Figshare
OV = os.path.join(R, "data", "external", "processed", "meta.csv")
if os.path.exists(OV):
    bm = pd.read_csv(OV)
    dup = bm[bm.near_duplicate]
    ext = pd.read_csv(os.path.join(R, "data", "external", "processed", "external_test_set.csv"))
    te = bm[bm.brisc_split == "test"]
    leak = te[te.near_duplicate].best_figshare_idx.isin(bm[(bm.brisc_split == "train") & bm.near_duplicate].best_figshare_idx).sum()
    figm = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
    labdis = int((figm.label.values[dup.best_figshare_idx.values] != dup.label.values).sum())
    rows = [("BRISC tumour slices with masks (train + test parts)", "%d" % len(bm)),
            ("copies of a Figshare slice (correlation $\\geq 0.95$)", "%d (%.1f\\%%)" % (len(dup), 100 * len(dup) / len(bm))),
            ("\\quad of which mirrored left--right", "%d" % dup.mirrored_match.sum()),
            ("distinct Figshare slices copied", "%d of %d (%.1f\\%%)" % (dup.best_figshare_idx.nunique(), len(figm), 100 * dup.best_figshare_idx.nunique() / len(figm))),
            ("copies with a different tumour label", "%d" % labdis),
            ("BRISC test slices whose Figshare source is also in BRISC train", "%d of %d" % (leak, len(te))),
            ("slices independent of Figshare", "%d" % (~bm.near_duplicate).sum()),
            ("\\quad after removing repeats inside BRISC (external test set)", "%d (%d meningioma, %d glioma, %d pituitary)" % (
                len(ext), (ext.label == 0).sum(), (ext.label == 1).sum(), (ext.label == 2).sum()))]
    bc = bm.best_corr.values
    gap_sentence = ("The best correlations were at most %.2f below the threshold of 0.95 and at least %.3f above it, so the threshold "
                    "separates copies from different images." % (bc[bc < 0.95].max(), bc[bc >= 0.95].min()))
    write("tab_overlap", r"""\begin{table}
\centering
\caption{Overlap between BRISC 2025 and the Figshare dataset. Every BRISC tumour slice was compared with every Figshare slice and its mirror image (Pearson correlation of $64\times64$ down-sampled images, Eq.~\ref{eq:dup}). %s Pairs on both sides of the threshold were checked visually.}\label{tab:overlap}
\footnotesize
\begin{tabular}{p{0.56\textwidth}p{0.36\textwidth}}
\toprule
Quantity & Count\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (gap_sentence, "\n".join("%s & %s \\\\" % r for r in rows)))

# ---------------------------------------------------------------- dataset contents (from the processed meta file)
fm = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"), dtype={"pid": str, "root": str})
CN = ["Meningioma", "Glioma", "Pituitary"]
cnt = [int((fm.label == k).sum()) for k in range(3)]
ids = [fm[fm.label == k].pid.nunique() for k in range(3)]
roots = [fm[fm.label == k].root.nunique() for k in range(3)]
area = [100 * fm[fm.label == k].mask_fraction.mean() for k in range(3)]
write("tab_data", r"""\begin{table}
\centering
\caption{Contents of the Figshare dataset, checked file by file. Identifier root: identifier string without a final letter (Eq.~\ref{eq:root}). Tumour area: mean share of the slice covered by the expert outline.}\label{tab:data}
\small
\begin{tabular}{lrrrr}
\toprule
 & %s & Total\\
\midrule
Slices & %s & %d\\
Identifier strings & %s & %d\\
Identifier roots & %s & %d\\
Mean tumour area (\%% of slice) & %s & %.1f\\
\bottomrule
\end{tabular}
\end{table}
""" % (" & ".join(CN), " & ".join(map(str, cnt)), len(fm), " & ".join(map(str, ids)), fm.pid.nunique(), " & ".join(map(str, roots)),
       fm.root.nunique(), " & ".join("%.1f" % a for a in area), 100 * fm.mask_fraction.mean()))

# ---------------------------------------------------------------- experiment groups (run counts from the registries)
regp = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
lrp = pd.read_csv(LRR) if os.path.exists(LRR) else pd.DataFrame(columns=regp.columns)
gB = regp[(regp.phase == "B_baseline")]
gC = regp[regp.phase == "C_saliency"]
nG = int((regp.note == "added seeds, Research_Expert round").sum())
rowsP = [("A", "Splitting rules", "ResNet-50, ViT-S", r"4 rules $\times$ 5 folds; seed 0", int((regp.phase == "A_protocol").sum())),
         ("B", "Architectures", "all 8", r"ID root; 5 folds $\times$ 5 seeds\textsuperscript{a}", len(gB)),
         ("C", "Penalty weight", "ResNet-50, ConvNeXt-T, ViT-S, Swin-T",
          r"$\lambda\in\{0.25,0.5,1,2\}$; 5 folds $\times$ 3 seeds (5 seeds for $\lambda=0.25$)", len(gC)),
         ("D", "Repeated splits", "ResNet-50, ViT-S", "ID string and ID root; split seeds 1--4; 5 folds", int((regp.phase == "D_partition").sum())),
         ("E", "Mask widening", "ConvNeXt-T, ViT-S", r"$r\in\{0,2\}$; $\lambda=0.25$; 5 folds $\times$ 3 seeds", int((regp.phase == "E_dilation").sum())),
         ("F", "Amount of data", "ConvNeXt-T, ViT-S", r"25\% or 50\% of training roots; $\lambda\in\{0,0.25\}$; 5 folds $\times$ 3 seeds", int((regp.phase == "F_datafrac").sum())),
         ("H", "Learning rate", "all 8", r"$3\times10^{-5}$ and $3\times10^{-4}$; 5 folds; seed 0", len(lrp))]
write("tab_plan", r"""\begin{table}
\centering
\caption{Experiment groups and their run counts (%d runs in total). Unless stated, grouped splits use split seed 0, mask widening $r=1$ and the full training set. Group G (%d runs, added in revision) supplied seeds 3 and 4 and MobileViT-S to groups B and C; its runs are counted there.}\label{tab:plan}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llp{3.3cm}p{5.0cm}r}
\toprule
Group & Question & Models & Settings & Runs\\
\midrule
%s
\bottomrule
\multicolumn{5}{l}{\footnotesize \textsuperscript{a} including the seed-0 ID-root runs of group A.}
\end{tabular}
\end{table}
""" % (len(regp) + len(lrp), nG, "\n".join("%s & %s & %s & %s & %d\\\\" % r for r in rowsP)))

# ---------------------------------------------------------------- values behind Fig. per-class F1
pc = []
for r in Bm.itertuples():
    p = os.path.join(R, "experiments", r.exp_id, "results.json")
    if os.path.exists(p):
        v = _json.load(open(p))["test"]["per_class_f1"] if "_json" in dir() else __import__("json").load(open(p))["test"]["per_class_f1"]
        pc.append(dict(model=r.model, men=v[0], gli=v[1], pit=v[2]))
pc = pd.DataFrame(pc)
lines = []
for m in [m for m in ORDER if m in pc.model.unique()]:
    g = pc[pc.model == m]
    lines.append("%s & %s & %s & %s & %d \\\\" % (NICE[m], ms(g.men.mean(), g.men.std()), ms(g.gli.mean(), g.gli.std()), ms(g.pit.mean(), g.pit.std()), len(g)))
write("tab_perclassvals", r"""\begin{table}
\centering
\caption{Values behind Fig.~\ref{fig:perclass}: per-class F1 under ID-root folds without the penalty (mean $\pm$ SD over $n$ runs).}\label{tab:perclassvals}
\footnotesize
\begin{tabular}{lcccr}
\toprule
Model & Meningioma & Glioma & Pituitary & $n$\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines))

# ---------------------------------------------------------------- values behind Fig. robustness
RR = os.path.join(ST, "robustness_runs.csv")
if os.path.exists(RR):
    rb = pd.read_csv(RR)
    mods = [m for m in MC4 if m in rb.model.unique()]
    corr = list(dict.fromkeys(rb.corruption))
    piv = rb.groupby(["corruption", "model", "lambda_sal"]).macro_f1.mean()
    lines = []
    for c in corr:
        cells = []
        for m in mods:
            for lam in (0.0, 0.25):
                cells.append("%.3f" % piv.get((c, m, lam), np.nan))
        lines.append("%s & %s \\\\" % (c.replace("_", " "), " & ".join(cells)))
    head1 = " & ".join(r"\multicolumn{2}{c}{%s}" % NICE[m] for m in mods)
    head2 = " & ".join(["0 & 0.25"] * len(mods))
    write("tab_robustvals", r"""\begin{table}
\centering
\caption{Values behind Fig.~\ref{fig:robust}: macro-F1 under each test-time corruption, mean over runs, for $\lambda=0$ and $\lambda=0.25$.}\label{tab:robustvals}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l%s}
\toprule
 & %s\\
Corruption & %s\\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % ("c" * (2 * len(mods)), head1, head2, "\n".join(lines)))

# ---------------------------------------------------------------- placeholders for results that do not exist yet (marked PENDING)
for name, label, what in [("tab_eff", "tab:eff", "Measured cost of the evidence networks"), ("tab_lr", "tab:lr", "Learning-rate check"),
                          ("tab_external", "tab:external", "External test on the independent BRISC slices")]:
    if not os.path.exists(os.path.join(OUT, name + ".tex")):
        write(name, r"""\begin{table}
\centering
\caption{%s. PENDING: the runs that produce this table have not finished.}\label{%s}
\begin{tabular}{l}\toprule PENDING\\ \bottomrule\end{tabular}
\end{table}
""" % (what, label))
