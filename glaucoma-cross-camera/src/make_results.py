"""Result tables and figures of glaucoma-cross-camera from results/statistics and results/processed (analyze.py).
Tables: tab_main (AUC per configuration, target and test set), tab_gap (camera gap on the same images), tab_hardsoft
(paired hard vs soft), tab_minority (error rate by expert minority size). Figures: fig_gap, fig_minority, fig_calib.
Usage: python make_results.py <glaucoma-cross-camera dir>"""
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
CN = {"all": "All cameras", "loco_Remidio": "Leave out Remidio", "loco_Bosch": "Leave out Bosch", "loco_Forus": "Leave out Forus"}
SETN = {"chaksu_test": "Ch\\'ak\\d{s}u test", "held_out_camera": "Held-out camera", "papila": "PAPILA", "rimone": "RIM-ONE DL"}
runs = pd.read_csv(os.path.join(P, "runs.csv"))


def w(name, body):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(body)


def fp(p):
    return "$<$0.001" if p < 0.001 else "%.3f" % p


def ms(v):
    return "%.3f (%.3f)" % (v.mean(), v.std()) if len(v) > 1 else ("%.3f" % v.mean() if len(v) else "--")


rows = []
for c in CN:
    for tg in ["hard", "soft"]:
        q = runs[(runs.config == c) & (runs.target == tg)]
        if not len(q):
            continue
        own = q[q.set == ("chaksu_test" if c == "all" else "held_out_camera")].auc.dropna()
        rows.append("%s & %s & %s & %s & %s \\\\" % (CN[c] if tg == "hard" else "", tg, ms(own), ms(q[q.set == "papila"].auc.dropna()), ms(q[q.set == "rimone"].auc.dropna())))
w("tab_main", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{AUC (Eq.~\ref{eq:auc}) of every configuration and target: on the Ch\'ak\d{s}u test split (all-camera models) or on all images of the held-out camera (leave-one-camera-out models), and on the two external data sets; mean (standard deviation) over two backbones and three seeds. $\uparrow$: higher is better.}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccc}
\toprule
Configuration & Target & Ch\'ak\d{s}u $\uparrow$ & PAPILA $\uparrow$ & RIM-ONE DL $\uparrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

g = pd.read_csv(os.path.join(S, "gap.csv"))
cig = pd.read_csv(os.path.join(S, "ci_gap.csv")).set_index(["camera", "target"]) if os.path.exists(os.path.join(S, "ci_gap.csv")) else None
rows = []
for cam in ["Remidio", "Bosch", "Forus"]:
    for tg in ["hard", "soft"]:
        q = g[(g.camera == cam) & (g.target == tg)]
        if not len(q):
            continue
        civ = "[$%+.3f$, $%+.3f$]" % tuple(cig.loc[(cam, tg), ["lo", "hi"]]) if cig is not None else "--"
        rows.append("%s & %s & %d (%d) & %s & %s & $%+.3f$ (%.3f) & %s \\\\" % (cam if tg == "hard" else "", tg, q.n.iloc[0], q.n_pos.iloc[0], ms(q.auc_all), ms(q.auc_loco),
                                                                            q.gap.mean(), q.gap.std(), civ))
w("tab_gap", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Camera gap on the same images (Eq.~\ref{eq:gap}): AUC on the official test-split images of each camera for the model that saw the camera (all cameras) and for the model that left it out, and their difference $\Delta_c$; mean (standard deviation) over two backbones and three seeds, and the 95\% bootstrap interval of the mean gap (Eq.~\ref{eq:boot}, images resampled). Images: test-split images of the camera (glaucoma). The Bosch and Forus test splits contain only one and two glaucoma images, so their gaps rest on one or two positive images and are not interpretable.}
\label{tab:gap}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccc}
\toprule
Camera & Target & Images & AUC all cameras $\uparrow$ & AUC camera left out $\uparrow$ & $\Delta_c$ & 95\% interval \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

hs = pd.read_csv(os.path.join(S, "hard_soft.csv"))
rows = []
for s in SETN:
    for mt, lab in [("auc", "AUC"), ("ece", "ECE")]:
        q = hs[(hs.set == s) & (hs.metric == mt)]
        if len(q):
            r = q.iloc[0]
            rows.append("%s & %s & %d & $%+.3f$ & %d & $%+.2f$ & %s \\\\" % (SETN[s] if mt == "auc" else "", lab, r.n_pairs, r.mean_diff, r.soft_better, r.rb, fp(r.p_holm)))
w("tab_hardsoft", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Soft against hard targets on paired runs (same configuration, backbone and seed; Eq.~\ref{eq:diff}): mean difference (soft minus hard), number of pairs in which the soft target is better (higher AUC, lower ECE), rank-biserial correlation (Eq.~\ref{eq:rb}) and Holm-adjusted Wilcoxon $p$ over the test sets.}
\label{tab:hardsoft}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}
\toprule
Test set & Measure & Pairs & Mean difference & Soft better & $r_{\mathrm{rb}}$ & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

mn = pd.read_csv(os.path.join(S, "minority.csv"))
rows = []
for s in ["chaksu_test", "held_out_camera"]:
    for tg in ["hard", "soft"]:
        q = mn[(mn.set == s) & (mn.target == tg)]
        if len(q):
            r = q.iloc[0]
            rows.append("%s & %s & %.3f & %.3f & %.3f & %.2f \\\\" % (SETN[s] if tg == "hard" else "", tg, r.err_m0, r.err_m1, r.err_m2, r.rho_H_m))
cnt = runs[runs.set == "chaksu_test"][["n_m0", "n_m1", "n_m2"]].iloc[0] if "n_m0" in runs else None
w("tab_minority", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Error rate at the validation threshold by expert minority size $m$ (Eq.~\ref{eq:minority}; $m=0$: all five experts agree, $m=2$: a 3--2 split) and Spearman correlation between the prediction entropy (Eq.~\ref{eq:entropy}) and $m$, mean over runs.""" + (" On the Ch\\'ak\\d{s}u test split, %d, %d and %d images have $m=0$, 1 and 2." % tuple(int(v) for v in cnt) if cnt is not None else "") + r""" $\downarrow$: lower is better.}
\label{tab:minority}
\begin{tabular}{llcccc}
\toprule
Test set & Target & Error, $m=0$ $\downarrow$ & Error, $m=1$ $\downarrow$ & Error, $m=2$ $\downarrow$ & $\rho(H,m)$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(3.46, 2.8))
for k, (tg, mk) in enumerate([("hard", "o"), ("soft", "s")]):
    for j, cam in enumerate(["Remidio", "Bosch", "Forus"]):
        q = g[(g.camera == cam) & (g.target == tg)]
        if len(q):
            ax.plot(np.full(len(q), j + (k - 0.5) * 0.25), q.gap, mk, color=V.CAT[k], ms=4.5, mfc="white" if tg == "soft" else V.CAT[k], label=tg if j == 0 else None)
ax.axhline(0, color=V.INK2, lw=0.8)
ax.set_xticks(range(3), ["Remidio", "Bosch", "Forus"])
ax.set_ylabel("Camera gap $\\Delta_c$ (AUC)")
ax.legend(title="Target", fontsize=8, title_fontsize=8)
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_gap"))

fig, ax = plt.subplots(figsize=(3.46, 2.8))
for k, (s, ls) in enumerate([("chaksu_test", "-"), ("held_out_camera", "--")]):
    for j, tg in enumerate(["hard", "soft"]):
        q = mn[(mn.set == s) & (mn.target == tg)]
        if len(q):
            ax.plot([0, 1, 2], q[["err_m0", "err_m1", "err_m2"]].values[0], ls, marker="os"[j], color=V.CAT[j], ms=5,
                    mfc="white" if s == "held_out_camera" else V.CAT[j], label="%s, %s" % (tg, "test split" if s == "chaksu_test" else "held-out camera"))
ax.set_xticks([0, 1, 2], ["5-0", "4-1", "3-2"])
ax.set_xlabel("Expert vote split")
ax.set_ylabel("Error rate at threshold")
ax.legend(fontsize=8)
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_minority"))
# ---------------------------------------------------------------- added analyses (analyze_extra.py)
SETN2 = dict(SETN, chaksu_test="Ch\\'ak\\d{s}u test")
CN2 = dict(CN, all_n271="All cameras, %d images" % 271)
if os.path.exists(os.path.join(S, "ci_auc.csv")):
    ci = pd.read_csv(os.path.join(S, "ci_auc.csv"))
    rows = []
    for c in CN2:
        for tg in ["hard", "soft"]:
            q = ci[(ci.config == c) & (ci.target == tg)]
            if not len(q):
                continue
            cells = []
            for s in ["own", "papila", "rimone"]:
                r = q[q.set.isin(["chaksu_test", "held_out_camera"])] if s == "own" else q[q.set == s]
                r = r.iloc[0]
                cells.append("%.3f [%.3f, %.3f]" % (r.auc_mean, r.lo, r.hi))
            rows.append("%s & %s & %s \\\\" % (CN2[c] if tg == "hard" else "", tg, " & ".join(cells)))
    w("tab_ci", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Mean AUC of the six runs of each configuration and target with its 95\% bootstrap interval (Eq.~\ref{eq:boot}; $B=2000$; images resampled, patients for PAPILA; one resample shared by the six runs). Own test set: Ch\'ak\d{s}u test split for all-camera models, all images of the held-out camera otherwise.}
\label{tab:ci}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc}
\toprule
Configuration & Target & Own test set & PAPILA & RIM-ONE DL \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

if os.path.exists(os.path.join(S, "prior_corr_pairs.csv")):
    pp = pd.read_csv(os.path.join(S, "prior_corr_pairs.csv"))
    rows = []
    for s in SETN2:
        for k, lab in [("raw", "as trained"), ("corr", "weight removed")]:
            q = pp[(pp.set == s) & (pp.version == k)]
            if len(q):
                r = q.iloc[0]
                rows.append("%s & %s & %.3f & %.3f & $%+.3f$ & %d/%d & $%+.2f$ & %s \\\\" % (SETN2[s] if k == "raw" else "", lab, r.ece_hard, r.ece_soft, r.mean_diff,
                                                                                    r.soft_better, r.n_pairs, r.rb, fp(r.p_holm)))
    w("tab_calib", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{ECE (Eq.~\ref{eq:calib}) of hard- and soft-target models as trained and after removing the class weight of the loss, $p'=\sigma(z-\log w_+)$ (Eq.~\ref{eq:prior}; added after the main results). Mean over paired runs; soft better: pairs in which the soft target has the lower ECE; $r_{\mathrm{rb}}$ and Holm-adjusted Wilcoxon $p$ over the four test sets within each version. $\downarrow$: lower is better.}
\label{tab:calib}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccc}
\toprule
Test set & Probabilities & ECE hard $\downarrow$ & ECE soft $\downarrow$ & Difference & Soft better & $r_{\mathrm{rb}}$ & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

if os.path.exists(os.path.join(S, "control.csv")) and os.path.getsize(os.path.join(S, "control.csv")) > 5:
    ct = pd.read_csv(os.path.join(S, "control.csv"))
    rows = []
    for r in ct.itertuples():
        rows.append("%s & %d & %.3f & %.3f & %.3f & $%+.3f$ [$%+.3f$, $%+.3f$] & $%+.3f$ [$%+.3f$, $%+.3f$] \\\\" % (
            r.target, r.n_pairs, r.auc_all, r.auc_n271, r.auc_loco, r.size_effect, r.size_lo, r.size_hi, r.camera_effect, r.cam_lo, r.cam_hi))
    w("tab_control", r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Size-matched control on the %d official test-split Remidio images (%d glaucoma; added after the main results). AUC of the all-camera model (all: 1,009 training images), of the all-camera model trained on a random label-stratified subset of 271 images (271), and of the model that left Remidio out (no Remidio: 271 images of Bosch and Forus); mean over paired runs (backbone and seed). Size effect: all $-$ 271 images; camera effect: 271 images $-$ Remidio left out; 95\%% bootstrap intervals of the mean (images resampled, $B=2000$).}
\label{tab:control}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Target & Pairs & AUC all $\uparrow$ & AUC 271 $\uparrow$ & AUC no Remidio $\uparrow$ & Size effect & Camera effect \\
\midrule
""" % (ct.n.iloc[0], ct.n_pos.iloc[0]) + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

if os.path.exists(os.path.join(S, "prob_by_votes.csv")):
    pv = pd.read_csv(os.path.join(S, "prob_by_votes.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (k, ttl) in zip(axs, [("p_raw", "As trained"), ("p_corr", "Class weight removed")]):
        for j, (tg, mk) in enumerate([("hard", "o"), ("soft", "s")]):
            for s, ls in [("chaksu_test", "-"), ("held_out_camera", "--")]:
                q = pv[(pv.set == s) & (pv.target == tg)].sort_values("n_suspect")
                ax.plot(q.n_suspect, q[k], ls, marker=mk, color=V.CAT[j], ms=5, mfc="white" if s == "held_out_camera" else V.CAT[j],
                        label="%s, %s" % (tg, "test split" if s == "chaksu_test" else "held-out camera"))
        a = np.arange(6) / 5.0
        if k == "p_raw":
            wv = pv.w_pos.mean()
            ax.plot(np.arange(6), wv * a / (wv * a + 1 - a), ":", color=V.INK2, lw=1.2, label="$p^\\ast$ (Eq. 5)")
        else:
            ax.plot(np.arange(6), a, ":", color=V.INK2, lw=1.2, label="$a/E$")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks(range(6))
        ax.set_xlabel("Experts calling the image suspect, $a(x)$")
    axs[0].set_ylabel("Mean predicted probability")
    axs[0].set_ylim(0, 1.02)
    axs[1].legend(fontsize=8, loc="upper left")
    axs[0].legend(fontsize=8, loc="upper left", handles=[axs[0].lines[-1]])
    fig.tight_layout()
    V.save(fig, os.path.join(F, "fig_calib"))
print("RESULTS_DONE")
