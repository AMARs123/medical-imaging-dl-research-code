"""Figures of Paper 03 from stored outputs (results/attributions, results/statistics). Usage: python figures.py <Paper_03 dir>

Drawn at print size (double column 7.09 in, single column 3.46 in) with type of at least 8 pt. Every method has one fixed colour
and one marker shape, every architecture one colour and one marker, so the figures stay readable in greyscale."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from segments import segment_masks  # noqa: E402

R = sys.argv[1]
A = os.path.join(R, "results", "attributions")
S = os.path.join(R, "results", "statistics")
F = os.path.join(R, "figures", "final")
os.makedirs(F, exist_ok=True)
V.setup()
W2, W1 = 7.09, 3.46  # print widths in inches
ARCH = ["xresnet1d50", "inception1d", "convtran1d"]
AN = {"xresnet1d50": "XResNet1d-50", "inception1d": "InceptionTime-1d", "convtran1d": "Conv-Transformer-1d"}
AC = dict(zip(ARCH, V.CAT[:3]))
AM = {"xresnet1d50": "o", "inception1d": "s", "convtran1d": "^"}
ALL = ["saliency", "gxi", "ig", "smoothgrad", "gradshap", "occlusion", "cam", "gradcam_mid"]
METH = [m for m in ALL if os.path.exists(os.path.join(A, "profiles_%s.npy" % m))]
MN = {"saliency": "Saliency", "gxi": "Gradient × input", "ig": "Integrated gradients", "occlusion": "Occlusion", "cam": "CAM",
      "smoothgrad": "SmoothGrad", "gradshap": "GradientSHAP", "gradcam_mid": "Grad-CAM (mid)",
      "amplitude": "Amplitude (no model)"}
# same colours and markers as figures_r2.py
COL = {"saliency": "#2a78d6", "gxi": "#eb6834", "ig": "#1baf7a", "smoothgrad": "#eda100", "gradshap": "#e87ba4", "occlusion": "#008300",
       "cam": "#6250d6", "gradcam_mid": "#a8509a", "amplitude": V.INK}
MARK = {"saliency": "o", "gxi": "s", "ig": "D", "smoothgrad": "P", "gradshap": "X", "occlusion": "^", "cam": "v", "gradcam_mid": "*", "amplitude": "o"}
HATCH = {"saliency": "", "gxi": "///", "ig": "...", "smoothgrad": "xx", "gradshap": "\\\\\\", "occlusion": "--", "cam": "++", "gradcam_mid": "oo"}
SEG = ["P", "QRS", "STT", "other"]
SEGC = {"P": "#cde2fb", "QRS": "#f4a582", "STT": "#b7e4c7", "other": "#ffffff"}
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
items = pd.read_csv(os.path.join(A, "items.csv"))
X = np.load(os.path.join(R, "data", "processed", "signals.npy"), mmap_mode="r")

# ---------------------------------------------------------------- example item: one MI and one STTC recording
for lab in ["MI", "STTC"]:
    cand = items[(items.label_name == lab)]
    probs = cand[["prob_" + e for e in reg[reg.random_init == 0].exp_id]].mean(1)
    it = cand.index[np.argsort(-probs.values)[len(cand) // 4]]  # a typical detected case (75th percentile of confidence)
    eid, ai = int(items.ecg_id[it]), int(items.array_index[it])
    mk = segment_masks(eid)
    t = np.arange(1000) / 100.0
    rows = METH + ["amplitude"]
    fig, axs = plt.subplots(len(rows) + 1, 1, figsize=(W2, 1.2 + 0.62 * len(rows)), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.6] + [1] * len(rows)))
    for ax in axs:
        for si, sg in enumerate(SEG[:3]):
            on = np.where(np.diff(np.r_[0, mk[si].astype(int), 0]) != 0)[0].reshape(-1, 2)
            for a, b in on:
                ax.axvspan(a / 100, b / 100, color=SEGC[sg], lw=0, zorder=0)
        ax.grid(False)
    axs[0].plot(t, X[ai][1], color=V.INK, lw=0.8)
    axs[0].set_ylabel("Lead II\n(mV)")
    axs[0].set_title("ECG %d (%s). Shading: P wave (light blue), QRS (orange), ST-T (green)" % (eid, lab), loc="left", fontsize=8.5)
    j_tr = int(reg[(reg.model == "xresnet1d50") & (reg.seed == 0)].index[0])
    j_rn = int(reg[(reg.model == "xresnet1d50") & (reg.random_init == 1)].index[0])
    for k, mt in enumerate(rows):
        P = np.load(os.path.join(A, "profiles_%s.npy" % mt), mmap_mode="r")
        ax = axs[k + 1]
        for j, col, ls, lab2 in [(j_tr, V.CAT[0], "-", "trained (seed 0)"), (j_rn, V.INK2, ":", "untrained")]:
            r = np.asarray(P[it, j], np.float32)
            r = r / (r.max() + 1e-12)
            ax.plot(t, r, color=col, lw=1.0, ls=ls, label=lab2)
        ax.set_ylabel(MN[mt].replace(" (", "\n("), fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axs[1].legend(loc="upper right", fontsize=8, ncol=2)
    axs[-1].set_xlabel("Time (s)")
    V.save(fig, os.path.join(F, "fig_example_%s" % lab))

# ---------------------------------------------------------------- stability: one panel per method, 2 rows
st = pd.read_csv(os.path.join(S, "rq1_stability.csv"))
st = st[st.label == "all"]
nc = 4
nr = int(np.ceil(len(METH) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(W2, 2.0 * nr + 0.3), sharey=True, squeeze=False)
for ax in axs.flat[len(METH):]:
    ax.set_visible(False)
for ax, mt in zip(axs.flat, METH):
    q = st[st.method == mt]
    for k, a in enumerate(ARCH):
        for typ, off in [("seed", -0.15), ("random", 0.15)]:
            r = q[(q.type == typ) & (q.arch == a)]
            if len(r):
                ax.errorbar(k + off, r.spearman.iloc[0], yerr=[[r.spearman.iloc[0] - r.ci_low.iloc[0]], [r.ci_high.iloc[0] - r.spearman.iloc[0]]],
                            fmt=AM[a], color=AC[a], ms=5, mfc=AC[a] if typ == "seed" else "white", mew=1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["XR", "IT", "CT"])
    ax.set_title(MN[mt], fontsize=8.5)
    ax.axhline(0, color=V.GRID)
for r_ in range(nr):
    axs[r_, 0].set_ylabel("Spearman correlation\nof profiles")
# explanation of markers (filled: seed pairs; open: trained vs untrained) is in the LaTeX caption
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_stability"))

# ---------------------------------------------------------------- localisation: lift per segment, trained vs amplitude
lo = pd.read_csv(os.path.join(S, "rq2_localisation.csv"))
labs = ["NORM", "MI", "STTC", "CD", "HYP"]
fig, axs = plt.subplots(1, 4, figsize=(W2, 3.4), sharey=True)
rowsm = METH + ["amplitude"]
for ax, sg in zip(axs, SEG):
    for k, mt in enumerate(rowsm):
        q = lo[(lo.method == mt) & (lo.random_init == 0)].groupby("label")["lift_" + sg].mean().reindex(labs)
        off = (k - (len(rowsm) - 1) / 2) * 0.07
        ax.plot(q.values, np.arange(len(labs)) + off, MARK[mt], color=COL[mt], ms=4.5,
                mfc="white" if mt == "amplitude" else COL[mt], mew=1.1, label=MN[mt], ls="none")
    ax.axvline(1, color=V.INK2, lw=0.8, ls="--")
    ax.set_title({"STT": "ST-T", "other": "Other"}.get(sg, sg), fontsize=9)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Lift (share / time share)")
axs[0].invert_yaxis()
axs[0].set_ylabel("Diagnostic superclass")
h, l_ = axs[0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.18, 1, 1))
V.save(fig, os.path.join(F, "fig_localisation"))

# ---------------------------------------------------------------- learned excess for the expected segment
le = pd.read_csv(os.path.join(S, "rq2_learned_excess.csv"))
le = le[le.method != "amplitude"]
pairs_ = [("MI", "QRS"), ("MI", "STT"), ("STTC", "STT"), ("CD", "QRS"), ("HYP", "QRS"), ("NORM", "P")]
fig, ax = plt.subplots(figsize=(W2, 2.9))
w = 0.8 / len(METH)
for k, mt in enumerate(METH):
    v = [le[(le.method == mt) & (le.label == a) & (le.segment == b)].learned_excess.mean() for a, b in pairs_]
    ax.bar(np.arange(len(pairs_)) + (k - (len(METH) - 1) / 2) * w, v, width=w, color=COL[mt], edgecolor="white", lw=0.4,
           hatch=HATCH[mt], label=MN[mt])
ax.axhline(0, color=V.INK2, lw=0.8)
ax.set_xticks(range(len(pairs_)))
ax.set_xticklabels(["%s: %s" % (a, {"STT": "ST-T"}.get(b, b)) for a, b in pairs_])
ax.set_ylabel("Share gained by training\n(trained - untrained)")
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_learned_excess"))

# ---------------------------------------------------------------- faithfulness
fa = pd.read_csv(os.path.join(S, "rq3_faithfulness.csv"))
fa = fa[fa.random_init == 0]
order = METH + ["amplitude"]
fig, ax = plt.subplots(figsize=(W2, 2.9))
for k, a in enumerate(ARCH):
    q = fa[fa.arch == a].set_index("method").reindex(order)
    x = np.arange(len(q)) + (k - 1) * 0.25
    ax.errorbar(x, q.gain, yerr=[q.gain - q.ci_low, q.ci_high - q.gain], fmt=AM[a], color=AC[a], ms=4.5, label=AN[a])
amp = fa[fa.method == "amplitude"].gain
ax.axhspan(amp.min(), amp.max(), color=V.GRID, zorder=0, label="Amplitude reference (range)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([MN[m] for m in order], fontsize=8, rotation=25, ha="right")
ax.set_ylabel("Deletion gain (logit drop:\ntop 10% - random 10%)")
ax.axhline(0, color=V.INK2, lw=0.8)
ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_faithfulness"))

# ---------------------------------------------------------------- stability vs detection
r4 = pd.read_csv(os.path.join(S, "rq4_stability_vs_detection.csv"))
fig, ax = plt.subplots(figsize=(W2, 2.7))
for k, a in enumerate(ARCH):
    q = r4[r4.arch == a].set_index("method").reindex(METH)
    ax.plot(np.arange(len(METH)) + (k - 1) * 0.2, q.stab_detected - q.stab_missed, AM[a], color=AC[a], ms=5, label=AN[a], ls="none")
ax.axhline(0, color=V.INK2, lw=0.8)
ax.set_xticks(range(len(METH)))
ax.set_xticklabels([MN[m] for m in METH], fontsize=8, rotation=25, ha="right")
ax.set_ylabel("Seed stability: detected\nminus missed labels")
ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_stability_detection"))
print("FIGURES_DONE")
