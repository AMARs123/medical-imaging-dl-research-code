"""Figures of Paper 05 at print size (7.09 in / 3.46 in, type >= 8 pt); one fixed colour and marker per formulation.
  fig_ser        severe-error rate and QWK of single models per formulation on DDR test and DeepDRiD (runs as dots)
  fig_abstain    severe-error rate among accepted images against coverage, ensemble ordered by the ordinal variance U
  fig_shift      QWK on DDR test against DeepDRiD for every single run (does the order of formulations survive the shift?)
Usage: python figures.py <Paper_05 dir>"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

V.setup()
R = sys.argv[1]
P = os.path.join(R, "results", "processed")
F = os.path.join(R, "figures", "final")
os.makedirs(F, exist_ok=True)
FORMS = ["ce", "sord", "reg", "coral", "corn"]
FN = {"ce": "CE", "sord": "SORD", "reg": "REG", "coral": "CORAL", "corn": "CORN"}
COL = dict(zip(FORMS, V.CAT))
MARK = dict(zip(FORMS, ["o", "s", "D", "^", "v"]))
MOD = {"resnet50.a1_in1k": "ResNet-50", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
ev = pd.read_csv(os.path.join(P, "eval.csv"))
sg = ev[ev.source == "single"]

fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.8))
for ax, (metric, lab) in zip(axs, [("severe_error_rate", "Severe-error rate (lower is better)"), ("qwk", "QWK (higher is better)")]):
    for j, (s, sl) in enumerate([("test", "DDR test"), ("ext", "DeepDRiD")]):
        for k, f in enumerate(FORMS):
            v = sg[(sg.form == f) & (sg.set == s)][metric].values
            if not len(v):
                continue
            x = j * (len(FORMS) + 1) + k
            ax.bar(x, v.mean(), color=COL[f], width=0.8, alpha=0.85, label=FN[f] if j == 0 else None)
            ax.plot(np.full(len(v), x), v, ".", color=V.INK, ms=3)
    ax.set_xticks([(len(FORMS) - 1) / 2, len(FORMS) + 1 + (len(FORMS) - 1) / 2], ["DDR test", "DeepDRiD"])
    ax.set_ylabel(lab)
axs[0].legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(1.1, 1.02))
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_ser"))

fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.9), sharey=True)
cov = np.linspace(0.05, 1, 96)
for ax, (s, sl) in zip(axs, [("test", "DDR test"), ("ext", "DeepDRiD")]):
    for f in FORMS:
        curves = []
        for m in MOD:
            fn = os.path.join(P, "perimage_%s_%s_%s.npz" % (m.split(".")[0], f, s))
            if not os.path.exists(fn):
                continue
            z = np.load(fn)
            if "ens_u" not in z.files or not np.isfinite(z["ens_u"]).all():
                continue
            o = np.argsort(z["ens_u"], kind="stable")
            e = z["ens_sev"][o]
            curves.append([e[:max(1, int(round(c * len(e))))].mean() for c in cov])
        if curves:
            ax.plot(cov, np.mean(curves, 0), "-", color=COL[f], lw=1.6, label=FN[f], marker=MARK[f], markevery=12, ms=4)
    ax.set_title(sl, fontsize=9, loc="left")
    ax.set_xlabel("Coverage (share of images graded automatically)")
    ax.invert_xaxis()
axs[0].set_ylabel("Severe-error rate among accepted")
axs[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_abstain"))

fig, ax = plt.subplots(figsize=(3.46, 3.0))
pv = sg.pivot_table(index=["model", "form", "seed"], columns="set", values="qwk").reset_index()
for f in FORMS:
    q = pv[pv.form == f]
    ax.plot(q.test, q.ext, MARK[f], color=COL[f], ms=5, ls="none", label=FN[f])
lo, hi = np.nanmin(pv[["test", "ext"]].values) - 0.02, np.nanmax(pv[["test", "ext"]].values) + 0.02
ax.plot([lo, hi], [lo, hi], "--", color=V.INK2, lw=0.8)
ax.set_xlabel("QWK, DDR test")
ax.set_ylabel("QWK, DeepDRiD")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_shift"))
print("FIGURES_DONE")
