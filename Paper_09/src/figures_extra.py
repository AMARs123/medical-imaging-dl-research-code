"""Figure of the ablations of Paper 09 (from extra.py outputs). Usage: python figures_extra.py <Paper_09 dir>
  fig_ablation  left: AURC of foreground entropy against the widening d (whole image as a separate marker);
                right: AURC of ensemble agreement against the number of members K, with TTA agreement of one network as reference
Print width 7.09 in, type >= 8 pt; data sets distinguished by colour and marker."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
O = os.path.join(R, "results", "processed")
F = os.path.join(R, "figures", "final")
V.setup()
ST = {"kvasir": (V.CAT[0], "o", "Kvasir-SEG"), "ham": (V.CAT[1], "s", "HAM10000")}
d = pd.read_csv(os.path.join(O, "ablation_dilation.csv"))
k = pd.read_csv(os.path.join(O, "ensemble_size.csv"))
f = pd.read_csv(os.path.join(O, "failure.csv"))
f = f[(f.set != "val") & (f.member != "ens")]
fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.9))
xs = [0, 1, 2, 5, 10, 20]
for dd, (c, m, lab) in ST.items():
    h = d[d.data == dd]
    y = [100 * h[h.d == str(x)].aurc.mean() for x in xs]
    axs[0].plot(range(len(xs)), y, color=c, marker=m, lw=1.5, label="%s, foreground" % lab)
    axs[0].plot([len(xs) + 0.6], [100 * h[h.d == "image"].aurc.mean()], marker=m, color=c, mfc="white", mew=1.4, ls="none",
                label="%s, whole image" % lab)
    y2 = [100 * k[(k.data == dd) & (k.K == K)].aurc.mean() for K in range(2, 6)]
    axs[1].plot(range(2, 6), y2, color=c, marker=m, lw=1.5, label="%s, ensemble agreement" % lab)
    axs[1].axhline(100 * f[(f.data == dd) & (f.score == "unc_pdice_tta")].aurc.mean(), color=c, ls="--", lw=1.0,
                   label="%s, TTA agreement (1 network)" % lab)
axs[0].set_xticks(list(range(len(xs))) + [len(xs) + 0.6])
axs[0].set_xticklabels([str(x) for x in xs] + ["image"])
axs[0].set_xlabel("Foreground widening d (pixels)")
axs[0].set_ylabel("AURC x 100 (lower is better)")
axs[1].set_xticks(range(2, 6))
axs[1].set_xlabel("Ensemble members K")
fig.set_size_inches(7.09, 4.0)
for ax in axs:
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.36, wspace=0.28)
V.save(fig, os.path.join(F, "fig_ablation"))
print("FIGURES_EXTRA_DONE")
