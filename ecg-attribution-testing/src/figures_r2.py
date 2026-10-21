"""Figures added in the Research_Expert round for ecg-attribution-testing (CPU, from stored results only).
  fig_profile       multi-metric profile: rank of each method on stability, learning above the untrained level, faithfulness,
                    faithfulness above the amplitude reference, and cost (1 = best), cell text = measured value
  fig_rank          rank view: rank of each method by seed agreement for each architecture (bump chart)
  fig_distribution  distribution view: per-item seed agreement (mean Spearman over the 10 seed pairs) per method, XResNet1d-50
  fig_tradeoff      trade-off view: deletion gain against CPU time per item, with the amplitude reference
Usage: python figures_r2.py <ecg-attribution-testing dir>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
F = os.path.join(R, "figures", "final")
os.makedirs(F, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
ARCH = ["xresnet1d50", "inception1d", "convtran1d"]
AN = {"xresnet1d50": "XResNet1d-50", "inception1d": "InceptionTime-1d", "convtran1d": "Conv-Transformer-1d"}
ALL = ["saliency", "gxi", "ig", "smoothgrad", "gradshap", "occlusion", "cam", "gradcam_mid"]
MN = {"saliency": "Saliency", "gxi": "Gradient × input", "ig": "Integrated gradients", "smoothgrad": "SmoothGrad", "gradshap": "GradientSHAP",
      "occlusion": "Occlusion", "cam": "CAM", "gradcam_mid": "Grad-CAM (mid)", "amplitude": "Amplitude (no model)"}
COL = {"saliency": "#2a78d6", "gxi": "#eb6834", "ig": "#1baf7a", "smoothgrad": "#eda100", "gradshap": "#e87ba4", "occlusion": "#008300",
       "cam": "#6250d6", "gradcam_mid": "#a8509a"}
MARK = {"saliency": "o", "gxi": "s", "ig": "D", "smoothgrad": "P", "gradshap": "X", "occlusion": "^", "cam": "v", "gradcam_mid": "*"}
METH = [m for m in ALL if os.path.exists(os.path.join(R, "results", "attributions", "profiles_%s.npy" % m))]


def save(fig, name):
    fig.savefig(os.path.join(F, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(F, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


st = pd.read_csv(os.path.join(S, "rq1_stability.csv"))
st = st[st.label == "all"]
fa = pd.read_csv(os.path.join(S, "rq3_faithfulness.csv"))
ft = fa[fa.random_init == 0]
le = pd.read_csv(os.path.join(S, "rq2_learned_excess.csv"))
EF = os.path.join(S, "efficiency.json")
ef = json.load(open(EF)) if os.path.exists(EF) else None

# ---------------------------------------------------------------- multi-metric profile
prof = pd.DataFrame(index=METH)
prof["Seed rho"] = [st[(st.method == m) & (st.type == "seed")].spearman.mean() for m in METH]
prof["Seed - untrained rho"] = [st[(st.method == m) & (st.type == "seed")].spearman.mean() - st[(st.method == m) & (st.type == "random")].spearman.mean()
                                for m in METH]
prof["Top-10% overlap"] = [st[(st.method == m) & (st.type == "seed")].jaccard_top10.mean() for m in METH]
prof["Learned MI QRS"] = [le[(le.method == m) & (le.label == "MI") & (le.segment == "QRS")].learned_excess.mean() for m in METH]
prof["Deletion gain"] = [ft[ft.method == m].gain.mean() for m in METH]
prof["Gain - amplitude"] = prof["Deletion gain"] - ft[ft.method == "amplitude"].gain.mean()
if ef:
    prof["CPU ms per item"] = [ef["method"][m]["cpu_ms_per_item"] for m in METH]
low = {"CPU ms per item"}
ranks = prof.apply(lambda c: rankdata(c if c.name in low else -c, method="min"))
fig, ax = plt.subplots(figsize=(7.09, 3.0))
im = ax.imshow(ranks.values, cmap="Blues_r", vmin=1, vmax=len(METH) + 2, aspect="auto")
for i in range(ranks.shape[0]):
    for j in range(ranks.shape[1]):
        v = prof.values[i, j]
        ax.text(j, i, ("%.0f" % v) if prof.columns[j] in low else ("%+.2f" % v if "-" in prof.columns[j] or "Learned" in prof.columns[j] else "%.2f" % v),
                ha="center", va="center", fontsize=8, color="white" if ranks.values[i, j] <= 2 else INK)
ax.set_xticks(range(ranks.shape[1]), [c + (" (low)" if c in low else "") for c in ranks.columns], rotation=20, ha="right")
ax.set_yticks(range(len(METH)), [MN[m] for m in METH])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=range(1, len(METH) + 1))
cb.set_label("Rank (1 = best)", fontsize=8)
save(fig, "fig_profile")

# ---------------------------------------------------------------- rank view (bump chart)
fig, ax = plt.subplots(figsize=(3.46, 3.0))
rk = {}
for a in ARCH:
    v = st[(st.type == "seed") & (st.arch == a)].set_index("method").spearman.reindex(METH)
    rk[a] = pd.Series(rankdata(-v.values, method="min"), index=METH)
for m in METH:
    ys = [rk[a][m] for a in ARCH]
    ax.plot(range(3), ys, "-", marker=MARK[m], color=COL[m], ms=6, lw=1.5)
    ax.annotate(MN[m], (2, ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
ax.set_xticks(range(3), ["XResNet", "Inception", "ConvTran"])
ax.set_yticks(range(1, len(METH) + 1))
ax.invert_yaxis()
ax.set_ylabel("Rank by seed agreement\n(1 = most stable)")
ax.set_xlim(-0.2, 2.2)
save(fig, "fig_rank")

# ---------------------------------------------------------------- per-item distribution of seed agreement (XResNet1d-50)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
idx = {r["exp_id"]: i for i, r in enumerate(reg.to_dict("records"))}
pairs = []
tr = {a: [idx[e] for e in reg[(reg.model == a) & (reg.random_init == 0)].exp_id] for a in ARCH}
rnd = {a: idx[reg[(reg.model == a) & (reg.random_init == 1)].exp_id.iloc[0]] for a in ARCH}
for a in ARCH:  # same order as stats.py
    for i in range(5):
        for j in range(i + 1, 5):
            pairs.append(("seed", a))
        pairs.append(("random", a))
for x in range(3):
    for y in range(x + 1, 3):
        for _ in tr[ARCH[x]]:
            for _ in tr[ARCH[y]]:
                pairs.append(("arch", ARCH[x]))
cols = [k for k, (t, a) in enumerate(pairs) if t == "seed" and a == "xresnet1d50"]
fig, ax = plt.subplots(figsize=(7.09, 2.6))
data = []
for m in METH:
    P = np.load(os.path.join(S, "pair_spearman_%s.npy" % m))
    v = np.nanmean(P[:, cols], 1)
    data.append(v[np.isfinite(v)])
parts = ax.violinplot(data, showmedians=True, widths=0.8)
for k, b in enumerate(parts["bodies"]):
    b.set_facecolor(COL[METH[k]])
    b.set_alpha(0.55)
    b.set_hatch(["", "///", "...", "xx", "\\\\", "++", "--", "oo"][k])
for key in ("cbars", "cmins", "cmaxes", "cmedians"):
    parts[key].set_color(INK)
    parts[key].set_linewidth(0.8)
ax.set_xticks(range(1, len(METH) + 1), [MN[m] for m in METH], rotation=20, ha="right")
ax.set_ylabel("Seed agreement of one item (Spearman)")
ax.set_title("XResNet1d-50; each violin: %d test items; line: median" % len(data[0]), fontsize=8, color=MUTED, loc="left")
save(fig, "fig_distribution")

# ---------------------------------------------------------------- cost against faithfulness
if ef:
    fig, ax = plt.subplots(figsize=(3.46, 2.8))
    for m in METH:
        x = ef["method"][m]["cpu_ms_per_item"]
        y = ft[ft.method == m].gain.mean()
        ax.scatter(x, y, s=40, marker=MARK[m], color=COL[m], edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(MN[m], (x, y), xytext=(4, 2), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(ft[ft.method == "amplitude"].gain.mean(), color=MUTED, ls="--", lw=1)
    ax.margins(x=0.25, y=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("CPU time per item (ms, log)")
    ax.set_ylabel("Deletion gain (logit units)")
    save(fig, "fig_tradeoff")
print("FIGURES_R2_DONE")
