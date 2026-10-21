"""Figures of waste-sorting-edge from stored results, drawn at print size (double column 7.09 in, single column 3.46 in), type >= 8 pt.
Every backbone has one fixed colour and marker; training sources are distinguished by marker fill and position.
  fig_gap        balanced accuracy of every backbone on both test sets, by training source (seed dots, mean bar)
  fig_pool       pooling effect per backbone and test set, paired by seed
  fig_confusion  TACO-test confusion for TrashNet-trained and pooled models (pooled over backbones and seeds)
  fig_pareto     one-thread CPU latency against TACO balanced accuracy, fp32 and int8 (when efficiency results exist)
Usage: python figures.py <waste-sorting-edge dir>"""
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
F = os.path.join(R, "figures", "final")
os.makedirs(F, exist_ok=True)
W2, W1 = 7.09, 3.46
MN = {"mobilenetv3_small_100.lamb_in1k": "MobileNetV3-S", "mobilenetv3_large_100.ra_in1k": "MobileNetV3-L", "efficientnet_b0.ra_in1k": "EfficientNet-B0",
      "resnet18.a1_in1k": "ResNet-18", "resnet50.a1_in1k": "ResNet-50", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
ORDER = list(MN)
COL = dict(zip(ORDER, V.CAT))
MARK = dict(zip(ORDER, ["o", "s", "D", "^", "v", "P"]))
SRC = [("trashnet", "TrashNet only"), ("taco", "TACO only"), ("both", "Both")]
HATCH = {"trashnet": "", "taco": "///", "both": "..."}
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
runs = pd.read_csv(os.path.join(P, "runs.csv"))

# ---------------------------------------------------------------- balanced accuracy by source and test set
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
for ax, (te, title) in zip(axs, [("TN", "TrashNet test (studio)"), ("TA", "TACO test (litter in the wild)")]):
    for k, m in enumerate(ORDER):
        for j, (tr, lab) in enumerate(SRC):
            v = runs[(runs.model == m) & (runs.train_on == tr) & (runs.test == te)].bacc.values
            if not len(v):
                continue
            x = k + (j - 1) * 0.27
            ax.bar(x, v.mean(), width=0.25, color=COL[m], alpha=[1.0, 0.45, 0.75][j], hatch=HATCH[tr], edgecolor="white", lw=0.4,
                   label=lab if k == 0 else None)
            ax.plot(np.full(len(v), x), v, ".", color=V.INK, ms=3)
    ax.set_xticks(range(len(ORDER)), [MN[m] for m in ORDER], rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_ylim(0, 1)
axs[0].set_ylabel("Balanced accuracy")
h, l_ = axs[0].get_legend_handles_labels()
fig.legend(h, l_, title="Trained on", ncol=3, fontsize=8, title_fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_gap"))

# ---------------------------------------------------------------- pooling effect
pool = pd.read_csv(os.path.join(S, "pool.csv"))
fig, ax = plt.subplots(figsize=(W1, 2.8))
for j, te in enumerate(["TN", "TA"]):
    for k, m in enumerate(ORDER):
        v = pool[(pool.model == m) & (pool.test == te)].pool_effect.values
        if len(v):
            x = j + (k - 2.5) * 0.11
            ax.plot(np.full(len(v), x), v, MARK[m], color=COL[m], ms=4.5, mfc=COL[m] if te == "TA" else "white", label=MN[m] if j == 1 else None)
ax.axhline(0, color=V.INK2, lw=0.8)
ax.set_xticks([0, 1], ["TrashNet test", "TACO test"])
ax.set_ylabel("Pooled minus single-source\nbalanced accuracy")
ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_pool"))

# ---------------------------------------------------------------- confusion on TACO
cf = pd.read_csv(os.path.join(P, "confusion.csv"))
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0))
for ax, (tr, title) in zip(axs, [("trashnet", "Trained on TrashNet only"), ("both", "Trained on both sources")]):
    q = cf[cf.train_on == tr].pivot(index="true", columns="pred", values="share").reindex(index=CLASSES, columns=CLASSES)
    im = ax.imshow(q.values, cmap=V.SEQ, vmin=0, vmax=1)
    for i in range(5):
        for k in range(5):
            ax.text(k, i, "%.2f" % q.values[i, k], ha="center", va="center", fontsize=8, color="white" if q.values[i, k] > 0.55 else V.INK)
    ax.set_xticks(range(5), CLASSES, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(5), CLASSES, fontsize=8)
    ax.set_xlabel("Predicted material")
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(False)
axs[0].set_ylabel("True material (TACO test)")
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="Share of row")
V.save(fig, os.path.join(F, "fig_confusion"))

# ---------------------------------------------------------------- cost against accuracy
ef = os.path.join(S, "efficiency.csv")
if os.path.exists(ef):
    e = pd.read_csv(ef).set_index("model")
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    for m in ORDER:
        if m not in e.index:
            continue
        r = e.loc[m]
        ax.plot(r.fp32_ms_1t, r.fp32_bacc_taco, MARK[m], color=COL[m], ms=6, label=MN[m])
        if r.int8_ok and pd.notna(r.get("int8_ms_1t", np.nan)):
            ax.plot(r.int8_ms_1t, r.int8_bacc_taco, MARK[m], color=COL[m], ms=6, mfc="white")
            ax.annotate("", (r.int8_ms_1t, r.int8_bacc_taco), (r.fp32_ms_1t, r.fp32_bacc_taco), arrowprops=dict(arrowstyle="->", color=COL[m], lw=0.8))
    fr = e[e.pareto_fp32].sort_values("fp32_ms_1t")
    ax.plot(fr.fp32_ms_1t, fr.fp32_bacc_taco, "--", color=V.INK2, lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency, 1 thread (ms, log)")
    ax.set_ylabel("Balanced accuracy, TACO test")
    ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    V.save(fig, os.path.join(F, "fig_pareto"))
print("FIGURES_DONE")
