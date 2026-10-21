"""Figures of Paper 09 from results/processed and results/statistics. Usage: python figures.py <Paper_09 dir>

Print size: double column 7.09 in, single column 3.46 in; type of at least 8 pt. Each score has one fixed colour and one
marker or line style, so the figures stay readable in greyscale.
  fig_rc          risk-coverage curves of six scores, Kvasir and HAM10000 test sets (clean and noise), ConvNeXt-T seed 0 and ensemble
  fig_decomp      aggregation effect vs sampling effect (AURC differences) per data set and evaluation set
  fig_tradeoff    mean AURC against measured CPU time per image
  fig_rank        rank of the ten single-model scores in each of the 18 settings (heat map with the rank printed)
  fig_area        why whole-image entropy fails: score against predicted area, whole image vs predicted foreground
  fig_examples    two Kvasir test images: image, reference, prediction, entropy map, with the scores
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_resources"))
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
O = os.path.join(R, "results", "processed")
S = os.path.join(R, "results", "statistics")
F = os.path.join(R, "figures", "final")
os.makedirs(F, exist_ok=True)
V.setup()
W2, W1 = 7.09, 3.46
NAME = {"ent_det": "Entropy, image (1 pass)", "ent_det_fg": "Entropy, foreground (1 pass)", "ent_det_fg_e": "Fg entropy + empty flag (1 pass)", "ens_ent_fg_e": "Ensemble fg entropy + empty flag", "ent_mc": "MC entropy", "mi_mc": "MC mutual inf.",
        "unc_pdice_mc": "MC agreement", "ent_tta": "TTA entropy", "unc_pdice_tta": "TTA agreement", "ref_area": "Area (reference)",
        "ref_parts": "Parts (reference)", "ref_irreg": "Irregularity (reference)", "ens_ent": "Ensemble entropy",
        "ens_ent_fg": "Ensemble entropy, fg", "unc_pdice_ens": "Ensemble agreement", "oracle": "Oracle (sorted by Dice)"}
COL = {"ent_det": "#2a78d6", "ent_det_fg": "#eb6834", "ent_det_fg_e": "#a8509a", "ens_ent_fg_e": "#7a3b72", "ent_mc": "#86b6ef", "mi_mc": "#6250d6", "unc_pdice_mc": "#1baf7a", "ent_tta": "#e87ba4",
       "unc_pdice_tta": "#eda100", "ref_area": "#8a8a8a", "ref_parts": "#b0b0b0", "ref_irreg": "#555555", "unc_pdice_ens": "#008300",
       "ens_ent": "#104281", "ens_ent_fg": "#c2410c", "oracle": "#222222"}
LS = {"ent_det": "-", "ent_det_fg": "--", "unc_pdice_mc": "-.", "unc_pdice_tta": ":", "unc_pdice_ens": (0, (5, 1, 1, 1)), "ref_irreg": (0, (1, 2)),
      "oracle": "-"}
MK = {"ent_det": "o", "ent_det_fg": "s", "ent_det_fg_e": "p", "ens_ent_fg_e": "8", "ent_mc": "o", "mi_mc": "D", "unc_pdice_mc": "^", "ent_tta": "v", "unc_pdice_tta": "P",
      "ref_area": "x", "ref_parts": "+", "ref_irreg": "*", "unc_pdice_ens": "X", "ens_ent": "h", "ens_ent_fg": "H"}
SET = {"test": "Clean", "test_blur": "Blur", "test_noise": "Noise", "test_contrast": "Low contrast", "shift": "Other camera"}
SET2 = {"test": "Clean", "test_blur": "Blur", "test_noise": "Noise", "test_contrast": "Low\ncontrast", "shift": "Other\ncamera"}
DATA = {"kvasir": "Kvasir-SEG", "ham": "HAM10000"}
SINGLE = ["ent_det", "ent_det_fg", "ent_det_fg_e", "ent_mc", "mi_mc", "unc_pdice_mc", "ent_tta", "unc_pdice_tta", "ref_area", "ref_parts", "ref_irreg"]
img = pd.read_csv(os.path.join(O, "image_scores.csv"), low_memory=False)
fa = pd.read_csv(os.path.join(O, "failure.csv"))
JUDGE = {"ent_det": "dice_det", "ent_det_fg": "dice_det", "ent_det_fg_e": "dice_det", "ref_irreg": "dice_det", "unc_pdice_mc": "dice_mc", "unc_pdice_tta": "dice_tta"}


def rc_curve(dsc, score):
    o = np.argsort(score, kind="stable")
    r = 1 - dsc[o]
    cov = np.arange(1, len(r) + 1) / len(r)
    return cov, np.cumsum(r) / np.arange(1, len(r) + 1)


# ---------------------------------------------------------------- fig_rc
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0), sharex=True)
for ax, (data, st) in zip(axs.flat, [("kvasir", "test"), ("kvasir", "test_noise"), ("ham", "test"), ("ham", "test_noise")]):
    d = img[(img.data == data) & (img.encoder == "convnext_tiny.fb_in1k") & (img.set == st)]
    s0 = d[d.member == "0"]
    en = d[d.member == "ens"]
    for k in ["ent_det", "ent_det_fg", "unc_pdice_mc", "unc_pdice_tta", "ref_irreg"]:
        cov, risk = rc_curve(s0[JUDGE[k]].values, s0[k].values)
        ax.plot(cov, risk, color=COL[k], ls=LS[k], lw=1.4, label=NAME[k])
    cov, risk = rc_curve(en.dice_ens.values, en.unc_pdice_ens.values)
    ax.plot(cov, risk, color=COL["unc_pdice_ens"], ls=LS["unc_pdice_ens"], lw=1.4, label=NAME["unc_pdice_ens"])
    cov, risk = rc_curve(s0.dice_det.values, -s0.dice_det.values)
    ax.plot(cov, risk, color=COL["oracle"], lw=0.8, label=NAME["oracle"])
    ax.set_title("%s, %s images (n = %d)" % (DATA[data], SET[st].lower(), len(s0)), fontsize=9)
    ax.set_ylim(bottom=0)
for ax in axs[1]:
    ax.set_xlabel("Coverage (share of images accepted)")
for ax in axs[:, 0]:
    ax.set_ylabel("Risk (mean 1 - Dice\nof accepted images)")
h, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
V.save(fig, os.path.join(F, "fig_rc"))

# ---------------------------------------------------------------- fig_decomp
dec = pd.read_csv(os.path.join(S, "decomposition.csv"))
order = [("kvasir", s) for s in ["test", "test_blur", "test_noise", "test_contrast"]] + [("ham", s) for s in ["test", "test_blur", "test_noise", "test_contrast", "shift"]]
g = dec.groupby(["data", "set"])[["aggregation", "sampling_mc", "sampling_tta"]].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(W2, 2.9))
x = np.arange(len(order))
w = 0.26
for k, (c, lab, col, hatch) in enumerate([("aggregation", "Aggregation: image -> foreground (1 pass)", COL["ent_det_fg"], ""),
                                          ("sampling_mc", "Sampling: foreground entropy -> MC agreement", COL["unc_pdice_mc"], "///"),
                                          ("sampling_tta", "Sampling: foreground entropy -> TTA agreement", COL["unc_pdice_tta"], "...")]):
    m = [g.loc[o, (c, "mean")] for o in order]
    sd = [g.loc[o, (c, "std")] for o in order]
    ax.bar(x + (k - 1) * w, m, w, yerr=sd, color=col, hatch=hatch, edgecolor="white", lw=0.4, label=lab, error_kw=dict(lw=0.7, ecolor=V.INK2))
ax.axhline(0, color=V.INK2, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["%s\n%s" % ("Kvasir" if d == "kvasir" else "HAM", SET2[s]) for d, s in order], fontsize=8)
ax.set_ylabel("AURC reduction\n(higher = larger gain)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_decomp"))

# ---------------------------------------------------------------- fig_tradeoff
ef = json.load(open(os.path.join(S, "efficiency.json")))
cost = {}
for enc, e in ef["encoders"].items():
    for k, key in [("ent_det", "single_ms"), ("ent_det_fg", "single_ms"), ("ent_det_fg_e", "single_ms"), ("ref_irreg", "single_ms"), ("ent_mc", "mc10_ms"), ("mi_mc", "mc10_ms"),
                   ("unc_pdice_mc", "mc10_ms"), ("ent_tta", "tta8_ms"), ("unc_pdice_tta", "tta8_ms"), ("ens_ent", "ens5_ms"), ("ens_ent_fg", "ens5_ms"), ("ens_ent_fg_e", "ens5_ms"),
                   ("unc_pdice_ens", "ens5_ms")]:
        cost.setdefault(k, []).append(e[key])
ft = fa[fa.set != "val"]
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), sharex=True)
for ax, data in zip(axs, ["kvasir", "ham"]):
    for k in cost:
        a = ft[(ft.data == data) & (ft.score == k)].aurc.mean()
        ax.plot(np.mean(cost[k]), a, MK[k], color=COL[k], ms=7, mfc=COL[k], mec="white", mew=0.6, label=NAME[k], ls="none")
    ax.set_xscale("log")
    ax.set_title(DATA[data] + " (mean over evaluation sets)", fontsize=9)
    ax.set_xlabel("CPU time per image (ms, log scale)")
axs[0].set_ylabel("AURC (lower is better)")
h, l_ = axs[0].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(rect=(0, 0.1, 1, 1))
V.save(fig, os.path.join(F, "fig_tradeoff"))

# ---------------------------------------------------------------- fig_rank
rk = ft[ft.member != "ens"].groupby(["data", "encoder", "set", "score"]).aurc.mean().unstack("score")[SINGLE]
ranks = rk.rank(axis=1)
ranks = ranks.loc[[i for i in ranks.index if i[2] in SET]]
fig, ax = plt.subplots(figsize=(W2, 3.6))
ax.imshow(ranks.values.T, cmap=V.SEQ.reversed(), aspect="auto", vmin=1, vmax=len(SINGLE))
for i in range(ranks.shape[0]):
    for j in range(ranks.shape[1]):
        v = ranks.values[i, j]
        ax.text(i, j, "%d" % v, ha="center", va="center", fontsize=8, color="white" if v <= 4 else V.INK)
ax.set_yticks(range(len(SINGLE)))
ax.set_yticklabels([NAME[k] for k in SINGLE], fontsize=8)
ax.set_xticks(range(ranks.shape[0]))
ax.set_xticklabels(["%s %s\n%s" % ("K" if d == "kvasir" else "H", "CNX" if "convnext" in e else "R34", SET[s]) for d, e, s in ranks.index],
                   fontsize=8, rotation=90)
ax.grid(False)
ax.set_xlabel("Setting (K = Kvasir-SEG, H = HAM10000; CNX = ConvNeXt-T, R34 = ResNet-34)")
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_rank"))

# ---------------------------------------------------------------- fig_area
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8), sharey=False)
d = img[(img.member == "0") & (img.set == "test") & (img.encoder == "convnext_tiny.fb_in1k")]
for ax, k in zip(axs, ["ent_det", "ent_det_fg"]):
    for data, mk_ in [("kvasir", "o"), ("ham", "s")]:
        q = d[d.data == data]
        fail = q.dice_det < 0.5
        ax.scatter(100 * q.area_pred[~fail], q[k][~fail], s=8, marker=mk_, color=COL[k], alpha=0.5, lw=0, label="%s, Dice >= 0.5" % DATA[data])
        ax.scatter(100 * q.area_pred[fail], q[k][fail], s=26, marker=mk_, facecolor="white", edgecolor=V.INK, lw=1.0, label="%s, Dice < 0.5" % DATA[data])
    ax.set_xlabel("Predicted area (% of image)")
    ax.set_ylabel(NAME[k])
    ax.set_xscale("symlog", linthresh=1)
h, l_ = axs[1].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=4, fontsize=8, markerscale=1.2, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1))
V.save(fig, os.path.join(F, "fig_area"))

# ---------------------------------------------------------------- fig_examples (Kvasir test, ConvNeXt-T seed 0)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
eid = reg[(reg.data == "kvasir") & (reg.encoder == "convnext_tiny.fb_in1k") & (reg.seed == 0)].exp_id.iloc[0]
sc = pd.read_csv(os.path.join(R, "experiments", eid, "scores_test.csv"))
mp = np.load(os.path.join(R, "experiments", eid, "maps_test.npy"), mmap_mode="r")
X = np.load(os.path.join(R, "data", "processed", "kvasir_images.npy"), mmap_mode="r")
M = np.load(os.path.join(R, "data", "processed", "kvasir_masks.npy"), mmap_mode="r")
sc["unc_pdice_mc"] = 1 - sc.pdice_mc
pr = {k: 100 * sc[k].rank(pct=True) for k in ["ent_det", "ent_det_fg", "unc_pdice_mc"]}
picks = [int(np.argmin(sc.dice_det.values)), int(np.argsort(sc.dice_det.values)[len(sc) // 2])]
fig, axs = plt.subplots(2, 4, figsize=(W2, 3.9))
for r_, i in enumerate(picks):
    j = int(sc.array_index[i])
    p = np.asarray(mp[i], np.float32) / 255
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    e = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))
    for c, (im, title, cmap) in enumerate([(np.asarray(X[j]), "Image", None), (np.asarray(M[j]), "Reference mask", "gray"),
                                           (p > 0.5, "Prediction (Dice %.2f)" % sc.dice_det[i], "gray"), (e, "Pixel entropy", V.SEQ)]):
        ax = axs[r_, c]
        ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=(np.log(2) if c == 3 else 1) if cmap else None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=8.5)
    axs[r_, 0].set_ylabel("Worst case" if r_ == 0 else "Median case", fontsize=8.5)
    axs[r_, 3].text(1.03, 0.5, "Uncertainty percentile\nimage entropy %.0f\nforeground entropy %.0f\nMC agreement %.0f" % (
        pr["ent_det"][i], pr["ent_det_fg"][i], pr["unc_pdice_mc"][i]), transform=axs[r_, 3].transAxes, fontsize=8, va="center")
fig.tight_layout()
V.save(fig, os.path.join(F, "fig_examples"))
print("FIGURES_DONE")
