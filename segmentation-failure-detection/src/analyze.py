"""segmentation-failure-detection analysis: segmentation quality, image-level failure detection and pixel-level error detection, from stored runs only.

Usage: python analyze.py <segmentation-failure-detection dir>
Inputs : experiments/EXPnnn/{scores,maps,pix}_<set>.*, data/processed/<data>_{masks.npy,meta.csv}
Outputs: results/processed/
  image_scores.csv   one row per (condition, member, set, image): Dice of the judged segmentation and every score
                     member = seed 0..4 (single models) or "ens" (deep ensemble of the 5 seeds)
  failure.csv        per (condition, member, set, score): AURC, e-AURC, AUROC for failure (Dice < 0.5), Spearman(score, Dice), n, failures
  pixel.csv          per (condition, seed, set, score): AUROC of pixel uncertainty for pixel errors of the deterministic prediction
  calibration.csv    per (condition, member, set): pixel expected calibration error (15 bins) of the foreground probability
  quality.csv        per (condition, member, set): mean Dice of the deterministic, MC-dropout, TTA and ensemble predictions
Scores (higher = less trustworthy; sign of the shape references fixed on the validation set, never on test):
  ent_det, ent_det_fg               single pass: mean binary entropy over the image / over the dilated predicted foreground
  ent_mc, mi_mc, unc_pdice_mc       MC dropout (10 samples): predictive entropy, mutual information, 1 - mean pairwise Dice
  ent_tta, unc_pdice_tta            test-time augmentation (8 dihedral views): entropy, 1 - mean pairwise Dice
  ens_ent, ens_ent_fg, unc_pdice_ens  deep ensemble of 5 seeds: entropy of the mean map (image / dilated foreground), 1 - pairwise Dice
  ref_area, ref_parts, ref_irreg    references from the predicted mask only: (signed) area, number of connected parts,
                                    boundary irregularity (boundary pixels^2 / (4 pi area))
Each score judges the segmentation it comes with (deterministic for single-pass and reference scores, MC mean for MC scores,
TTA mean for TTA scores, ensemble mean for ensemble scores).
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage, stats

R = sys.argv[1]
E = os.path.join(R, "experiments")
P = os.path.join(R, "data", "processed")
O = os.path.join(R, "results", "processed")
os.makedirs(O, exist_ok=True)
reg = pd.read_csv(os.path.join(E, "experiment_registry.csv"))
SETS = {"kvasir": ["val", "test", "test_blur", "test_noise", "test_contrast"],
        "ham": ["val", "test", "shift", "test_blur", "test_noise", "test_contrast"]}
FAIL = 0.5  # an image counts as a failure when its Dice is below 0.5
JUDGE = {"ent_det": "dice_det", "ent_det_fg": "dice_det", "ent_det_fg_e": "dice_det", "ref_area": "dice_det", "ref_parts": "dice_det", "ref_irreg": "dice_det",
         "ent_mc": "dice_mc", "mi_mc": "dice_mc", "unc_pdice_mc": "dice_mc", "ent_tta": "dice_tta", "unc_pdice_tta": "dice_tta",
         "ens_ent": "dice_ens", "ens_ent_fg": "dice_ens", "ens_ent_fg_e": "dice_ens", "unc_pdice_ens": "dice_ens"}
SINGLE = ["ent_det", "ent_det_fg", "ent_det_fg_e", "ent_mc", "mi_mc", "unc_pdice_mc", "ent_tta", "unc_pdice_tta", "ref_area", "ref_parts", "ref_irreg"]
ENS = ["ens_ent", "ens_ent_fg", "ens_ent_fg_e", "unc_pdice_ens", "ref_area", "ref_parts", "ref_irreg"]
dil = np.ones((11, 11), bool)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def bent(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def shape_refs(m):
    """Area, number of connected parts and boundary irregularity of a binary mask."""
    a = float(m.mean())
    lab, n = ndimage.label(m)
    edge = m & ~ndimage.binary_erosion(m)
    irr = float(edge.sum() ** 2 / (4 * np.pi * max(m.sum(), 1)))
    return a, n, irr


def aurc(risk, score):
    """Area under the risk-coverage curve: images sorted by increasing score (most trusted first), risk = 1 - Dice."""
    o = np.argsort(score, kind="stable")
    r = risk[o]
    return float(np.mean(np.cumsum(r) / np.arange(1, len(r) + 1)))


def failure_metrics(dsc, score):
    risk = 1 - dsc
    a = aurc(risk, score)
    opt = aurc(risk, risk)
    y = (dsc < FAIL).astype(int)
    auroc = float(stats.mannwhitneyu(score[y == 1], score[y == 0]).statistic / (y.sum() * (1 - y).sum())) if 0 < y.sum() < len(y) else np.nan
    rho = float(stats.spearmanr(score, dsc).statistic)
    return dict(aurc=a, eaurc=a - opt, auroc_fail=auroc, spearman=rho, n=len(dsc), failures=int(y.sum()), mean_dice=float(dsc.mean()))


def ece(prob, gt, bins=15):
    """Pixel expected calibration error of the foreground probability (confidence = max(p, 1-p))."""
    conf = np.maximum(prob, 1 - prob)
    pred = prob > 0.5
    acc = pred == gt
    e = 0.0
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        s = idx == b
        if s.any():
            e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return float(e)


rows_img, rows_q, rows_cal, rows_pix = [], [], [], []
for (data, enc), g in reg.groupby(["data", "encoder"], sort=False):
    cond = "%s|%s" % (data, enc)
    masks = np.load(os.path.join(P, "%s_masks.npy" % data), mmap_mode="r")
    for st in SETS[data]:
        members = []
        for r in g.sort_values("seed").itertuples():
            sc = pd.read_csv(os.path.join(E, r.exp_id, "scores_%s.csv" % st))
            mp = np.load(os.path.join(E, r.exp_id, "maps_%s.npy" % st), mmap_mode="r")
            members.append((r.seed, r.exp_id, sc, mp))
            pz = np.load(os.path.join(E, r.exp_id, "pix_%s.npz" % st))
            err = pz["err"].astype(bool)
            for k in ["ent_det", "ent_mc", "mi_mc", "ent_tta"]:
                u = pz[k].astype(np.float32)
                au = float(stats.mannwhitneyu(u[err], u[~err]).statistic / (err.sum() * (~err).sum())) if 0 < err.sum() < len(err) else np.nan
                rows_pix.append(dict(cond=cond, data=data, encoder=enc, seed=r.seed, set=st, score=k, auroc_pixel=au,
                                     error_rate=float(err.mean()), n_pixels=len(err)))
        idx = members[0][2].array_index.values
        for _, _, sc, _ in members:
            assert (sc.array_index.values == idx).all()
        n = len(idx)
        ens_rows = []
        cal_single = {s: [] for s, _, _, _ in members}
        cal_ens_p, cal_gt = [], []
        for i in range(n):
            gt = np.asarray(masks[idx[i]]).astype(bool)
            probs = np.stack([np.asarray(mp[i], np.float32) / 255.0 for _, _, _, mp in members])
            bins_ = probs > 0.5
            pm = probs.mean(0)
            pe = pm > 0.5
            fg = ndimage.binary_dilation(pe, dil)
            e = bent(pm)
            pd_ = [dice(bins_[a], bins_[b]) for a, b in itertools.combinations(range(len(bins_)), 2)]
            a_, n_, irr_ = shape_refs(pe)
            ens_rows.append(dict(array_index=int(idx[i]), dice_ens=dice(pe, gt), ens_ent=float(e.mean()),
                                 ens_ent_fg=float(e[fg].mean()) if fg.any() else 0.0, unc_pdice_ens=1 - float(np.mean(pd_)),
                                 area=a_, parts=n_, irreg=irr_))
            # pixel calibration on a fixed random subsample of 4096 pixels per image (same pixels for every member)
            sel = np.random.default_rng(i).choice(gt.size, 4096, replace=False)
            cal_gt.append(gt.ravel()[sel])
            cal_ens_p.append(pm.ravel()[sel])
            for (s, _, _, _), p_ in zip(members, probs):
                cal_single[s].append(p_.ravel()[sel])
        ens = pd.DataFrame(ens_rows)
        gt_all = np.concatenate(cal_gt)
        for s, _, sc, mp in members:
            shp = np.array([shape_refs(np.asarray(mp[i]) > 127) for i in range(n)])
            d = sc.copy()
            d["unc_pdice_mc"] = 1 - d.pdice_mc
            d["unc_pdice_tta"] = 1 - d.pdice_tta
            d["area"], d["parts"], d["irreg"] = shp[:, 0], shp[:, 1], shp[:, 2]
            d.insert(0, "member", str(s))
            d.insert(0, "set", st)
            d.insert(0, "encoder", enc)
            d.insert(0, "data", data)
            rows_img.append(d)
            rows_q.append(dict(cond=cond, data=data, encoder=enc, member=str(s), set=st, n=n, dice_det=d.dice_det.mean(),
                               dice_mc=d.dice_mc.mean(), dice_tta=d.dice_tta.mean(), dice_ens=np.nan))
            rows_cal.append(dict(cond=cond, data=data, encoder=enc, member=str(s), set=st, ece=ece(np.concatenate(cal_single[s]), gt_all)))
        ens.insert(0, "member", "ens")
        ens.insert(0, "set", st)
        ens.insert(0, "encoder", enc)
        ens.insert(0, "data", data)
        rows_img.append(ens)
        rows_q.append(dict(cond=cond, data=data, encoder=enc, member="ens", set=st, n=n, dice_det=np.nan, dice_mc=np.nan, dice_tta=np.nan,
                           dice_ens=ens.dice_ens.mean()))
        rows_cal.append(dict(cond=cond, data=data, encoder=enc, member="ens", set=st, ece=ece(np.concatenate(cal_ens_p), gt_all)))
        print(cond, st, "n", n, "ens dice %.3f" % ens.dice_ens.mean(), flush=True)

img = pd.concat(rows_img, ignore_index=True)
# signs of the shape references, fixed on the validation sets of the single models (pooled over conditions and seeds):
# the reference counts as "untrustworthy when large" if its Spearman correlation with Dice on validation is negative
val = img[(img.set == "val") & (img.member != "ens")]
SIGN = {k: (1.0 if stats.spearmanr(val[k], val.dice_det).statistic < 0 else -1.0) for k in ["area", "parts", "irreg"]}
for k in ["area", "parts", "irreg"]:
    img["ref_" + k] = SIGN[k] * img[k]
# post-hoc variant (added after inspecting the worst cases): an empty predicted mask has no foreground to measure,
# so foreground entropy would call it certain; every image of both data sets contains a lesion, so an empty mask is
# given the largest possible score (1 nat, above the binary-entropy maximum ln 2)
img["ent_det_fg_e"] = np.where(img.area == 0, 1.0, img.ent_det_fg)
img["ens_ent_fg_e"] = np.where(img.area == 0, 1.0, img.ens_ent_fg)
img.to_csv(os.path.join(O, "image_scores.csv"), index=False)
pd.Series(SIGN).to_csv(os.path.join(O, "reference_signs.csv"), header=["sign"])

rows_f = []
for (data, enc, member, st), d in img.groupby(["data", "encoder", "member", "set"], sort=False):
    for k in (ENS if member == "ens" else SINGLE):
        dsc = d["dice_ens" if member == "ens" else JUDGE[k]].values
        rows_f.append(dict(cond="%s|%s" % (data, enc), data=data, encoder=enc, member=member, set=st, score=k,
                           **failure_metrics(dsc, d[k].values)))
pd.DataFrame(rows_f).to_csv(os.path.join(O, "failure.csv"), index=False)
pd.DataFrame(rows_q).to_csv(os.path.join(O, "quality.csv"), index=False)
pd.DataFrame(rows_cal).to_csv(os.path.join(O, "calibration.csv"), index=False)
pd.DataFrame(rows_pix).to_csv(os.path.join(O, "pixel.csv"), index=False)
print("signs", SIGN)
print("ANALYZE_DONE")
