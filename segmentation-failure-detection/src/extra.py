"""Additional analyses of segmentation-failure-detection from the stored maps (CPU only, vectorised). Usage: python extra.py <segmentation-failure-detection dir>

Writes results/processed/:
  ablation_dilation.csv   AURC of single-pass foreground entropy for widening d in {0, 1, 2, 5, 10, 20} pixels, recomputed from
                          the stored probability maps (uint8), plus the whole-image entropy from the same maps (d = image)
  ensemble_size.csv       AURC of ensemble agreement and Dice of the ensemble for every subset of K = 2..5 of the five seeds
  failure_types.csv       every failed single-pass prediction (Dice < 0.5) with its type (empty / misplaced / under / over) and the
                          percentile of each score within its evaluation set (100 = most doubtful)
  operating_points.csv    for each score, run and set: mean Dice of the accepted images and number of accepted failures at
                          coverage 0.8 and 0.9
Widening uses a square maximum filter of side 2d+1, the same operation as the 11 x 11 max-pooling in train.py for d = 5.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

R = sys.argv[1]
E = os.path.join(R, "experiments")
P = os.path.join(R, "data", "processed")
O = os.path.join(R, "results", "processed")
reg = pd.read_csv(os.path.join(E, "experiment_registry.csv"))
SETS = {"kvasir": ["test", "test_blur", "test_noise", "test_contrast"], "ham": ["test", "test_blur", "test_noise", "test_contrast", "shift"]}
DIL = [0, 1, 2, 5, 10, 20]
pv = np.clip(np.arange(256) / 255.0, 1e-6, 1 - 1e-6)
ENT = (-(pv * np.log(pv) + (1 - pv) * np.log(1 - pv))).astype(np.float32)


def aurc(risk, score):
    o = np.argsort(score, kind="stable")
    r = risk[o]
    return float(np.mean(np.cumsum(r) / np.arange(1, len(r) + 1)))


def dice_v(a, b):
    """Dice per image for boolean stacks [N, H, W]; 1 when both are empty."""
    s = a.sum((1, 2)) + b.sum((1, 2))
    inter = np.logical_and(a, b).sum((1, 2))
    return np.where(s == 0, 1.0, 2.0 * inter / np.maximum(s, 1))


rows_d, rows_k = [], []
for (data, enc), g in reg.groupby(["data", "encoder"], sort=False):
    masks = np.load(os.path.join(P, "%s_masks.npy" % data), mmap_mode="r")
    for st in SETS[data]:
        maps, idx = [], None
        for r in g.sort_values("seed").itertuples():
            sc = pd.read_csv(os.path.join(E, r.exp_id, "scores_%s.csv" % st))
            idx = sc.array_index.values
            mp = np.load(os.path.join(E, r.exp_id, "maps_%s.npy" % st))
            maps.append(mp)
            e = ENT[mp]                      # [N, H, W] float32
            m = mp > 127
            risk = 1 - sc.dice_det.values
            rows_d.append(dict(data=data, encoder=enc, seed=r.seed, set=st, d="image", aurc=aurc(risk, e.mean((1, 2)))))
            for d in DIL:
                f = ndimage.maximum_filter(m, size=(1, 2 * d + 1, 2 * d + 1)) if d else m
                cnt = f.sum((1, 2))
                fg = np.where(cnt > 0, (e * f).sum((1, 2)) / np.maximum(cnt, 1), 0.0)
                row = dict(data=data, encoder=enc, seed=r.seed, set=st, d=str(d), aurc=aurc(risk, fg))
                if d == 5:
                    row["recomputed_vs_stored"] = float(np.corrcoef(fg, sc.ent_det_fg.values)[0, 1])
                rows_d.append(row)
            del e, m
        gt = np.stack([np.asarray(masks[j]).astype(bool) for j in idx])
        bins = [mp > 127 for mp in maps]
        pair = {(a, b): dice_v(bins[a], bins[b]) for a, b in itertools.combinations(range(5), 2)}
        for K in range(2, 6):
            for sub in itertools.combinations(range(5), K):
                tot = np.zeros(maps[0].shape, np.float32)
                for k in sub:
                    tot += maps[k]
                pe = tot / (255.0 * K) > 0.5
                de = dice_v(pe, gt)
                agr = 1 - np.mean([pair[ab] for ab in itertools.combinations(sub, 2)], axis=0)
                rows_k.append(dict(data=data, encoder=enc, set=st, K=K, subset="".join(map(str, sub)), dice=float(de.mean()), aurc=aurc(1 - de, agr)))
        del maps, bins, gt
        print(data, enc, st, flush=True)
dd = pd.DataFrame(rows_d)
dd.to_csv(os.path.join(O, "ablation_dilation.csv"), index=False)
pd.DataFrame(rows_k).to_csv(os.path.join(O, "ensemble_size.csv"), index=False)

# ---------------------------------------------------------------- failure types and operating points
img = pd.read_csv(os.path.join(O, "image_scores.csv"), low_memory=False)
img = img[(img.member != "ens") & (img.set != "val")]
US = ["ent_det", "ent_det_fg", "ent_det_fg_e", "unc_pdice_mc", "unc_pdice_tta", "ref_irreg"]
JUDGE = {"ent_det": "dice_det", "ent_det_fg": "dice_det", "ent_det_fg_e": "dice_det", "ref_irreg": "dice_det", "unc_pdice_mc": "dice_mc",
         "unc_pdice_tta": "dice_tta"}
ft, op = [], []
for (data, enc, member, st), g in img.groupby(["data", "encoder", "member", "set"]):
    pc = g[US].rank(pct=True) * 100
    eid = reg[(reg.data == data) & (reg.encoder == enc) & (reg.seed == int(member))].exp_id.iloc[0]
    mp = np.load(os.path.join(E, eid, "maps_%s.npy" % st), mmap_mode="r")
    masks = np.load(os.path.join(P, "%s_masks.npy" % data), mmap_mode="r")
    pos = {a: k for k, a in enumerate(pd.read_csv(os.path.join(E, eid, "scores_%s.csv" % st)).array_index.values)}
    for i in np.where(g.dice_det.values < 0.5)[0]:
        row = g.iloc[i]
        pr = np.asarray(mp[pos[row.array_index]]) > 127
        gt = np.asarray(masks[int(row.array_index)]).astype(bool)
        inter = np.logical_and(pr, gt).sum()
        prec = inter / pr.sum() if pr.sum() else np.nan
        rec = inter / gt.sum()
        typ = "empty" if pr.sum() == 0 else "misplaced" if (prec < 0.5 and rec < 0.5) else "under" if prec >= 0.5 else "over"
        ft.append(dict(data=data, encoder=enc, member=member, set=st, array_index=int(row.array_index), dice=row.dice_det, type=typ,
                       precision=prec, recall=rec, **{"pct_" + k: pc.iloc[i][k] for k in US}))
    n = len(g)
    for k in US:
        o = np.argsort(g[k].values, kind="stable")
        dk = g[JUDGE[k]].values[o]
        for cov in [0.8, 0.9]:
            m = int(round(cov * n))
            op.append(dict(data=data, encoder=enc, member=member, set=st, score=k, coverage=cov, mean_dice_accepted=dk[:m].mean(),
                           failures_accepted=int((dk[:m] < 0.5).sum()), failures_total=int((dk < 0.5).sum()), n=n))
pd.DataFrame(ft).to_csv(os.path.join(O, "failure_types.csv"), index=False)
pd.DataFrame(op).to_csv(os.path.join(O, "operating_points.csv"), index=False)
print("EXTRA_DONE")
