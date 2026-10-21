"""External check on the part of BRISC 2025 that is independent of the Figshare data (inference only).

Test set: BRISC tumour slices with best correlation < 0.95 to every Figshare slice (preprocess_brisc.py), after
removing repeats inside BRISC (greedy: a slice is dropped if it correlates >= 0.95 with a slice already kept).
No setting is tuned on this set. For every finished main ID-root run with lambda 0 or 0.25 (split seed 0, mask
widening 1, full data), the stored best_model.pt predicts every external slice, and the evidence map of the
predicted class is scored against the BRISC mask with the same definitions as on Figshare:
  pointing (pixel), mass_inside, iou_area (common.localisation_metrics), pointing at map resolution and its
  chance level (diag_evidence.py), and the occlusion drop for the tumour and for a control region (explain.py
  rule: same shape, no tumour contact, >= 80% non-background, up to 30 tries, fill = image mean).
Writes external.json into each run directory (skipped if present) and results/processed/external_runs.csv.
Usage: python eval_external.py <Paper_01 dir> [--device cpu|cuda] [--limit N]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MEAN, STD, EvidenceNet, localisation_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--device", default="cpu")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
R = args.root
X = os.path.join(R, "data", "external", "processed")
meta = pd.read_csv(os.path.join(X, "meta.csv"))
images = np.load(os.path.join(X, "images.npy"))
masks = np.load(os.path.join(X, "masks.npy"))

# independent slices, then repeats inside BRISC removed
cand = np.where(~meta.near_duplicate.values)[0]
v = np.stack([cv2.resize(images[i], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for i in cand])
v -= v.mean(1, keepdims=True)
v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
keep = []
for k in range(len(cand)):
    if not keep or (v[keep] @ v[k]).max() < 0.95:
        keep.append(k)
idx = cand[keep]
pd.DataFrame({"external_index": idx, "file": meta.file.values[idx], "label": meta.label.values[idx]}).to_csv(
    os.path.join(X, "external_test_set.csv"), index=False)
print("independent slices: %d, after removing repeats inside BRISC: %d, labels %s" % (
    len(cand), len(idx), np.bincount(meta.label.values[idx], minlength=3).tolist()), flush=True)

x = torch.from_numpy(images[idx].astype(np.float32) / 255.0)[:, None]
x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
x = ((x - MEAN) / STD).repeat(1, 3, 1, 1)
m224 = F.interpolate(torch.from_numpy(masks[idx].astype(np.float32))[:, None], size=224, mode="nearest")[:, 0].numpy()
y = meta.label.values[idx]
rng = np.random.default_rng(0)


def control_mask(mk, img):
    """Same tumour shape moved to a random place that does not touch the tumour and is >= 80% non-background."""
    ys, xs = np.nonzero(mk)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    shape = mk[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    fg = img > 0.05
    for _ in range(30):
        oy, ox = rng.integers(0, 224 - h + 1), rng.integers(0, 224 - w + 1)
        c = np.zeros_like(mk)
        c[oy:oy + h, ox:ox + w] = shape
        if (c * mk).sum() == 0 and fg[c > 0].mean() >= 0.8:
            return c
    return None


raw224 = F.interpolate(torch.from_numpy(images[idx].astype(np.float32) / 255.0)[:, None], size=224, mode="bilinear", align_corners=False)[:, 0].numpy()
ctrl = [control_mask(m224[i], raw224[i]) for i in range(len(idx))]

reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
for c, d in [("split_seed", 0), ("dilate", 1), ("train_frac", 1.0)]:
    reg[c] = reg[c].fillna(d)
sel = reg[(reg.protocol == "root") & (reg.split_seed == 0) & (reg.dilate == 1) & (reg.train_frac == 1.0) & reg.lambda_sal.isin([0.0, 0.25])]
if args.limit:
    sel = sel.head(args.limit)
rows = []
torch.set_num_threads(4)
for r in sel.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    out = os.path.join(d, "external.json")
    if not os.path.exists(os.path.join(d, "results.json")):
        continue
    if os.path.exists(out):
        rows.append(dict(r, **json.load(open(out))))
        continue
    net = EvidenceNet(r["model"], 3, pretrained=False).to(args.device).eval()
    net.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=args.device))
    probs, loc, cell_hit, cell_chance, occ_t, occ_c = [], [], [], [], [], []
    with torch.no_grad():
        for b in range(0, len(idx), 32):
            xb = x[b:b + 32].to(args.device)
            logits, e = net(xb, return_evidence=True)
            p = logits.float().softmax(1)
            pred = p.argmax(1)
            ep = e.float()[torch.arange(len(pred)), pred]  # predicted-class map [B, h, w]
            heat = F.interpolate(F.relu(ep)[:, None], size=224, mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            hh, ww = ep.shape[-2:]
            mc = F.adaptive_max_pool2d(torch.from_numpy(m224[b:b + 32])[:, None], (hh, ww))[:, 0].numpy() > 0
            am = ep.reshape(len(pred), -1).argmax(1).cpu().numpy()
            for k in range(len(pred)):
                loc.append(localisation_metrics(heat[k], m224[b + k]))
                cell_hit.append(float(mc[k].ravel()[am[k]]))
                cell_chance.append(float(mc[k].mean()))
            # occlusion: fill the tumour (or the control region) with the image mean, watch the predicted-class probability
            for k in range(len(pred)):
                i = b + k
                fill = xb[k].mean()
                xt = xb[k].clone()
                xt[:, torch.from_numpy(m224[i] > 0)] = fill
                pt = net(xt[None]).float().softmax(1)[0, pred[k]]
                occ_t.append(float(p[k, pred[k]] - pt))
                if ctrl[i] is not None:
                    xc = xb[k].clone()
                    xc[:, torch.from_numpy(ctrl[i] > 0)] = fill
                    occ_c.append(float(p[k, pred[k]] - net(xc[None]).float().softmax(1)[0, pred[k]]))
            probs.append(p.cpu().numpy())
    prob = np.concatenate(probs)
    pred = prob.argmax(1)
    L = pd.DataFrame(loc)
    res = dict(n=int(len(y)), accuracy=float((pred == y).mean()),
               recall_meningioma=float((pred[y == 0] == 0).mean()) if (y == 0).any() else None,
               recall_pituitary=float((pred[y == 2] == 2).mean()) if (y == 2).any() else None,
               predicted_share=np.bincount(pred, minlength=3).tolist(), mean_true_prob=float(prob[np.arange(len(y)), y].mean()),
               loc_pointing=float(L.pointing.mean()), loc_mass_inside=float(L.mass_inside.mean()), loc_iou_area=float(L.iou_area.mean()),
               pointing_cell=float(np.mean(cell_hit)), pointing_cell_chance=float(np.mean(cell_chance)),
               occlusion_tumour=float(np.mean(occ_t)), occlusion_control=float(np.mean(occ_c)), control_found=len(occ_c))
    res["pointing_lift"] = res["pointing_cell"] / res["pointing_cell_chance"]
    json.dump(res, open(out, "w"), indent=1)
    rows.append(dict(r, **res))
    print(r["exp_id"], r["model"], r["lambda_sal"], r["seed"], r["fold"], "acc %.3f lift %.2f mass %.3f" % (res["accuracy"], res["pointing_lift"], res["loc_mass_inside"]), flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(R, "results", "processed", "external_runs.csv"), index=False)
print(df.groupby(["model", "lambda_sal"])[["accuracy", "recall_meningioma", "pointing_lift", "loc_mass_inside", "occlusion_tumour", "occlusion_control"]].mean().round(3).to_string())
print("EXTERNAL_DONE", len(df))
