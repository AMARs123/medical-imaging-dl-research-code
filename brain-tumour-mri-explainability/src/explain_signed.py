"""Deletion test with patches ranked by SIGNED evidence (complements explain.py).

explain.py ranks patches by positive evidence only. When most of a map is <= 0 (as after evidence
supervision), many patches tie at zero and argsort breaks the ties by position, so the later part of the
deletion order is not evidence-based. Here patches are ranked by their mean signed evidence, which has no
ties in practice. Writes explain_signed.json (mean curve and mean AUC, plus the random-order reference with
the same generator as explain.py) into each run directory of the main experiments (phases A-C).
Usage: python explain_signed.py <brain-tumour-mri-explainability dir> [--device cuda] [--limit N]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import EvidenceNet, TumourSet, load_arrays

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
PATCH, FRACS = 16, np.arange(0, 0.5001, 0.05)
images, masks, meta = load_arrays(os.path.join(args.root, "data", "processed"))
reg = pd.read_csv(os.path.join(args.root, "experiments", "experiment_registry.csv"))
sel = reg[reg.phase.isin(["A_protocol", "B_baseline", "C_saliency"])]

for e in sel.exp_id:
    d = os.path.join(args.root, "experiments", e)
    if not os.path.exists(os.path.join(d, "results.json")) or os.path.exists(os.path.join(d, "explain_signed.json")):
        continue
    cfg = json.load(open(os.path.join(d, "results.json")))["config"]
    te = np.load(os.path.join(d, "split_indices.npz"))["test"]
    if args.limit:
        te = te[:args.limit]
    model = EvidenceNet(cfg["model"], pretrained=False, img_size=cfg["img_size"]).to(args.device)
    model.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=args.device))
    model.eval()
    g = cfg["img_size"] // PATCH
    curves = []
    with torch.no_grad():
        for x, y, m, j in torch.utils.data.DataLoader(TumourSet(images, masks, meta, te, size=cfg["img_size"]), batch_size=64):
            x = x.to(args.device)
            ar = torch.arange(len(x), device=args.device)
            logits, ev = model(x, return_evidence=True)
            c = logits.argmax(1)
            fill = x.mean((1, 2, 3), keepdim=True)
            heat = F.interpolate(ev[ar, c][:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)
            order = F.avg_pool2d(heat, PATCH).flatten(1).argsort(1, descending=True)
            cur = []
            for f in FRACS:
                pm = torch.zeros(len(x), g * g, device=args.device)
                pm.scatter_(1, order[:, :int(round(f * g * g))], 1.0)
                up = pm.view(-1, 1, g, g).repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)
                cur.append(model(x * (1 - up) + fill * up).softmax(1)[ar, c].cpu().numpy())
            curves.append(np.stack(cur, 1))
    cv = np.concatenate(curves)
    res = dict(del_auc_signed=float((np.trapezoid(cv, FRACS, axis=1) / FRACS[-1]).mean()), curve_signed=cv.mean(0).tolist(), n_test=len(cv))
    if args.limit:
        print("smoke", e, res["del_auc_signed"])
        break
    json.dump(res, open(os.path.join(d, "explain_signed.json"), "w"), indent=1)
    print(e, cfg["model"], cfg["lambda_sal"], "%.3f" % res["del_auc_signed"], flush=True)
