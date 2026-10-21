"""Perturbation tests on the test fold of finished runs: does the prediction depend on the tumour,
and is the class-evidence map faithful to the model?

Usage: python explain.py <brain-tumour-mri-explainability dir> [--only EXP001,EXP002] [--limit N] [--device cuda]
For every run with results.json and no explain.json, writes into the run directory:
  explain.csv   one row per test image
  explain.json  means over the test fold and the two deletion curves
Nothing is trained and no existing file is changed.

Tests (the class followed is the one the model predicts on the intact image; removed pixels are set
to the image's own mean intensity):
  tumour occlusion  : remove the pixels of the tumour mask; record the drop in probability and whether
                      the prediction flips.
  control occlusion : remove the same shape moved to a random place that does not touch the tumour and
                      lies at least 80% on non-background pixels (up to 30 tries; otherwise missing).
  deletion curve    : remove 16-pixel patches in order of decreasing evidence, 5% of the patches at a
                      time up to 50%; area under the probability curve (lower = more faithful map).
                      The same with a random patch order is the reference.
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
ap.add_argument("--only", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--batch-size", type=int, default=64)
args = ap.parse_args()

PATCH, FRACS, BG = 16, np.arange(0, 0.5001, 0.05), -1.8  # BG: normalised value of raw intensity 0.05
images, masks, meta = load_arrays(os.path.join(args.root, "data", "processed"))
reg = pd.read_csv(os.path.join(args.root, "experiments", "experiment_registry.csv"))
only = set(filter(None, args.only.split(",")))


def control_mask(m, fg, rng):
    n = m.sum()
    for _ in range(30):
        c = np.roll(m, (rng.integers(-m.shape[0] // 2, m.shape[0] // 2), rng.integers(-m.shape[1] // 2, m.shape[1] // 2)), (0, 1))
        if not (c & m).any() and (c & fg).sum() >= 0.8 * n:
            return c
    return None


@torch.no_grad()
def explain(exp):
    d = os.path.join(args.root, "experiments", exp)
    cfg = json.load(open(os.path.join(d, "results.json")))["config"]
    te = np.load(os.path.join(d, "split_indices.npz"))["test"]
    if args.limit:
        te = te[:args.limit]
    model = EvidenceNet(cfg["model"], pretrained=False, img_size=cfg["img_size"]).to(args.device)
    model.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=args.device))
    model.eval()
    dl = torch.utils.data.DataLoader(TumourSet(images, masks, meta, te, size=cfg["img_size"]), batch_size=args.batch_size)
    rng, gen = np.random.default_rng(0), torch.Generator().manual_seed(0)
    g = cfg["img_size"] // PATCH
    rows, curves = [], {"evidence": [], "random": []}
    for x, y, m, j in dl:
        x, m = x.to(args.device), m.to(args.device)
        ar = torch.arange(len(x), device=args.device)
        logits, ev = model(x, return_evidence=True)
        p = logits.softmax(1)
        c = p.argmax(1)
        p0 = p[ar, c]
        fill = x.mean((1, 2, 3), keepdim=True)

        def prob(mask01):
            q = model(x * (1 - mask01) + fill * mask01).softmax(1)
            return q[ar, c], q.argmax(1)

        pt, ct = prob(m)
        mnp, fg = m[:, 0].cpu().numpy() > 0, (x[:, 0] > BG).cpu().numpy()
        ctrl = [control_mask(mnp[b], fg[b], rng) for b in range(len(x))]
        ok = np.array([k is not None for k in ctrl])
        cm = torch.from_numpy(np.stack([k if k is not None else np.zeros_like(mnp[0]) for k in ctrl]).astype(np.float32))[:, None].to(args.device)
        pc, cc = prob(cm)

        heat = F.interpolate(F.relu(ev[ar, c])[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)
        orders = {"evidence": F.avg_pool2d(heat, PATCH).flatten(1).argsort(1, descending=True),
                  "random": torch.rand(len(x), g * g, generator=gen).argsort(1).to(args.device)}
        auc = {}
        for name, order in orders.items():
            curve = []
            for f in FRACS:
                pm = torch.zeros(len(x), g * g, device=args.device)
                pm.scatter_(1, order[:, :int(round(f * g * g))], 1.0)
                up = pm.view(-1, 1, g, g).repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)
                curve.append(prob(up)[0].cpu().numpy())
            curve = np.stack(curve, 1)  # [B, steps]
            curves[name].append(curve)
            auc[name] = np.trapezoid(curve, FRACS, axis=1) / FRACS[-1]
        for b in range(len(x)):
            rows.append(dict(array_index=int(j[b]), label=int(y[b]), pred=int(c[b]), p_intact=float(p0[b]),
                             drop_tumour=float(p0[b] - pt[b]), flip_tumour=int(ct[b] != c[b]),
                             drop_control=float(p0[b] - pc[b]) if ok[b] else np.nan,
                             flip_control=int(cc[b] != c[b]) if ok[b] else np.nan,
                             del_auc_evidence=float(auc["evidence"][b]), del_auc_random=float(auc["random"][b])))
    df = pd.DataFrame(rows)
    summary = {k: float(df[k].mean()) for k in df.columns if k not in ("array_index", "label", "pred")}
    summary.update(n_test=len(df), n_control_found=int(df["drop_control"].notna().sum()), fractions=FRACS.tolist(),
                   curve_evidence=np.concatenate(curves["evidence"]).mean(0).tolist(),
                   curve_random=np.concatenate(curves["random"]).mean(0).tolist())
    if args.limit:  # smoke test: print only, write nothing
        print(exp, json.dumps(summary))
        return
    df.to_csv(os.path.join(d, "explain.csv"), index=False)
    json.dump(summary, open(os.path.join(d, "explain.json"), "w"), indent=1)
    print(exp, "drop tumour %.3f control %.3f | deletion AUC evidence %.3f random %.3f" % (
        summary["drop_tumour"], summary["drop_control"], summary["del_auc_evidence"], summary["del_auc_random"]), flush=True)


for e in reg["exp_id"]:
    d = os.path.join(args.root, "experiments", e)
    if only and e not in only:
        continue
    if os.path.exists(os.path.join(d, "results.json")) and not os.path.exists(os.path.join(d, "explain.json")):
        explain(e)
