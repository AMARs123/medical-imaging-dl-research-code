"""Attribution maps of all Paper 03 models on the PTB-XL test fold, and per-map measures of localisation and faithfulness.

Usage: python attribute.py <Paper_03 dir> [--limit N]
For every test ECG and every superclass that is present in its label (ground truth), each of the 18 models
(15 trained + 3 randomly initialised) and each method gives a relevance profile over time r(t), t = 0..999 (10 ms steps).

Methods (attribution of the class logit):
  saliency : |d logit / d x|, summed over leads                               (unsigned by construction)
  gxi      : x * d logit / d x, positive part, summed over leads
  ig       : integrated gradients, 32 steps, baseline 0 (= training-mean signal after standardisation), positive part, summed over leads
  occlusion: drop of the logit when a 200 ms window (all leads) is set to 0, stride 100 ms, positive part, spread over the window
  cam      : class-activation map (time map of the last feature layer, exact logit decomposition), positive part, linearly upsampled
Positive part = evidence *for* the class. Leads are summed so that all five methods live on the same time axis.

Outputs (results/attributions/):
  profiles_<method>.npy  float16 [n_items, n_models, 1000]   relevance profiles
  items.csv              one row per (test ECG, label): ecg_id, label, array_index, prob of each model
  local_<method>.csv     per item x model: share of relevance in P, QRS, ST-T, other; time shares of the segments
  deletion_<method>.csv  per item x model: logit drop after deleting the top 10% / random 10% of time steps
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ECGNet  # noqa: E402
from segments import segment_masks  # noqa: E402

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
METHODS = ["saliency", "gxi", "ig", "occlusion", "cam"]
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--models", default="", help="smoke test only: comma-separated exp_ids to use")
ap.add_argument("--out", default="", help="smoke test only: output folder")
args = ap.parse_args()
R = args.root
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = args.out or os.path.join(R, "results", "attributions")
os.makedirs(out, exist_ok=True)

reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
if args.models:
    reg = reg[reg.exp_id.isin(args.models.split(","))].reset_index(drop=True)
X = np.load(os.path.join(R, "data", "processed", "signals.npy"), mmap_mode="r")
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
te = np.where(meta.strat_fold == 10)[0]
items = [(i, k) for i in te for k in range(len(SUPER)) if meta[SUPER[k]].values[i] == 1]
if args.limit:
    items = items[:args.limit]
items = pd.DataFrame(items, columns=["array_index", "label"])
items["ecg_id"] = meta.ecg_id.values[items.array_index]
items["label_name"] = [SUPER[k] for k in items.label]
n = len(items)
print("items", n, "models", len(reg), flush=True)

models = []
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    m = ECGNet(r["model"]).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=dev))
    nz = np.load(os.path.join(d, "norm.npz"))
    models.append((r, m, torch.tensor(nz["mean"], device=dev), torch.tensor(nz["std"], device=dev)))

prof = {mt: np.lib.format.open_memmap(os.path.join(out, "profiles_%s.npy" % mt), "w+", np.float16, (n, len(models), 1000)) for mt in METHODS}
probs = np.zeros((n, len(models)), np.float32)
dele = {mt: np.zeros((n, len(models), 3), np.float32) for mt in METHODS}  # base logit, drop top 10%, drop random 10%
rng = np.random.default_rng(0)


def logit_of(m, x, lab):
    return m(x).gather(1, lab[:, None])[:, 0]


for s in range(0, n, args.batch):
    b = items.iloc[s:s + args.batch]
    raw = torch.from_numpy(np.asarray(X[b.array_index.values])).to(dev)
    lab = torch.tensor(b.label.values, device=dev)
    for j, (r, m, mu, sd) in enumerate(models):
        x = ((raw - mu) / sd).float()
        with torch.no_grad():
            base = logit_of(m, x, lab)
        probs[s:s + len(b), j] = torch.sigmoid(base).cpu().numpy()
        res = {}
        # gradient-based
        xg = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(logit_of(m, xg, lab).sum(), xg)
        res["saliency"] = g.abs().sum(1)
        res["gxi"] = (x * g).clamp(min=0).sum(1)
        steps = 32
        ig = torch.zeros_like(x)
        for a in (torch.arange(steps, device=dev) + 0.5) / steps:  # midpoint Riemann sum
            xa = (a * x).requires_grad_(True)
            ga, = torch.autograd.grad(logit_of(m, xa, lab).sum(), xa)
            ig += ga / steps
        res["ig"] = (x * ig).clamp(min=0).sum(1)
        # occlusion: 200 ms windows, stride 100 ms
        occ = torch.zeros(len(b), 1000, device=dev)
        cnt = torch.zeros(1000, device=dev)
        with torch.no_grad():
            for t0 in range(0, 1000 - 20 + 1, 10):
                xo = x.clone()
                xo[:, :, t0:t0 + 20] = 0
                occ[:, t0:t0 + 20] += (base - logit_of(m, xo, lab))[:, None]
                cnt[t0:t0 + 20] += 1
        res["occlusion"] = (occ / cnt.clamp(min=1)).clamp(min=0)
        with torch.no_grad():
            cam = m.cam(x)[torch.arange(len(b)), lab]  # [B, T']
            res["cam"] = F.interpolate(cam[:, None], size=1000, mode="linear", align_corners=False)[:, 0].clamp(min=0)
        # deletion test: top 10% of time steps of the profile vs 10% random time steps, all leads set to 0
        with torch.no_grad():
            for mt, p in res.items():
                prof[mt][s:s + len(b), j] = p.cpu().numpy().astype(np.float16)
                k = 100
                top = p.topk(k, dim=1).indices
                xd = x.clone()
                xd.scatter_(2, top[:, None, :].expand(-1, 12, -1), 0.0)
                rnd = torch.stack([torch.from_numpy(rng.choice(1000, k, replace=False)) for _ in range(len(b))]).to(dev)
                xr = x.clone()
                xr.scatter_(2, rnd[:, None, :].expand(-1, 12, -1), 0.0)
                dele[mt][s:s + len(b), j, 0] = base.cpu().numpy()
                dele[mt][s:s + len(b), j, 1] = (base - logit_of(m, xd, lab)).cpu().numpy()
                dele[mt][s:s + len(b), j, 2] = (base - logit_of(m, xr, lab)).cpu().numpy()
    print("done", min(s + args.batch, n), "of", n, flush=True)

for mt in METHODS:
    prof[mt].flush()
for j, (r, *_) in enumerate(models):
    items["prob_" + r["exp_id"]] = probs[:, j]
items.to_csv(os.path.join(out, "items.csv"), index=False)

# localisation in waveform segments (fiducial points from PTB-XL+); items without usable fiducials are recorded and skipped
masks, ok = [], []
for eid in items.ecg_id:
    mk = segment_masks(int(eid))
    ok.append(mk is not None)
    masks.append(mk if mk is not None else np.zeros((4, 1000), bool))
masks = np.stack(masks)  # [n, 4 segments (P, QRS, STT, other), 1000]
items["fiducials_ok"] = ok
items[["ecg_id", "label_name", "fiducials_ok"]].to_csv(os.path.join(out, "fiducial_usage.csv"), index=False)
SEG = ["P", "QRS", "STT", "other"]
for mt in METHODS:
    P = np.asarray(prof[mt], np.float32)
    tot = P.sum(2) + 1e-12
    rows = dict(item=np.repeat(np.arange(n), len(models)), model_idx=np.tile(np.arange(len(models)), n))
    for si, sg in enumerate(SEG):
        share = (P * masks[:, None, si, :]).sum(2) / tot
        rows["share_" + sg] = share.ravel()
        rows["time_" + sg] = np.repeat(masks[:, si, :].mean(1), len(models))
    rows["fiducials_ok"] = np.repeat(np.array(ok), len(models))
    rows["zero_map"] = (tot.ravel() < 1e-9)
    pd.DataFrame(rows).to_csv(os.path.join(out, "local_%s.csv" % mt), index=False)
    d = dele[mt]
    pd.DataFrame(dict(item=np.repeat(np.arange(n), len(models)), model_idx=np.tile(np.arange(len(models)), n),
                      base_logit=d[:, :, 0].ravel(), drop_top10=d[:, :, 1].ravel(), drop_rand10=d[:, :, 2].ravel())).to_csv(
        os.path.join(out, "deletion_%s.csv" % mt), index=False)
json.dump(dict(items=n, models=reg.exp_id.tolist(), methods=METHODS, segments=SEG, fiducials_ok=int(np.sum(ok))),
          open(os.path.join(out, "info.json"), "w"), indent=1)
print("ATTRIBUTE_DONE")
