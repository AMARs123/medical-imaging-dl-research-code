"""Three further attribution methods for Paper 03 (Research_Expert round, 2026-10-16), with the same outputs as attribute.py.

Methods (attribution of the class logit, positive part, summed over leads, on the same items and models as attribute.py):
  smoothgrad : mean of |d logit / d x| over 16 noisy copies of the input, noise SD = 0.15 x the range of the standardised ECG
  gradshap   : expected gradients (GradientSHAP): mean over 16 samples of (x - x') * d logit / d x at x' + a (x - x'),
               x' a random training-fold ECG (standardised), a ~ U(0, 1)
  gradcam_mid: Grad-CAM on the input of the last stage (XResNet1d: end of stage 3; Inception1d: output of block 4;
               ConvTran1d: output of encoder layer 2), channel weights = time-averaged gradients, ReLU, linear upsampling
Outputs (results/attributions/): profiles_<m>.npy, local_<m>.csv, deletion_<m>.csv, info_extra.json
Usage: python attribute_extra.py <Paper_03 dir> [--limit N] [--threads 8]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ECGNet  # noqa: E402
from segments import segment_masks  # noqa: E402

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
METHODS = ["smoothgrad", "gradshap", "gradcam_mid"]
NS = 16
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="")
args = ap.parse_args()
torch.set_num_threads(args.threads)
R = args.root
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = args.out or os.path.join(R, "results", "attributions")
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
rng = np.random.default_rng(0)

reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
X = np.load(os.path.join(R, "data", "processed", "signals.npy"), mmap_mode="r")
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
te = np.where(meta.strat_fold == 10)[0]
trn = np.where(meta.strat_fold <= 8)[0]
items = [(i, k) for i in te for k in range(len(SUPER)) if meta[SUPER[k]].values[i] == 1]  # same order as attribute.py
if args.limit:
    items = items[:args.limit]
items = pd.DataFrame(items, columns=["array_index", "label"])
items["ecg_id"] = meta.ecg_id.values[items.array_index]
n = len(items)
ref_idx = rng.choice(trn, size=(n, NS))  # GradientSHAP baselines, fixed per item so every model sees the same ones


def mid_module(m):
    net = m.net
    if m.name == "xresnet1d50":
        return net.body[12]          # last block of stage 3
    if m.name == "inception1d":
        return net.blocks[3]
    return net.encoder.layers[1]     # ConvTran1d: output of encoder layer 2 (tokens x d, batch first)


models = []
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    m = ECGNet(r["model"]).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=dev))
    nz = np.load(os.path.join(d, "norm.npz"))
    models.append((r, m, torch.tensor(nz["mean"], device=dev), torch.tensor(nz["std"], device=dev)))
print("items", n, "models", len(models), "device", dev, flush=True)

prof = {mt: np.lib.format.open_memmap(os.path.join(out, "profiles_%s.npy" % mt), "w+", np.float16, (n, len(models), 1000)) for mt in METHODS}
dele = {mt: np.zeros((n, len(models), 3), np.float32) for mt in METHODS}


def logit_of(m, x, lab):
    return m(x).gather(1, lab[:, None])[:, 0]


t0 = time.time()
for s in range(0, n, args.batch):
    b = items.iloc[s:s + args.batch]
    raw = torch.from_numpy(np.asarray(X[b.array_index.values])).to(dev)
    lab = torch.tensor(b.label.values, device=dev)
    for j, (r, m, mu, sd) in enumerate(models):
        x = ((raw - mu) / sd).float()
        with torch.no_grad():
            base = logit_of(m, x, lab)
        res = {}
        # SmoothGrad
        sig = 0.15 * (x.amax((1, 2), keepdim=True) - x.amin((1, 2), keepdim=True))
        acc = torch.zeros_like(x)
        for _ in range(NS):
            xn = (x + sig * torch.randn_like(x)).requires_grad_(True)
            g, = torch.autograd.grad(logit_of(m, xn, lab).sum(), xn)
            acc += g.abs() / NS
        res["smoothgrad"] = acc.sum(1)
        # GradientSHAP (expected gradients) with training-fold baselines
        acc = torch.zeros_like(x)
        for k in range(NS):
            xb = ((torch.from_numpy(np.asarray(X[ref_idx[s:s + len(b), k]])).to(dev) - mu) / sd).float()
            a = torch.rand(len(b), 1, 1, device=dev)
            xi = (xb + a * (x - xb)).requires_grad_(True)
            g, = torch.autograd.grad(logit_of(m, xi, lab).sum(), xi)
            acc += (x - xb) * g / NS
        res["gradshap"] = acc.clamp(min=0).sum(1)
        # Grad-CAM on an intermediate layer
        store = {}
        h = mid_module(m).register_forward_hook(lambda mod, inp, outp: store.__setitem__("a", outp))
        lg = logit_of(m, x, lab)
        h.remove()
        A = store["a"]
        g, = torch.autograd.grad(lg.sum(), A)  # gradient w.r.t. the hooked output itself (a later view would not be in the graph)
        if m.name == "convtran1d":
            A, g = A.transpose(1, 2), g.transpose(1, 2)  # [B, d, tokens]
        w = g.mean(2, keepdim=True)
        cam = (w * A).sum(1).clamp(min=0)
        res["gradcam_mid"] = F.interpolate(cam[:, None].detach(), size=1000, mode="linear", align_corners=False)[:, 0]
        with torch.no_grad():
            for mt, p in res.items():
                p = p.detach()
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
    print("done", min(s + args.batch, n), "of", n, "%.0f s" % (time.time() - t0), flush=True)

for mt in METHODS:
    prof[mt].flush()
masks, ok = [], []
for eid in items.ecg_id:
    mk = segment_masks(int(eid))
    ok.append(mk is not None)
    masks.append(mk if mk is not None else np.zeros((4, 1000), bool))
masks = np.stack(masks)
SEG = ["P", "QRS", "STT", "other"]
for mt in METHODS:
    P = np.asarray(prof[mt], np.float32)
    tot = P.sum(2) + 1e-12
    rows = dict(item=np.repeat(np.arange(n), len(models)), model_idx=np.tile(np.arange(len(models)), n))
    for si, sg in enumerate(SEG):
        rows["share_" + sg] = ((P * masks[:, None, si, :]).sum(2) / tot).ravel()
        rows["time_" + sg] = np.repeat(masks[:, si, :].mean(1), len(models))
    rows["fiducials_ok"] = np.repeat(np.array(ok), len(models))
    rows["zero_map"] = (tot.ravel() < 1e-9)
    pd.DataFrame(rows).to_csv(os.path.join(out, "local_%s.csv" % mt), index=False)
    d = dele[mt]
    pd.DataFrame(dict(item=np.repeat(np.arange(n), len(models)), model_idx=np.tile(np.arange(len(models)), n),
                      base_logit=d[:, :, 0].ravel(), drop_top10=d[:, :, 1].ravel(), drop_rand10=d[:, :, 2].ravel())).to_csv(
        os.path.join(out, "deletion_%s.csv" % mt), index=False)
json.dump(dict(items=n, models=reg.exp_id.tolist(), methods=METHODS, samples=NS, seconds=time.time() - t0),
          open(os.path.join(out, "info_extra.json"), "w"), indent=1)
print("ATTRIBUTE_EXTRA_DONE")
