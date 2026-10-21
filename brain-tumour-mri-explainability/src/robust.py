"""Test-time robustness of finished runs to image corruptions (no training).

Usage: python robust.py <brain-tumour-mri-explainability dir> [--device cuda]
Runs: ID-root folds, split seed 0, default dilation and data, the four phase-C architectures, lambda 0 and 0.25
(5 folds x 3 seeds each). Writes robust.json into each run directory (skipped if present).
Corruptions act on the [0, 1] image before normalisation and are clipped to [0, 1]:
  noise_0.05 / noise_0.10  additive Gaussian noise (SD as share of the intensity range)
  blur_1 / blur_2          Gaussian blur, sigma 1 or 2 pixels
  contrast_0.6             contrast scaled by 0.6 around the image mean
  gamma_0.7 / gamma_1.5    intensity gamma
  lowres_0.5               down-sampled to half resolution and back (bilinear)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import MEAN, STD, EvidenceNet, TumourSet, load_arrays

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--limit", type=int, default=0, help="smoke test: first run only, this many images, nothing written")
args = ap.parse_args()
images, masks, meta = load_arrays(os.path.join(args.root, "data", "processed"))
reg = pd.read_csv(os.path.join(args.root, "experiments", "experiment_registry.csv"))
for c, v in [("split_seed", 0), ("dilate", 1), ("train_frac", 1.0)]:
    if c not in reg:
        reg[c] = v
sel = reg[(reg.protocol == "root") & (reg.split_seed.fillna(0) == 0) & (reg.dilate.fillna(1) == 1) & (reg.train_frac.fillna(1) == 1)
          & reg.model.isin(["resnet50", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"])
          & reg.lambda_sal.isin([0.0, 0.25]) & reg.phase.isin(["A_protocol", "B_baseline", "C_saliency"])]


def gblur(x, s):
    k = int(3 * s) * 2 + 1
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


def corrupt(x01, name, gen):
    if name == "clean":
        return x01
    kind, val = name.split("_")
    v = float(val)
    if kind == "noise":
        y = x01 + v * torch.randn(x01.shape, generator=gen, device="cpu").to(x01.device)
    elif kind == "blur":
        y = gblur(x01, v)
    elif kind == "contrast":
        m = x01.mean((1, 2, 3), keepdim=True)
        y = m + v * (x01 - m)
    elif kind == "gamma":
        y = x01.clamp(min=0) ** v
    elif kind == "lowres":
        s = x01.shape[-1]
        y = F.interpolate(F.interpolate(x01, scale_factor=v, mode="bilinear", align_corners=False), size=(s, s), mode="bilinear", align_corners=False)
    return y.clamp(0, 1)


CORR = ["clean", "noise_0.05", "noise_0.10", "blur_1", "blur_2", "contrast_0.6", "gamma_0.7", "gamma_1.5", "lowres_0.5"]
for e in sel.exp_id:
    d = os.path.join(args.root, "experiments", e)
    if not os.path.exists(os.path.join(d, "results.json")) or os.path.exists(os.path.join(d, "robust.json")):
        continue
    cfg = json.load(open(os.path.join(d, "results.json")))["config"]
    te = np.load(os.path.join(d, "split_indices.npz"))["test"]
    if args.limit:
        te = te[:args.limit]
    model = EvidenceNet(cfg["model"], pretrained=False, img_size=cfg["img_size"]).to(args.device)
    model.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=args.device))
    model.eval()
    dl = torch.utils.data.DataLoader(TumourSet(images, masks, meta, te, size=cfg["img_size"]), batch_size=64)
    preds, ys = {c: [] for c in CORR}, []
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for x, y, m, j in dl:
            x01 = (x[:, :1] * STD + MEAN).to(args.device)
            ys.append(y.numpy())
            for c in CORR:
                xi = ((corrupt(x01, c, gen) - MEAN) / STD).repeat(1, 3, 1, 1)
                preds[c].append(model(xi).argmax(1).cpu().numpy())
    y = np.concatenate(ys)
    out = {c: dict(macro_f1=float(f1_score(y, np.concatenate(p), average="macro")), accuracy=float((np.concatenate(p) == y).mean()))
           for c, p in preds.items()}
    if args.limit:
        print("smoke", e, {c: round(v["accuracy"], 3) for c, v in out.items()})
        break
    json.dump(out, open(os.path.join(d, "robust.json"), "w"), indent=1)
    print(e, cfg["model"], cfg["lambda_sal"], " ".join("%s %.3f" % (c, v["macro_f1"]) for c, v in out.items()), flush=True)
