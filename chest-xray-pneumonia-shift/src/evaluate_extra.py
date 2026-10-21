"""Post-training evaluations of every finished chest-xray-pneumonia-shift run (no retraining). Usage: python evaluate_extra.py <chest-xray-pneumonia-shift dir> [--phase A]

1. Test-time BatchNorm re-estimation (AdaBN; Li et al. 2016): for each group that was NOT used for training, the BatchNorm
   running statistics are re-estimated on that group's images (labels unused; cumulative average over one pass), then the
   group is predicted again. Only for backbones with BatchNorm layers (DenseNet-121, ResNet-50); ConvNeXt uses LayerNorm.
2. Corruption robustness (phase A runs only): the in-domain test fold and the external groups are evaluated under six
   corruptions at five severities (Hendrycks & Dietterich 2019 style, adapted to grey radiographs):
     noise  : additive Gaussian noise, sigma 0.02, 0.04, 0.06, 0.08, 0.10 (image in [0, 1])
     blur   : Gaussian blur, sigma 0.5, 1.0, 1.5, 2.0, 2.5 px at 224 px
     contrast: x -> mean + c (x - mean), c = 0.8, 0.65, 0.5, 0.35, 0.2
     gamma  : x -> x^g, g = 0.6, 0.75, 1.33, 1.67, 2.0 (both darker and brighter exposures)
     jpeg   : JPEG quality 50, 30, 20, 12, 8
     lowres : down-sampling to 160, 128, 96, 72, 56 px and back to 224
Writes experiments/<EXP>/adabn.json and robust.json.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MEAN, STD, binary_metrics, build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--phase", default="")
args = ap.parse_args()
R = args.root
dev = "cuda" if torch.cuda.is_available() else "cpu"
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"), keep_default_na=False, na_values=[""])
arr = {"raw": np.load(os.path.join(R, "data", "processed", "images.npy"), mmap_mode="r"),
       "clahe": np.load(os.path.join(R, "data", "processed", "images_clahe.npy"), mmap_mode="r")}
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"), keep_default_na=False, na_values=[""])
y_all = meta.label.values
SEV = {"noise": [0.02, 0.04, 0.06, 0.08, 0.10], "blur": [0.5, 1.0, 1.5, 2.0, 2.5], "contrast": [0.8, 0.65, 0.5, 0.35, 0.2],
       "gamma": [0.6, 0.75, 1.33, 1.67, 2.0], "jpeg": [50, 30, 20, 12, 8], "lowres": [160, 128, 96, 72, 56]}


def batch(images, idx):
    x = torch.from_numpy(np.stack([np.asarray(images[i], np.float32) / 255.0 for i in idx]))[:, None]
    return F.interpolate(x, size=224, mode="bilinear", antialias=True, align_corners=False)


def corrupt(x, kind, s, rng):
    if kind == "noise":
        return (x + torch.from_numpy(rng.normal(0, s, x.shape).astype(np.float32))).clamp(0, 1)
    if kind == "blur":
        k = torch.arange(-6, 7, dtype=torch.float32); g = torch.exp(-k ** 2 / (2 * s ** 2)); g = (g / g.sum())
        x = F.conv2d(F.pad(x, (6, 6, 0, 0), mode="reflect"), g.view(1, 1, 1, 13))
        return F.conv2d(F.pad(x, (0, 0, 6, 6), mode="reflect"), g.view(1, 1, 13, 1))
    if kind == "contrast":
        m = x.mean((2, 3), keepdim=True); return (m + s * (x - m)).clamp(0, 1)
    if kind == "gamma":
        return x.clamp(0, 1) ** s
    if kind == "jpeg":
        out = []
        for im in x[:, 0].numpy():
            b = io.BytesIO(); Image.fromarray((im * 255).round().astype(np.uint8)).save(b, "JPEG", quality=int(s))
            out.append(np.asarray(Image.open(io.BytesIO(b.getvalue())), np.float32) / 255.0)
        return torch.from_numpy(np.stack(out))[:, None]
    if kind == "lowres":
        return F.interpolate(F.interpolate(x, size=int(s), mode="bilinear", antialias=True, align_corners=False), size=224,
                             mode="bilinear", align_corners=False)
    raise ValueError(kind)


@torch.no_grad()
def predict(model, images, idx, kind=None, s=None, bs=64):
    model.eval(); rng = np.random.default_rng(0); out = []
    for b in range(0, len(idx), bs):
        x = batch(images, idx[b:b + bs])
        if kind:
            x = corrupt(x, kind, s, rng)
        x = ((x - MEAN) / STD).repeat(1, 3, 1, 1).to(dev)
        out.append(torch.sigmoid(model(x).float()[:, 0]).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def adabn(model, images, idx, bs=64):
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None  # cumulative moving average over the pass
    model.train()
    for b in range(0, len(idx), bs):
        x = ((batch(images, idx[b:b + bs]) - MEAN) / STD).repeat(1, 3, 1, 1).to(dev)
        model(x)
    model.eval()


for r in reg.to_dict("records"):
    if args.phase and r["phase"] != args.phase:
        continue
    d = os.path.join(R, "experiments", r["exp_id"])
    if not os.path.exists(os.path.join(d, "results.json")):
        continue
    res = json.load(open(os.path.join(d, "results.json")))
    thr = res["threshold"]
    images = arr[r["prep"]]
    pred = pd.read_csv(os.path.join(d, "predictions.csv"), keep_default_na=False, na_values=[""])
    tg = r["train_groups"].split(",")
    wp = os.path.join(d, "best_model.pt")
    if not os.path.exists(wp) and os.environ.get("WEIGHTS_DIR"):  # weights kept outside /home (17 Oct 2026)
        wp = os.path.join(os.environ["WEIGHTS_DIR"], os.path.basename(os.path.normpath(d)), "best_model.pt")
    state = torch.load(wp, map_location=dev)
    model = build_model(r["model"], pretrained=False).to(dev)
    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if has_bn and not os.path.exists(os.path.join(d, "adabn.json")):
        out = {}
        for g in [g for g in ["G", "NA", "NP"] if g not in tg]:
            idx = pred[(pred.group == g) & (pred.part == "test_out")].array_index.values
            model.load_state_dict(state); adabn(model, images, idx)
            out[g] = binary_metrics(y_all[idx], predict(model, images, idx), thr)
        json.dump(out, open(os.path.join(d, "adabn.json"), "w"), indent=1)
    if r["phase"] == "A" and not os.path.exists(os.path.join(d, "robust.json")):
        model.load_state_dict(state)
        out = {}
        for g in ["G", "NA", "NP"]:
            q = pred[(pred.group == g) & (pred.part != "val")]
            idx = q.array_index.values
            out[g] = {"clean": binary_metrics(y_all[idx], predict(model, images, idx), thr)}
            for kind, sevs in SEV.items():
                out[g][kind] = [binary_metrics(y_all[idx], predict(model, images, idx, kind, s), thr) for s in sevs]
        json.dump(out, open(os.path.join(d, "robust.json"), "w"), indent=1)
    print(r["exp_id"], "done", flush=True)
print("EVAL_EXTRA_DONE")
