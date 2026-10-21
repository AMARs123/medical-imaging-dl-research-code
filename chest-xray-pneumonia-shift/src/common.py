"""Shared pieces for chest-xray-pneumonia-shift: splits, data set with augmentation variants, models, metrics."""
import random

import numpy as np
import torch
import torch.nn.functional as F

MODELS = {"densenet121.tv_in1k": "DenseNet-121", "resnet50.a1_in1k": "ResNet-50", "convnext_tiny.fb_in1k": "ConvNeXt-T"}
GROUPS = {"G": "Guangzhou (children)", "NA": "NIH (adults)", "NP": "NIH (children)"}
MEAN, STD = 0.5, 0.25


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def folds_for(meta, groups, seed=0, n=5):
    """Patient-grouped, label-stratified fold number (0..4) for every image of the given groups (-1 elsewhere).
    Folds are formed separately inside each group, so every fold contains each training group in the same proportion."""
    from sklearn.model_selection import StratifiedGroupKFold
    a = np.full(len(meta), -1)
    for g in groups:
        idx = np.where((meta.group == g).values & meta.ok.values)[0]
        y, p = meta.label.values[idx], meta.patient.values[idx]
        for k, (_, te) in enumerate(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(y, y, p)):
            a[idx[te]] = k
    return a


class CXRSet(torch.utils.data.Dataset):
    """aug = 'none' | 'base' | 'strong'.
    base  : random resized crop (80-100% area, aspect 0.9-1.1), rotation +-7 deg, brightness/contrast +-10%.
            No horizontal flip: it would move the heart to the wrong side.
    strong: base + gamma 0.7-1.4, contrast 0.6-1.4, Gaussian blur (sigma 0-1.5 px, p=0.5), Gaussian noise
            (sigma 0-0.04, p=0.5) and random down- and up-sampling to 96-224 px (p=0.3): simulates acquisition differences."""

    def __init__(self, images, labels, index, aug="none", size=224):
        self.images, self.labels, self.index, self.aug, self.size = images, labels, index, aug, size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        x = torch.from_numpy(np.asarray(self.images[j], np.float32) / 255.0)[None, None]  # [1,1,H,W]
        if self.aug in ("base", "strong"):
            ang = np.deg2rad(random.uniform(-7, 7))
            sc = np.sqrt(random.uniform(0.8, 1.0))
            ar = random.uniform(0.9, 1.1)
            sx, sy = sc * np.sqrt(ar), sc / np.sqrt(ar)
            tx, ty = random.uniform(-(1 - sx), 1 - sx), random.uniform(-(1 - sy), 1 - sy)
            th = torch.tensor([[[np.cos(ang) * sx, -np.sin(ang) * sy, tx], [np.sin(ang) * sx, np.cos(ang) * sy, ty]]], dtype=torch.float32)
            x = F.grid_sample(x, F.affine_grid(th, (1, 1, self.size, self.size), align_corners=False), align_corners=False)
            x = x * random.uniform(0.9, 1.1)
            m = x.mean(); x = (x - m) * random.uniform(0.9, 1.1) + m
            if self.aug == "strong":
                x = x.clamp(0, 1) ** random.uniform(0.7, 1.4)
                m = x.mean(); x = (x - m) * random.uniform(0.6, 1.4) + m
                if random.random() < 0.5:
                    s = random.uniform(0.1, 1.5)
                    k = torch.arange(-4, 5, dtype=torch.float32)
                    g = torch.exp(-k ** 2 / (2 * s ** 2)); g = g / g.sum()
                    x = F.conv2d(F.pad(x, (4, 4, 0, 0), mode="reflect"), g.view(1, 1, 1, 9))
                    x = F.conv2d(F.pad(x, (0, 0, 4, 4), mode="reflect"), g.view(1, 1, 9, 1))
                if random.random() < 0.5:
                    x = x + torch.randn_like(x) * random.uniform(0, 0.04)
                if random.random() < 0.3:
                    r = random.randint(96, 224)
                    x = F.interpolate(F.interpolate(x, size=r, mode="bilinear", antialias=True, align_corners=False), size=self.size,
                                      mode="bilinear", align_corners=False)
            x = x.clamp(0, 1)
        else:
            x = F.interpolate(x, size=self.size, mode="bilinear", antialias=True, align_corners=False)
        x = ((x[0] - MEAN) / STD).repeat(3, 1, 1)
        return x, int(self.labels[j]), j


def build_model(name, pretrained=True):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


def binary_metrics(y, p, thr=0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    out = dict(n=int(len(y)), n_pos=int(y.sum()))
    out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    out["ap"] = float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    pred = p >= thr
    out["sens"] = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    out["spec"] = float((~pred)[y == 0].mean()) if (y == 0).any() else float("nan")
    out["bacc"] = float(np.nanmean([out["sens"], out["spec"]]))
    out["brier"] = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (p > lo) & (p <= hi)
        if s.any():
            ece += s.mean() * abs(y[s].mean() - p[s].mean())
    out["ece"] = float(ece)
    return out
