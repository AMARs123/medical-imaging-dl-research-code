"""Shared pieces for brain-tumour-mri-explainability: data, splits, models with an exact class-evidence map, metrics."""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["meningioma", "glioma", "pituitary"]  # dataset labels 1, 2, 3
MEAN, STD = 0.5, 0.25  # single-channel statistics applied after scaling to [0, 1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- splits
def make_split(meta, protocol, fold, seed, n_folds=5):
    """Return train / val / test index arrays.

    protocol = "root":    images sharing an ID root (the PID with trailing letters removed) stay in one fold.
                          Most conservative grouping; used for all main experiments.
    protocol = "pid":     images sharing the exact PID string stay in one fold (the usual patient-level split).
    protocol = "slice":   images are assigned to folds ignoring identity (StratifiedKFold). Reference only.
    protocol = "authors": the dataset authors' own 5-fold index (column `fold`, values 1..5).
    Fold `fold` is the test fold, the next fold is validation, the rest is training.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = meta["label"].values
    if protocol == "authors":
        assign = meta["fold"].values.astype(int) - 1
    elif protocol in ("pid", "root"):
        assign = np.zeros(len(meta), int)
        for k, (_, te) in enumerate(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(y, y, meta[protocol].values)):
            assign[te] = k
    elif protocol == "slice":
        assign = np.zeros(len(meta), int)
        for k, (_, te) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(y, y)):
            assign[te] = k
    else:
        raise ValueError(protocol)
    test = np.where(assign == fold)[0]
    val = np.where(assign == (fold + 1) % n_folds)[0]
    train = np.where((assign != fold) & (assign != (fold + 1) % n_folds))[0]
    return train, val, test


def patient_overlap(meta, a, b, col="pid"):
    return len(set(meta[col].values[a]) & set(meta[col].values[b]))


# --------------------------------------------------------------------------- data
class TumourSet(torch.utils.data.Dataset):
    def __init__(self, images, masks, meta, index, size=224, train=False):
        self.images, self.masks, self.meta, self.index, self.size, self.train = images, masks, meta, index, size, train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        x = torch.from_numpy(self.images[j].astype(np.float32) / 255.0)[None]
        m = torch.from_numpy(self.masks[j].astype(np.float32))[None]
        if self.train:  # identical geometric transform for image and mask
            if random.random() < 0.5:
                x, m = x.flip(-1), m.flip(-1)
            ang = random.uniform(-15, 15)
            sc = random.uniform(0.9, 1.1)
            tx, ty = random.uniform(-0.06, 0.06), random.uniform(-0.06, 0.06)
            c, s = np.cos(np.deg2rad(ang)) / sc, np.sin(np.deg2rad(ang)) / sc
            theta = torch.tensor([[c, -s, tx], [s, c, ty]], dtype=torch.float32)[None]
            grid = F.affine_grid(theta, (1, 1, self.size, self.size), align_corners=False)
            x = F.grid_sample(x[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
            m = F.grid_sample(m[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
            x = (x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        else:
            x = F.interpolate(x[None], size=self.size, mode="bilinear", align_corners=False)[0]
            m = F.interpolate(m[None], size=self.size, mode="nearest")[0]
        x = ((x - MEAN) / STD).repeat(3, 1, 1)
        return x, int(self.meta["label"].values[j]), m, j


def load_arrays(proc_dir):
    images = np.load(os.path.join(proc_dir, "images.npy"), mmap_mode="r")
    masks = np.load(os.path.join(proc_dir, "masks.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(proc_dir, "meta.csv"), dtype={"pid": str, "root": str})
    return images, masks, meta


# --------------------------------------------------------------------------- model
class EvidenceNet(nn.Module):
    """timm backbone + (global average pool -> linear) head.

    Because the head is linear and applied after average pooling of the final feature map,
    the logit of class c decomposes exactly as  logit_c = mean_{h,w} E_c(h, w) + b_c  with
    E_c = W_c . F(h, w).  E is the class-evidence map (the classic CAM), available in one
    forward pass and differentiable, for CNNs and for transformers alike.
    """

    def __init__(self, name, n_classes=3, pretrained=True, img_size=224):
        super().__init__()
        import timm

        kw = dict(pretrained=pretrained, num_classes=0)
        if name.startswith(("vit", "deit")):
            # final LayerNorm is kept before pooling so that pooling and the linear head commute
            kw.update(global_pool="avg", fc_norm=False, img_size=img_size)
        elif name.startswith("swin"):
            kw.update(img_size=img_size)
        self.name = name
        self.backbone = timm.create_model(name, **kw)
        self.kind = "vit" if name.startswith(("vit", "deit")) else ("swin" if name.startswith("swin") else "cnn")
        self.head = nn.Linear(self.backbone.num_features, n_classes)

    def features(self, x):
        f = self.backbone.forward_features(x)
        if self.kind == "vit":  # [B, prefix + N, C] -> [B, C, h, w]
            n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            f = f[:, n_prefix:]
            h = int(round(f.shape[1] ** 0.5))
            f = f.transpose(1, 2).reshape(f.shape[0], f.shape[2], h, h)
        elif self.kind == "swin" and f.ndim == 4 and f.shape[-1] == self.backbone.num_features:
            f = f.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return f

    def forward(self, x, return_evidence=False):
        f = self.features(x)
        logits = self.head(f.mean((2, 3)))
        if not return_evidence:
            return logits
        evidence = torch.einsum("kc,bchw->bkhw", self.head.weight, f)
        return logits, evidence


def saliency_loss(evidence, y, mask, dilate=1):
    """Share of positive true-class evidence that falls outside the (downsampled, dilated) tumour mask."""
    e = F.relu(evidence[torch.arange(len(y), device=y.device), y])  # [B, h, w]
    m = F.adaptive_max_pool2d(mask, e.shape[-2:])  # a cell counts as tumour if any pixel in it is
    if dilate:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    m = m[:, 0]
    has = (m.flatten(1).sum(1) > 0).float()
    outside = (e * (1 - m)).flatten(1).sum(1) / (e.flatten(1).sum(1) + 1e-6)
    return (outside * has).sum() / has.sum().clamp(min=1)


# --------------------------------------------------------------------------- metrics
def classification_metrics(y, prob):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

    pred = prob.argmax(1)
    out = dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro")),
        per_class_f1=[float(v) for v in f1_score(y, pred, average=None, labels=[0, 1, 2])],
        confusion=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        ece=float(expected_calibration_error(y, prob)),
        nll=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean()),
    )
    try:
        out["macro_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out


def expected_calibration_error(y, prob, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    ece = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        sel = (conf > lo) & (conf <= lo + 1.0 / bins)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return ece


def localisation_metrics(heat, mask):
    """heat: [h, w] non-negative map already resized to the mask; mask: [h, w] in {0, 1}.

    pointing   : 1 if the maximum of the map lies inside the mask.
    mass_inside: share of the map's total mass inside the mask (relevance-mass accuracy).
    iou_area   : IoU between the mask and the top-k pixels of the map, k = number of mask pixels
                 (threshold-free, so maps with different value ranges are comparable).
    """
    heat = np.maximum(heat, 0)
    n = int(mask.sum())
    if n == 0:
        return dict(pointing=np.nan, mass_inside=np.nan, iou_area=np.nan)
    flat = heat.ravel()
    point = float(mask.ravel()[flat.argmax()] > 0)
    total = flat.sum()
    mass = float((heat * mask).sum() / total) if total > 0 else 0.0
    top = np.zeros_like(flat, dtype=bool)
    top[np.argsort(-flat, kind="stable")[:n]] = True
    inter = (top & (mask.ravel() > 0)).sum()
    return dict(pointing=point, mass_inside=mass, iou_area=float(inter / (2 * n - inter)))
