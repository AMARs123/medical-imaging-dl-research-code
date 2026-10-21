"""Shared pieces for Paper 02: classes, splits, data set, models, losses, metrics."""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# order of the ISIC 2018 Task 3 challenge
CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
NAMES = {"mel": "Melanoma", "nv": "Melanocytic naevus", "bcc": "Basal cell carcinoma", "akiec": "Actinic keratosis / IEC",
         "bkl": "Benign keratosis", "df": "Dermatofibroma", "vasc": "Vascular lesion"}
K = len(CLASSES)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet statistics (all backbones are ImageNet-pretrained)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# model families; the names are timm identifiers with the pretrained tag
MODELS = {
    "resnet50.a1_in1k": "CNN",
    "efficientnet_b0.ra_in1k": "CNN",
    "convnext_tiny.fb_in1k": "CNN",
    "deit_small_patch16_224.fb_in1k": "Transformer",
    "deit3_small_patch16_224.fb_in1k": "Transformer",
    "swin_tiny_patch4_window7_224.ms_in1k": "Transformer",
    "coatnet_0_rw_224.sw_in1k": "Hybrid",
    "maxvit_tiny_tf_224.in1k": "Hybrid",
    "mobilevitv2_100.cvnets_in1k": "Hybrid",
}
SHORT = {
    "resnet50.a1_in1k": "ResNet-50", "efficientnet_b0.ra_in1k": "EfficientNet-B0", "convnext_tiny.fb_in1k": "ConvNeXt-T",
    "deit_small_patch16_224.fb_in1k": "DeiT-S", "deit3_small_patch16_224.fb_in1k": "DeiT-III-S",
    "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T", "coatnet_0_rw_224.sw_in1k": "CoAtNet-0",
    "maxvit_tiny_tf_224.in1k": "MaxViT-T", "mobilevitv2_100.cvnets_in1k": "MobileViTv2-1.0",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- splits
def fold_assignment(meta, protocol, seed, n_folds=5):
    """Fold number (0..n_folds-1) of every HAM10000 image.

    protocol = "lesion": all images of one lesion_id stay in one fold (StratifiedGroupKFold).
    protocol = "image":  images are assigned to folds ignoring lesion_id (StratifiedKFold),
                         the split used by many published HAM10000 studies. Reference only.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = meta["label"].values
    assign = np.zeros(len(meta), int)
    if protocol == "lesion":
        it = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(y, y, meta["lesion_id"].values)
    elif protocol == "image":
        it = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(y, y)
    else:
        raise ValueError(protocol)
    for k, (_, te) in enumerate(it):
        assign[te] = k
    return assign


def make_split(meta, protocol, fold, seed=0, n_folds=5):
    """Fold `fold` is the test fold, the next fold is validation, the remaining three are training."""
    a = fold_assignment(meta, protocol, seed, n_folds)
    test = np.where(a == fold)[0]
    val = np.where(a == (fold + 1) % n_folds)[0]
    train = np.where((a != fold) & (a != (fold + 1) % n_folds))[0]
    return train, val, test


def lesion_overlap(meta, a, b):
    return len(set(meta["lesion_id"].values[a]) & set(meta["lesion_id"].values[b]))


# --------------------------------------------------------------------------- data
class SkinSet(torch.utils.data.Dataset):
    """Images are stored as uint8 arrays of 256 x 256 x 3 (see preprocess.py)."""

    def __init__(self, images, labels, index, size=224, train=False):
        self.images, self.labels, self.index, self.size, self.train = images, labels, index, size, train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        x = torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        if self.train:
            # dermoscopy has no canonical orientation: flips and 90-degree turns are label-preserving
            if random.random() < 0.5:
                x = x.flip(-1)
            if random.random() < 0.5:
                x = x.flip(-2)
            x = torch.rot90(x, random.randint(0, 3), (1, 2))
            # random resized crop (scale 0.6-1.0 of the area, aspect 3/4-4/3)
            H, W = x.shape[1:]
            for _ in range(10):
                area = random.uniform(0.6, 1.0) * H * W
                ar = np.exp(random.uniform(np.log(3 / 4), np.log(4 / 3)))
                h, w = int(round((area / ar) ** 0.5)), int(round((area * ar) ** 0.5))
                if h <= H and w <= W:
                    t, l = random.randint(0, H - h), random.randint(0, W - w)
                    x = x[:, t:t + h, l:l + w]
                    break
            x = F.interpolate(x[None], size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)[0]
            # mild colour jitter: brightness, contrast, saturation
            x = x * random.uniform(0.85, 1.15)
            mu = x.mean()
            x = (x - mu) * random.uniform(0.85, 1.15) + mu
            g = x.mean(0, keepdim=True)
            x = (x - g) * random.uniform(0.85, 1.15) + g
            x = x.clamp(0, 1)
        else:
            x = F.interpolate(x[None], size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)[0]
        x = (x - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]
        return x, int(self.labels[j]), j


def load_arrays(proc_dir, part="ham"):
    images = np.load(os.path.join(proc_dir, "%s_images.npy" % part), mmap_mode="r")
    meta = pd.read_csv(os.path.join(proc_dir, "%s_meta.csv" % part))
    return images, meta


def build_model(name, pretrained=True):
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=K)


# --------------------------------------------------------------------------- losses
def loss_fn(kind, logits, y, prior, tau=1.0, gamma=2.0):
    """prior: training-split class frequencies (tensor, sums to 1).

    ce      : plain cross-entropy.
    wce     : cross-entropy weighted by inverse class frequency (weights normalised to mean 1 over samples).
    focal   : focal loss (Lin et al.), gamma = 2.
    la      : logit-adjusted cross-entropy (Menon et al.): add tau * log(prior) to the logits during training only.
    sampler : plain cross-entropy, but the training loader draws classes uniformly (handled by the loader).
    """
    if kind in ("ce", "sampler"):
        return F.cross_entropy(logits, y)
    if kind == "wce":
        w = 1.0 / (prior * K)
        return F.cross_entropy(logits, y, weight=w)
    if kind == "focal":
        logp = F.log_softmax(logits, 1).gather(1, y[:, None])[:, 0]
        return (-(1 - logp.exp()) ** gamma * logp).mean()
    if kind == "la":
        return F.cross_entropy(logits + tau * torch.log(prior)[None], y)
    raise ValueError(kind)


# --------------------------------------------------------------------------- metrics
def classification_metrics(y, prob):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                                 matthews_corrcoef, recall_score, roc_auc_score)

    pred = prob.argmax(1)
    labels = list(range(K))
    out = dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),  # mean per-class recall = ISIC 2018 Task 3 metric
        macro_f1=float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)),
        kappa=float(cohen_kappa_score(y, pred)),
        per_class_recall=[float(v) for v in recall_score(y, pred, average=None, labels=labels, zero_division=0)],
        per_class_f1=[float(v) for v in f1_score(y, pred, average=None, labels=labels, zero_division=0)],
        confusion=confusion_matrix(y, pred, labels=labels).tolist(),
        ece=float(expected_calibration_error(y, prob)),
        nll=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean()),
    )
    try:
        out["macro_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        out["macro_auc"] = float("nan")
    try:
        out["mel_auc"] = float(roc_auc_score((y == 0).astype(int), prob[:, 0]))
    except ValueError:
        out["mel_auc"] = float("nan")
    return out


def expected_calibration_error(y, prob, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    ece = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        sel = (conf > lo) & (conf <= lo + 1.0 / bins)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return ece
