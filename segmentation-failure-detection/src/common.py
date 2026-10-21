"""segmentation-failure-detection: 2-D binary segmentation with uncertainty (Kvasir-SEG polyps, HAM10000 skin lesions).

Model: timm encoder (features_only, ImageNet weights) + U-Net-style decoder (upsample, concatenate skip, two 3x3 conv
blocks) with spatial dropout p=0.2 before each decoder block, so that Monte-Carlo dropout can be switched on at test time.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Block(nn.Module):
    def __init__(self, ci, co, p):
        super().__init__()
        self.drop = nn.Dropout2d(p)
        self.c = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                               nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.c(self.drop(x))


class UNet(nn.Module):
    def __init__(self, encoder, p=0.2, pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained)
        ch = self.enc.feature_info.channels()  # e.g. [64, 64, 128, 256, 512] (resnet34) or [96, 192, 384, 768] (convnext)
        dec = [256, 128, 64, 32][:len(ch) - 1]
        self.blocks = nn.ModuleList()
        c = ch[-1]
        for i, d in enumerate(dec):
            self.blocks.append(Block(c + ch[-2 - i], d, p))
            c = d
        self.head = nn.Sequential(Block(c, 16, p), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        f = self.enc(x)
        y = f[-1]
        for i, b in enumerate(self.blocks):
            s = f[-2 - i]
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = b(torch.cat([y, s], 1))
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(y)[:, 0]


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


def dice(a, b, eps=1e-6):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / (s + eps))


class SegSet(torch.utils.data.Dataset):
    """images uint8 [N, S, S, 3], masks uint8 [N, S, S] (0/1). Training: flips, 90-degree rotations, scale/rotate, colour jitter."""

    def __init__(self, images, masks, index, train=False):
        self.images, self.masks, self.index, self.train = images, masks, index, train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        x = torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1).float() / 255.0
        m = torch.from_numpy(np.ascontiguousarray(self.masks[j])).float()[None]
        if self.train:
            if random.random() < 0.5:
                x, m = x.flip(-1), m.flip(-1)
            if random.random() < 0.5:
                x, m = x.flip(-2), m.flip(-2)
            k = random.randint(0, 3)
            x, m = torch.rot90(x, k, (1, 2)), torch.rot90(m, k, (1, 2))
            ang = np.deg2rad(random.uniform(-20, 20)); sc = random.uniform(0.85, 1.15)
            th = torch.tensor([[[np.cos(ang) / sc, -np.sin(ang) / sc, 0], [np.sin(ang) / sc, np.cos(ang) / sc, 0]]], dtype=torch.float32)
            g = F.affine_grid(th, (1, 1) + tuple(x.shape[-2:]), align_corners=False)
            x = F.grid_sample(x[None], g, align_corners=False, padding_mode="reflection")[0]
            m = (F.grid_sample(m[None], g, align_corners=False, mode="nearest", padding_mode="zeros")[0] > 0.5).float()
            x = x * random.uniform(0.85, 1.15)
            mu = x.mean(); x = ((x - mu) * random.uniform(0.85, 1.15) + mu).clamp(0, 1)
        x = (x - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]
        return x, m[0], j
