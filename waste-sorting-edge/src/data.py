"""Shared data pieces for waste-sorting-edge: classes, fixed grouped splits, data set with augmentation."""
import random

import numpy as np
import torch

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def splits(meta, seed=0):
    """70/15/15 per source, stratified by class over photographs (groups), fixed for all runs."""
    rng = np.random.default_rng(seed)
    part = np.empty(len(meta), object)
    for (s, c), g in meta.groupby(["source", "label"]):
        grp = g.group.unique()
        grp = grp[rng.permutation(len(grp))]
        n = len(grp)
        lab = {u: ("train" if k < 0.7 * n else "val" if k < 0.85 * n else "test") for k, u in enumerate(grp)}
        part[g.index.values] = g.group.map(lab).values
    # a TACO photo can hold objects of several classes; give the whole photo the split of its first object
    first = {}
    for i, gname in enumerate(meta.group.values):
        first.setdefault(gname, part[i])
    return np.array([first[gname] for gname in meta.group.values])


class ImgSet(torch.utils.data.Dataset):
    def __init__(self, images, labels, index, train=False):
        self.images, self.labels, self.index, self.train = images, labels, index, train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        x = torch.from_numpy(np.ascontiguousarray(self.images[j])).permute(2, 0, 1).float() / 255.0
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
            if random.random() < 0.5:
                x = x.flip(-2)
            x = torch.rot90(x, random.randint(0, 3), (1, 2))
            s = random.randint(176, 224); t, l = random.randint(0, 224 - s), random.randint(0, 224 - s)
            x = torch.nn.functional.interpolate(x[None, :, t:t + s, l:l + s], size=224, mode="bilinear", align_corners=False)[0]
            x = x * random.uniform(0.8, 1.2); m = x.mean(); x = ((x - m) * random.uniform(0.8, 1.2) + m).clamp(0, 1)
        return (x - MEAN) / STD, int(self.labels[j])
