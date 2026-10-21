"""Measured cost of each uncertainty method (segmentation-failure-detection), on the CPU of the login node (4 threads), with the stored best models.

For each encoder: parameters, time per image of one deterministic pass, of 10 MC-dropout passes, of 8 TTA views and of the
5-model ensemble, including the score computation (entropy, pairwise Dice) as in train.py. 24 Kvasir test images, batch 1,
3 warm-up images not timed. Writes results/statistics/efficiency.json. Usage: python efficiency.py <segmentation-failure-detection dir>
"""
import itertools
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MEAN, STD, UNet, dice, enable_mc_dropout  # noqa: E402

R = sys.argv[1]
torch.set_num_threads(4)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
X = np.load(os.path.join(R, "data", "processed", "kvasir_images.npy"), mmap_mode="r")
meta = pd.read_csv(os.path.join(R, "data", "processed", "kvasir_meta.csv"))
idx = np.where(meta.split.values == "test")[0][:27]
mean = torch.tensor(MEAN)[None, :, None, None]
std = torch.tensor(STD)[None, :, None, None]


def ent(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def pdice(b):
    return float(np.mean([dice(b[i], b[j]) for i, j in itertools.combinations(range(len(b)), 2)]))


out = dict(cpu=platform.processor() or platform.machine(), threads=4, torch=torch.__version__, images_timed=len(idx) - 3, encoders={})
for enc, g in reg[reg.data == "kvasir"].groupby("encoder"):
    models = []
    for e in g.sort_values("seed").exp_id:
        m = UNet(enc, pretrained=False)
        m.load_state_dict(torch.load(os.path.join(R, "experiments", e, "best_model.pt"), map_location="cpu"))
        m.eval()
        models.append(m)
    t = {k: [] for k in ["single", "mc10", "tta8", "ens5"]}
    with torch.no_grad():
        for n, j in enumerate(idx):
            x = (torch.from_numpy(np.asarray(X[j])).permute(2, 0, 1).float()[None] / 255 - mean) / std
            m0 = models[0]
            t0 = time.perf_counter()
            p = torch.sigmoid(m0(x))
            ent(p).mean().item()
            t1 = time.perf_counter()
            enable_mc_dropout(m0)
            ps = torch.stack([torch.sigmoid(m0(x)) for _ in range(10)])
            ent(ps.mean(0)).mean().item()
            pdice((ps[:, 0] > 0.5).numpy())
            m0.eval()
            t2 = time.perf_counter()
            vs = []
            for fl in range(8):
                xv = torch.rot90(x.flip(-1) if fl >= 4 else x, fl % 4, (2, 3))
                pv = torch.rot90(torch.sigmoid(m0(xv)), -(fl % 4), (1, 2))
                vs.append(pv.flip(-1) if fl >= 4 else pv)
            vs = torch.stack(vs)
            ent(vs.mean(0)).mean().item()
            pdice((vs[:, 0] > 0.5).numpy())
            t3 = time.perf_counter()
            pe = torch.stack([torch.sigmoid(m(x)) for m in models])
            ent(pe.mean(0)).mean().item()
            pdice((pe[:, 0] > 0.5).numpy())
            t4 = time.perf_counter()
            if n >= 3:
                for k, v in zip(t, [t1 - t0, t2 - t1, t3 - t2, t4 - t3]):
                    t[k].append(1000 * v)
    out["encoders"][enc] = dict(params_m=sum(q.numel() for q in models[0].parameters()) / 1e6,
                                **{k + "_ms": float(np.median(v)) for k, v in t.items()},
                                **{k + "_ms_iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] for k, v in t.items()})
    print(enc, json.dumps(out["encoders"][enc]), flush=True)
# training cost: the ensemble needs 5 trained models; median training time per run from results.json
for (data, enc), g in reg.groupby(["data", "encoder"]):
    secs = [json.load(open(os.path.join(R, "experiments", e, "results.json")))["train_seconds"] for e in g.exp_id]
    out.setdefault("train_minutes", {})["%s|%s" % (data, enc)] = float(np.median(secs) / 60)
json.dump(out, open(os.path.join(R, "results", "statistics", "efficiency.json"), "w"), indent=1)
print("EFFICIENCY_DONE")
