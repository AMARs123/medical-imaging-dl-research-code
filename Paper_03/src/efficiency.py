"""Measured cost for Paper 03 (CPU, same machine): classifiers and attribution methods.

Classifiers: parameters, forward FLOPs for one 12x1000 recording (torch FlopCounterMode), median CPU latency at batch 1
with 4 threads over 50 passes after 5 warm-up passes.
Attribution methods (XResNet1d-50, seed-0 model): median wall time per item over 3 repetitions on a batch of 32 test
items with 8 threads, and the number of network passes per item as defined in the paper.
Writes results/statistics/efficiency.json. Usage: python efficiency.py <Paper_03 dir>
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ECGNet  # noqa: E402

R = sys.argv[1]
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
out = {"arch": {}, "method": {}}
torch.manual_seed(0)
for a in ["xresnet1d50", "inception1d", "convtran1d"]:
    m = ECGNet(a).eval()
    x1 = torch.randn(1, 12, 1000)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x1)
    torch.set_num_threads(4)
    ts = []
    with torch.no_grad():
        for i in range(55):
            t = time.perf_counter()
            m(x1)
            if i >= 5:
                ts.append(time.perf_counter() - t)
    out["arch"][a] = dict(params_m=sum(p.numel() for p in m.parameters()) / 1e6, gflops=fc.get_total_flops() / 1e9, cpu_ms=1000 * float(np.median(ts)))
    print(a, out["arch"][a], flush=True)

r = reg[(reg.model == "xresnet1d50") & (reg.random_init == 0) & (reg.seed == 0)].iloc[0]
d = os.path.join(R, "experiments", r["exp_id"])
m = ECGNet("xresnet1d50").eval()
m.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location="cpu"))
nz = np.load(os.path.join(d, "norm.npz"))
X = np.load(os.path.join(R, "data", "processed", "signals.npy"), mmap_mode="r")
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
te = np.where(meta.strat_fold == 10)[0][:32]
x = ((torch.from_numpy(np.asarray(X[te])) - torch.tensor(nz["mean"])) / torch.tensor(nz["std"])).float()
lab = torch.zeros(len(te), dtype=torch.long)
torch.set_num_threads(8)


def logit(xx):
    return m(xx).gather(1, lab[:, None])[:, 0]


def grad(xx):
    xx = xx.clone().requires_grad_(True)
    g, = torch.autograd.grad(logit(xx).sum(), xx)
    return g


def ig():
    acc = torch.zeros_like(x)
    for a in (torch.arange(32) + 0.5) / 32:
        acc += grad(a * x) / 32
    return x * acc


def smooth():
    sig = 0.15 * (x.amax((1, 2), keepdim=True) - x.amin((1, 2), keepdim=True))
    return sum(grad(x + sig * torch.randn_like(x)).abs() for _ in range(16)) / 16


def shap():
    base = x[torch.randperm(len(x))]
    return sum((x - base) * grad(base + torch.rand(len(x), 1, 1) * (x - base)) for _ in range(16)) / 16


def occ():
    with torch.no_grad():
        b = logit(x)
        for t0 in range(0, 981, 10):
            xo = x.clone()
            xo[:, :, t0:t0 + 20] = 0
            logit(xo)


def cam():
    with torch.no_grad():
        F.interpolate(m.cam(x)[:, :1], size=1000, mode="linear", align_corners=False)


def gcam():
    store = {}
    h = m.net.body[12].register_forward_hook(lambda mod, i, o: store.__setitem__("a", o))
    lg = logit(x)
    h.remove()
    g, = torch.autograd.grad(lg.sum(), store["a"])
    (g.mean(2, keepdim=True) * store["a"]).sum(1)


FN = {"saliency": (lambda: grad(x), 1), "gxi": (lambda: grad(x) * x, 1), "ig": (ig, 32), "smoothgrad": (smooth, 16), "gradshap": (shap, 16),
      "occlusion": (occ, 100), "cam": (cam, 1), "gradcam_mid": (gcam, 1)}
for name, (fn, passes) in FN.items():
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    out["method"][name] = dict(passes=passes, cpu_ms_per_item=1000 * float(np.median(ts)) / len(te))
    print(name, out["method"][name], flush=True)
json.dump(out, open(os.path.join(R, "results", "statistics", "efficiency.json"), "w"), indent=1)
print("EFFICIENCY_DONE")
