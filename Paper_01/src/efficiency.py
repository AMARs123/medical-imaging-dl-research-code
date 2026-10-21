"""Measured cost of the eight evidence networks on the same hardware (Research_Expert rule E-13).

Writes results/processed/efficiency.csv with, per architecture:
  params_m         parameters (millions), including the linear evidence head
  gflops           forward FLOPs for one 224x224 image (torch FlopCounterMode, multiply-add = 2 FLOPs)
  gpu_ms_b1        median GPU latency, batch 1, AMP, over 100 timed passes after 20 warm-up passes
  gpu_ms_b32_per   median GPU latency per image at batch 32
  gpu_peak_mb_b32  peak allocated GPU memory for inference at batch 32
  cpu_ms_b1_t4     median CPU latency, batch 1, 4 threads, fp32, over 30 passes after 5 warm-up passes
  evidence_ms_b1   extra GPU time of returning the evidence maps at batch 1 (with minus without)
Usage: python efficiency.py <Paper_01 dir> [--device cuda]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EvidenceNet  # noqa: E402

root = sys.argv[1]
MODELS = ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilevit_s", "vit_small_patch16_224",
          "deit_small_patch16_224", "swin_tiny_patch4_window7_224"]


def timed(fn, n, warm, cuda):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


rows = []
for name in MODELS:
    torch.manual_seed(0)
    m = EvidenceNet(name, 3, pretrained=False).eval()
    r = dict(model=name, params_m=sum(p.numel() for p in m.parameters()) / 1e6)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(torch.randn(1, 3, 224, 224))
    r["gflops"] = fc.get_total_flops() / 1e9
    torch.set_num_threads(4)
    x1 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        r["cpu_ms_b1_t4"] = timed(lambda: m(x1), 30, 5, False)
    if torch.cuda.is_available():
        g = m.cuda()
        xb1, xb32 = x1.cuda(), torch.randn(32, 3, 224, 224, device="cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            r["gpu_ms_b1"] = timed(lambda: g(xb1), 100, 20, True)
            r["evidence_ms_b1"] = timed(lambda: g(xb1, return_evidence=True), 100, 20, True) - r["gpu_ms_b1"]
            torch.cuda.reset_peak_memory_stats()
            r["gpu_ms_b32_per"] = timed(lambda: g(xb32), 30, 5, True) / 32
            r["gpu_peak_mb_b32"] = torch.cuda.max_memory_allocated() / 2**20
        del g
        torch.cuda.empty_cache()
    rows.append(r)
    print(r, flush=True)
out = os.path.join(root, "results", "processed", "efficiency.csv")
pd.DataFrame(rows).to_csv(out, index=False)
env = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
           cpu=os.popen("lscpu | grep 'Model name'").read().strip())
open(out.replace(".csv", "_environment.txt"), "w").write(repr(env) + "\n")
print("wrote", out, env)
