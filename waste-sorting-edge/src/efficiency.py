"""waste-sorting-edge efficiency measurements on CPU (run inside a SLURM job so the CPU cores are allocated to it).

Usage: python efficiency.py <waste-sorting-edge dir>
For every backbone, using its 'both'-trained runs (seeds 0-2):
  - parameters; multiply-accumulate operations (MACs) for one 224 x 224 image (torch FLOP counter / 2)
  - CPU latency for batch size 1 at 1 and 4 threads: median of 50 timed forward passes after 10 warm-up passes (fp32)
  - int8 post-training static quantisation (PyTorch FX graph mode, x86 backend, per-channel weights) calibrated on
    256 training images; latency as above and balanced accuracy on both test sets (seed 0..2), so the accuracy cost of
    int8 can be compared with the fp32 accuracy of the same weights. Failures of the quantisation are recorded, not hidden.
The CPU is the server's (not an edge device); 1-thread latency is reported as the closest proxy for small hardware.
Output: results/efficiency.csv
"""
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, ImgSet, splits  # noqa: E402

R = sys.argv[1]
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
A = np.load(os.path.join(R, "data", "processed", "images.npy"), mmap_mode="r")
part = splits(meta); y = meta.label.values
tests = {"trashnet": np.where((meta.source == "TrashNet") & (part == "test"))[0], "taco": np.where((meta.source == "TACO") & (part == "test"))[0]}
calib = np.random.default_rng(0).choice(np.where(part == "train")[0], 256, replace=False)
import timm  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402


def batches(idx, bs=64):
    ds = ImgSet(A, y, idx, False)
    for s in range(0, len(idx), bs):
        yield torch.stack([ds[k][0] for k in range(s, min(s + bs, len(idx)))])


@torch.no_grad()
def bacc(model, idx):
    pr = np.concatenate([model(x).argmax(1).numpy() for x in batches(idx)])
    return float(balanced_accuracy_score(y[idx], pr))


@torch.no_grad()
def latency(model, threads):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 224, 224)
    for _ in range(10):
        model(x)
    t = []
    for _ in range(50):
        a = time.perf_counter(); model(x); t.append(time.perf_counter() - a)
    return float(np.median(t) * 1000)


rows = []
for m in reg.model.unique():
    runs = reg[(reg.model == m) & (reg.train_on == "both")].sort_values("seed")
    for k, r in enumerate(runs.to_dict("records")):
        d = os.path.join(R, "experiments", r["exp_id"])
        if not os.path.exists(os.path.join(d, "best_model.pt")):
            continue
        net = timm.create_model(m, pretrained=False, num_classes=len(CLASSES)).eval()
        net.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location="cpu"))
        row = dict(model=m, exp_id=r["exp_id"], seed=r["seed"], params_m=sum(q.numel() for q in net.parameters()) / 1e6)
        torch.set_num_threads(4)
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            net(torch.randn(1, 3, 224, 224))
        row["gmacs"] = fc.get_total_flops() / 2e9
        row["fp32_bacc_trashnet"], row["fp32_bacc_taco"] = bacc(net, tests["trashnet"]), bacc(net, tests["taco"])
        if k == 0:
            row["fp32_ms_1t"], row["fp32_ms_4t"] = latency(net, 1), latency(net, 4)
        try:
            from torch.ao.quantization import get_default_qconfig_mapping
            from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
            torch.backends.quantized.engine = "x86"
            ex = (torch.randn(1, 3, 224, 224),)
            q = prepare_fx(net, get_default_qconfig_mapping("x86"), ex)
            torch.set_num_threads(4)
            with torch.no_grad():
                for x in batches(calib):
                    q(x)
            q = convert_fx(q)
            row["int8_bacc_trashnet"], row["int8_bacc_taco"] = bacc(q, tests["trashnet"]), bacc(q, tests["taco"])
            if k == 0:
                row["int8_ms_1t"], row["int8_ms_4t"] = latency(q, 1), latency(q, 4)
            row["int8_status"] = "ok"
        except Exception as e:
            row["int8_status"] = "FAILED: " + repr(e)[:160]
        rows.append(row)
        print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
os.makedirs(os.path.join(R, "results"), exist_ok=True)
df = pd.DataFrame(rows)
df["cpu"] = platform.processor() or platform.machine()
df["torch"] = torch.__version__
df.to_csv(os.path.join(R, "results", "efficiency.csv"), index=False)
print("EFFICIENCY_DONE")
