"""Paper 08 follow-up quantisation test (added 16 Oct 2026 after extra.py showed that more calibration data and float
activations do not prevent the accuracy collapse of the four backbones without residual-only plain convolutions).
Hypothesis: the collapse comes from quantising depthwise convolutions (groups > 1), which ResNet-18/50 do not contain.
Variant 'no_depthwise_256': the default static x86 recipe of efficiency.py, but every Conv2d with groups > 1 (and the module
that follows it in a fused pattern is handled by FX) is left in float. Pooled models, seed 0; balanced accuracy on both
test sets and one-thread latency. Appends to results/quant_ablation.csv (rows with this variant are replaced).
Usage: python extra_q2.py <Paper_08 dir>"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, ImgSet, splits  # noqa: E402

import timm  # noqa: E402
from torch.ao.quantization import get_default_qconfig_mapping  # noqa: E402
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx  # noqa: E402

R = sys.argv[1]
torch.set_num_threads(4)
torch.backends.quantized.engine = "x86"
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
A = np.load(os.path.join(R, "data", "processed", "images.npy"), mmap_mode="r")
part = splits(meta); y = meta.label.values
tests = {"trashnet": np.where((meta.source == "TrashNet") & (part == "test"))[0], "taco": np.where((meta.source == "TACO") & (part == "test"))[0]}
cal = np.random.default_rng(0).choice(np.where(part == "train")[0], 256, replace=False)


def batches(idx, bs=64):
    ds = ImgSet(A, y, idx, False)
    for s in range(0, len(idx), bs):
        yield torch.stack([ds[k][0] for k in range(s, min(s + bs, len(idx)))])


@torch.no_grad()
def bacc(model, idx):
    return float(balanced_accuracy_score(y[idx], np.concatenate([model(x).argmax(1).numpy() for x in batches(idx)])))


@torch.no_grad()
def latency(model):
    torch.set_num_threads(1)
    x = torch.randn(1, 3, 224, 224)
    for _ in range(10):
        model(x)
    t = []
    for _ in range(50):
        a = time.perf_counter(); model(x); t.append(time.perf_counter() - a)
    torch.set_num_threads(4)
    return float(np.median(t) * 1000)


rows = []
for r in reg[(reg.train_on == "both") & (reg.seed == 0) & ~reg.model.str.startswith("resnet")].to_dict("records"):
    net = timm.create_model(r["model"], pretrained=False, num_classes=len(CLASSES)).eval()
    net.load_state_dict(torch.load(os.path.join(R, "experiments", r["exp_id"], "best_model.pt"), map_location="cpu"))
    mapping = get_default_qconfig_mapping("x86")
    dw = [n for n, m in net.named_modules() if isinstance(m, torch.nn.Conv2d) and m.groups > 1]
    for n in dw:
        mapping = mapping.set_module_name(n, None)
    row = dict(model=r["model"], exp_id=r["exp_id"], variant="no_depthwise_256", n_calib=256, n_depthwise=len(dw))
    try:
        q = prepare_fx(net, mapping, (torch.randn(1, 3, 224, 224),))
        with torch.no_grad():
            for x in batches(cal):
                q(x)
        q = convert_fx(q)
        row.update(bacc_trashnet=bacc(q, tests["trashnet"]), bacc_taco=bacc(q, tests["taco"]), ms_1t=latency(q), status="ok")
    except Exception as e:  # noqa: BLE001
        row["status"] = "FAILED: " + repr(e)[:160]
    rows.append(row)
    print(row, flush=True)
f = os.path.join(R, "results", "quant_ablation.csv")
old = pd.read_csv(f)
old = old[old.variant != "no_depthwise_256"]
pd.concat([old, pd.DataFrame(rows)]).to_csv(f, index=False)
print("EXTRA_Q2_DONE")
