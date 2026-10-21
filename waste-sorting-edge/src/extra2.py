"""waste-sorting-edge analyses added after the main results (CPU, stored models), part 2: calibration repair and selective prediction.

For every run: logits of the validation images of its training source(s) are computed with the stored best model; a single
temperature T is fitted on them by minimising the negative log-likelihood (bounded search over log T); the TACO and TrashNet
test logits (predictions.csv) are rescaled by T. Reported per run and test set: ECE before and after scaling, and the
selective accuracy when only the most confident fraction c of images is accepted (c = 1.0, 0.9, 0.8, 0.7, 0.5; the rest
would go to manual sorting). Temperature scaling does not change the predicted class, so accuracy is unchanged.
Writes results/calibration.csv and results/selective.csv.
Usage: python extra2.py <waste-sorting-edge dir>
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, ImgSet, splits  # noqa: E402

import timm  # noqa: E402

R = sys.argv[1]
torch.set_num_threads(8)
K = len(CLASSES)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
A = np.load(os.path.join(R, "data", "processed", "images.npy"), mmap_mode="r")
part = splits(meta); y = meta.label.values
SRC = {"trashnet": ["TrashNet"], "taco": ["TACO"], "both": ["TrashNet", "TACO"]}
COV = [1.0, 0.9, 0.8, 0.7, 0.5]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, yy, bins=15):
    c, h = p.max(1), p.argmax(1)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (c > a) & (c <= b)
        if m.any():
            e += m.mean() * abs((h[m] == yy[m]).mean() - c[m].mean())
    return e


def nll(z, yy, T):
    p = softmax(z / T)
    return -np.log(p[np.arange(len(yy)), yy] + 1e-12).mean()


cal, sel = [], []
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    va = np.where(meta.source.isin(SRC[r["train_on"]]) & (part == "val"))[0]
    net = timm.create_model(r["model"], pretrained=False, num_classes=K).eval()
    net.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location="cpu"))
    ds = ImgSet(A, y, va, False)
    with torch.no_grad():
        zv = np.concatenate([net(torch.stack([ds[k][0] for k in range(s, min(s + 64, len(va)))])).numpy() for s in range(0, len(va), 64)])
    res = minimize_scalar(lambda lt: nll(zv, y[va], np.exp(lt)), bounds=(-3, 3), method="bounded")
    T = float(np.exp(res.x))
    p = pd.read_csv(os.path.join(d, "predictions.csv"))
    for s, key in [("test_trashnet", "TN"), ("test_taco", "TA")]:
        q = p[p.set == s]
        z = q[["logit_%d" % k for k in range(K)]].values
        yy = q.label.values
        p0, p1 = softmax(z), softmax(z / T)
        cal.append(dict(exp_id=r["exp_id"], model=r["model"], train_on=r["train_on"], seed=r["seed"], test=key, T=T,
                        ece_before=ece(p0, yy), ece_after=ece(p1, yy), nll_before=nll(z, yy, 1.0), nll_after=nll(z, yy, T)))
        conf, corr = p1.max(1), (p1.argmax(1) == yy)
        order = np.argsort(-conf)
        for c in COV:
            n = max(1, int(round(c * len(yy))))
            sel.append(dict(exp_id=r["exp_id"], model=r["model"], train_on=r["train_on"], seed=r["seed"], test=key, coverage=c,
                            sel_acc=corr[order[:n]].mean()))
    print(r["exp_id"], r["model"], r["train_on"], "T=%.2f" % T, flush=True)
pd.DataFrame(cal).to_csv(os.path.join(R, "results", "calibration.csv"), index=False)
pd.DataFrame(sel).to_csv(os.path.join(R, "results", "selective.csv"), index=False)
print("EXTRA2_DONE")
