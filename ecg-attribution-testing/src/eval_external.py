"""External test of ecg-attribution-testing on the Chapman-Shaoxing-Ningbo subset (prepare_external.py). No setting is tuned here.

For every stored model (15 trained, 3 untrained): per-class and macro AUC on the external records. For every item
(record x one of its superclasses), relevance profiles of saliency, gradient x input, integrated gradients (32 steps)
and CAM, defined exactly as in attribute.py, plus the model-free amplitude profile; segment shares with the NeuroKit2
masks; deletion of the top 10% versus 10% random time steps; seed agreement (Spearman) between trained models.
Writes results/external/: performance.csv, local_<m>.csv, deletion_<m>.csv, seed_spearman.csv, items.csv
Usage: python eval_external.py <ecg-attribution-testing dir> [--threads 8] [--limit N]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ECGNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()
torch.set_num_threads(args.threads)
R = args.root
dev = "cuda" if torch.cuda.is_available() else "cpu"
O = os.path.join(R, "results", "external")
os.makedirs(O, exist_ok=True)
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
SEG = ["P", "QRS", "STT", "other"]
METHODS = ["saliency", "gxi", "ig", "cam", "amplitude"]
X = np.load(os.path.join(R, "data", "external", "processed", "signals.npy"))
M = np.load(os.path.join(R, "data", "external", "processed", "masks.npy"))
meta = pd.read_csv(os.path.join(R, "data", "external", "processed", "meta.csv"))
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
Y = meta[SUPER].values
items = [(i, k) for i in range(len(meta)) for k in range(5) if Y[i, k] == 1]
if args.limit:
    items = items[:args.limit]
items = pd.DataFrame(items, columns=["array_index", "label"])
items["label_name"] = [SUPER[k] for k in items.label]
n = len(items)
models = []
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    m = ECGNet(r["model"]).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location=dev))
    nz = np.load(os.path.join(d, "norm.npz"))
    models.append((r, m, torch.tensor(nz["mean"], device=dev), torch.tensor(nz["std"], device=dev)))
print("records", len(meta), "items", n, "models", len(models), flush=True)

# ---------------------------------------------------------------- classification on all external records
perf = []
for r, m, mu, sd in models:
    P = []
    with torch.no_grad():
        for s in range(0, len(X), 128):
            x = ((torch.from_numpy(X[s:s + 128]).to(dev) - mu) / sd).float()
            P.append(torch.sigmoid(m(x)).cpu().numpy())
    P = np.concatenate(P)
    aucs = {c: roc_auc_score(Y[:, k], P[:, k]) if 0 < Y[:, k].sum() < len(Y) else np.nan for k, c in enumerate(SUPER)}
    perf.append(dict(exp_id=r["exp_id"], model=r["model"], seed=r["seed"], random_init=r["random_init"], macro_auc=float(np.nanmean(list(aucs.values()))),
                     **{"auc_" + c: v for c, v in aucs.items()}))
    np.save(os.path.join(O, "probs_%s.npy" % r["exp_id"]), P.astype(np.float32))
pd.DataFrame(perf).to_csv(os.path.join(O, "performance.csv"), index=False)
print(pd.DataFrame(perf).groupby(["model", "random_init"]).macro_auc.agg(["mean", "std"]).round(3).to_string(), flush=True)

# ---------------------------------------------------------------- attributions on external items
prof = {mt: np.zeros((n, len(models), 1000), np.float32) for mt in METHODS}
dele = {mt: np.zeros((n, len(models), 2), np.float32) for mt in METHODS}
rng = np.random.default_rng(0)


def logit_of(m, x, lab):
    return m(x).gather(1, lab[:, None])[:, 0]


t0 = time.time()
for s in range(0, n, args.batch):
    b = items.iloc[s:s + args.batch]
    raw = torch.from_numpy(X[b.array_index.values]).to(dev)
    lab = torch.tensor(b.label.values, device=dev)
    for j, (r, m, mu, sd) in enumerate(models):
        x = ((raw - mu) / sd).float()
        with torch.no_grad():
            base = logit_of(m, x, lab)
        res = {}
        xg = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(logit_of(m, xg, lab).sum(), xg)
        res["saliency"] = g.abs().sum(1)
        res["gxi"] = (x * g).clamp(min=0).sum(1)
        ig = torch.zeros_like(x)
        for a in (torch.arange(32, device=dev) + 0.5) / 32:
            xa = (a * x).requires_grad_(True)
            ga, = torch.autograd.grad(logit_of(m, xa, lab).sum(), xa)
            ig += ga / 32
        res["ig"] = (x * ig).clamp(min=0).sum(1)
        with torch.no_grad():
            cam = m.cam(x)[torch.arange(len(b)), lab]
            res["cam"] = F.interpolate(cam[:, None], size=1000, mode="linear", align_corners=False)[:, 0].clamp(min=0)
            res["amplitude"] = x.abs().sum(1)
            for mt, p in res.items():
                prof[mt][s:s + len(b), j] = p.cpu().numpy()
                top = p.topk(100, dim=1).indices
                xd = x.clone()
                xd.scatter_(2, top[:, None, :].expand(-1, 12, -1), 0.0)
                rnd = torch.stack([torch.from_numpy(rng.choice(1000, 100, replace=False)) for _ in range(len(b))]).to(dev)
                xr = x.clone()
                xr.scatter_(2, rnd[:, None, :].expand(-1, 12, -1), 0.0)
                dele[mt][s:s + len(b), j, 0] = (base - logit_of(m, xd, lab)).cpu().numpy()
                dele[mt][s:s + len(b), j, 1] = (base - logit_of(m, xr, lab)).cpu().numpy()
    print("items", min(s + args.batch, n), "of", n, "%.0f s" % (time.time() - t0), flush=True)

ok = meta.fiducials_ok.values[items.array_index]
mk = M[items.array_index.values]
for mt in METHODS:
    P = prof[mt]
    tot = P.sum(2) + 1e-12
    rows = dict(item=np.repeat(np.arange(n), len(models)), model_idx=np.tile(np.arange(len(models)), n))
    for si, sg in enumerate(SEG):
        rows["share_" + sg] = ((P * mk[:, None, si, :]).sum(2) / tot).ravel()
        rows["time_" + sg] = np.repeat(mk[:, si, :].mean(1), len(models))
    rows["fiducials_ok"] = np.repeat(ok, len(models))
    rows["zero_map"] = (tot.ravel() < 1e-9)
    pd.DataFrame(rows).to_csv(os.path.join(O, "local_%s.csv" % mt), index=False)
    pd.DataFrame(dict(item=np.repeat(np.arange(n), len(models)), model_idx=np.tile(np.arange(len(models)), n),
                      drop_top10=dele[mt][:, :, 0].ravel(), drop_rand10=dele[mt][:, :, 1].ravel())).to_csv(os.path.join(O, "deletion_%s.csv" % mt), index=False)
# seed agreement between trained models of one architecture (mean Spearman over the 10 seed pairs)
sp = []
for mt in METHODS[:4]:
    for a in ["xresnet1d50", "inception1d", "convtran1d"]:
        idx = list(reg.index[(reg.model == a) & (reg.random_init == 0)])
        ridx = int(reg.index[(reg.model == a) & (reg.random_init == 1)][0])
        rk = stats.rankdata(prof[mt][:, idx + [ridx]], axis=2)
        rk = (rk - rk.mean(2, keepdims=True)) / (rk.std(2, keepdims=True) + 1e-9)
        seed = np.mean([(rk[:, i] * rk[:, j]).mean(1) for i in range(5) for j in range(i + 1, 5)], 0)
        rand = np.mean([(rk[:, i] * rk[:, 5]).mean(1) for i in range(5)], 0)
        sp.append(dict(method=mt, arch=a, seed_spearman=float(np.nanmean(seed)), untrained_spearman=float(np.nanmean(rand)), n=n))
pd.DataFrame(sp).to_csv(os.path.join(O, "seed_spearman.csv"), index=False)
items.to_csv(os.path.join(O, "items.csv"), index=False)
print("EVAL_EXTERNAL_DONE", "%.0f s" % (time.time() - t0))
