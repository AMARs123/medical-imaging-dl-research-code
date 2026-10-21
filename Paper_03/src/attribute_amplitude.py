"""Model-free control for Paper 03: the 'amplitude' profile r(t) = sum over leads |x_{l,t}| of the standardised signal.

Added after the first analysis showed that gradient x input and integrated gradients concentrate relevance in the
QRS complex for every class and even for randomly initialised models. If a model-free profile that only measures signal
amplitude shows the same localisation and a similar deletion gain, those properties cannot be credited to the model.
Usage: python attribute_amplitude.py <Paper_03 dir>   (inference only; runs on CPU)
Writes results/attributions/{profiles,local,deletion}_amplitude.* in the same format as attribute.py, using the same
items, the same deletion rule (top 10% vs 10% random time steps set to 0; random steps drawn with seed 0).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ECGNet  # noqa: E402
from segments import segment_masks  # noqa: E402

R = sys.argv[1]
A = os.path.join(R, "results", "attributions")
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
X = np.load(os.path.join(R, "data", "processed", "signals.npy"), mmap_mode="r")
items = pd.read_csv(os.path.join(A, "items.csv"))
n, M = len(items), len(reg)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
models = []
for r in reg.to_dict("records"):
    d = os.path.join(R, "experiments", r["exp_id"])
    m = ECGNet(r["model"]).eval()
    m.load_state_dict(torch.load(os.path.join(d, "best_model.pt"), map_location="cpu"))
    nz = np.load(os.path.join(d, "norm.npz"))
    models.append((m, torch.tensor(nz["mean"]), torch.tensor(nz["std"])))
prof = np.zeros((n, M, 1000), np.float16)
dele = np.zeros((n, M, 3), np.float32)
rng = np.random.default_rng(0)
B = 64
with torch.no_grad():
    for s in range(0, n, B):
        b = items.iloc[s:s + B]
        raw = torch.from_numpy(np.asarray(X[b.array_index.values]))
        lab = torch.tensor(b.label.values)
        rnd = torch.stack([torch.from_numpy(rng.choice(1000, 100, replace=False)) for _ in range(len(b))])
        for j, (m, mu, sd) in enumerate(models):
            x = ((raw - mu) / sd).float()
            p = x.abs().sum(1)
            prof[s:s + len(b), j] = p.numpy().astype(np.float16)
            base = m(x).gather(1, lab[:, None])[:, 0]
            xd = x.clone(); xd.scatter_(2, p.topk(100, 1).indices[:, None, :].expand(-1, 12, -1), 0.0)
            xr = x.clone(); xr.scatter_(2, rnd[:, None, :].expand(-1, 12, -1), 0.0)
            dele[s:s + len(b), j] = torch.stack([base, base - m(xd).gather(1, lab[:, None])[:, 0],
                                                base - m(xr).gather(1, lab[:, None])[:, 0]], 1).numpy()
        print("done", min(s + B, n), "of", n, flush=True)
np.save(os.path.join(A, "profiles_amplitude.npy"), prof)
masks = np.stack([segment_masks(int(e)) if segment_masks(int(e)) is not None else np.zeros((4, 1000), bool) for e in items.ecg_id])
ok = masks[:, 1].any(1)
P = prof.astype(np.float32); tot = P.sum(2) + 1e-12
rows = dict(item=np.repeat(np.arange(n), M), model_idx=np.tile(np.arange(M), n))
for si, sg in enumerate(["P", "QRS", "STT", "other"]):
    rows["share_" + sg] = ((P * masks[:, None, si, :]).sum(2) / tot).ravel()
    rows["time_" + sg] = np.repeat(masks[:, si, :].mean(1), M)
rows["fiducials_ok"] = np.repeat(ok, M); rows["zero_map"] = tot.ravel() < 1e-9
pd.DataFrame(rows).to_csv(os.path.join(A, "local_amplitude.csv"), index=False)
pd.DataFrame(dict(item=np.repeat(np.arange(n), M), model_idx=np.tile(np.arange(M), n), base_logit=dele[:, :, 0].ravel(),
                  drop_top10=dele[:, :, 1].ravel(), drop_rand10=dele[:, :, 2].ravel())).to_csv(os.path.join(A, "deletion_amplitude.csv"), index=False)
print("AMPLITUDE_DONE")
