"""Paper 08 additional analyses on CPU from the stored models (added 16 Oct 2026 after the main results; labelled as such).

1. Quantisation ablation (pooled models, seed 0): the default static recipe (efficiency.py) is compared with
   (a) 1,024 instead of 256 calibration images and (b) quantising only convolutions and linear layers, all other operations
   kept in float (activation functions, squeeze-and-excitation, normalisation, pooling). Balanced accuracy on both test sets
   and one-thread latency (median of 50 passes after 10 warm-up passes). -> results/quant_ablation.csv
2. Domain separability of the learned features (seed 0 of every backbone and training source): penultimate features of the
   TrashNet and TACO test images; a logistic-regression domain classifier, 5-fold cross-validated AUC (folds grouped by
   photograph). A high AUC means the two sources stay separate in feature space. -> results/domain_auc.csv
3. Reliability bins on the TACO test set by training source, pooled over backbones and seeds (15 equal-width confidence
   bins). -> results/reliability.csv
Usage: python extra.py <Paper_08 dir>
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, ImgSet, splits  # noqa: E402

import timm  # noqa: E402

R = sys.argv[1]
torch.set_num_threads(4)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
A = np.load(os.path.join(R, "data", "processed", "images.npy"), mmap_mode="r")
part = splits(meta); y = meta.label.values
tests = {"trashnet": np.where((meta.source == "TrashNet") & (part == "test"))[0], "taco": np.where((meta.source == "TACO") & (part == "test"))[0]}
train_idx = np.where(part == "train")[0]


def batches(idx, bs=64):
    ds = ImgSet(A, y, idx, False)
    for s in range(0, len(idx), bs):
        yield torch.stack([ds[k][0] for k in range(s, min(s + bs, len(idx)))])


@torch.no_grad()
def bacc(model, idx):
    pr = np.concatenate([model(x).argmax(1).numpy() for x in batches(idx)])
    return float(balanced_accuracy_score(y[idx], pr))


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


def load(r):
    net = timm.create_model(r["model"], pretrained=False, num_classes=len(CLASSES)).eval()
    net.load_state_dict(torch.load(os.path.join(R, "experiments", r["exp_id"], "best_model.pt"), map_location="cpu"))
    return net


OUT = os.path.join(R, "results")
# ---------------------------------------------------------------- 1. quantisation ablation
from torch.ao.quantization import QConfigMapping, get_default_qconfig, get_default_qconfig_mapping  # noqa: E402
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx  # noqa: E402

torch.backends.quantized.engine = "x86"
qrows = []
for r in reg[(reg.train_on == "both") & (reg.seed == 0)].to_dict("records"):
    for variant in ["default_256", "default_1024", "conv_linear_only_256"]:
        net = load(r)
        ncal = 1024 if variant.endswith("1024") else 256
        cal = np.random.default_rng(0).choice(train_idx, ncal, replace=False)
        if variant.startswith("conv_linear"):
            qc = get_default_qconfig("x86")
            mapping = QConfigMapping().set_global(None).set_object_type(torch.nn.Conv2d, qc).set_object_type(torch.nn.Linear, qc)
        else:
            mapping = get_default_qconfig_mapping("x86")
        row = dict(model=r["model"], exp_id=r["exp_id"], variant=variant, n_calib=ncal)
        try:
            q = prepare_fx(net, mapping, (torch.randn(1, 3, 224, 224),))
            with torch.no_grad():
                for x in batches(cal):
                    q(x)
            q = convert_fx(q)
            row.update(bacc_trashnet=bacc(q, tests["trashnet"]), bacc_taco=bacc(q, tests["taco"]), ms_1t=latency(q), status="ok")
        except Exception as e:  # noqa: BLE001
            row["status"] = "FAILED: " + repr(e)[:160]
        qrows.append(row)
        print(row, flush=True)
pd.DataFrame(qrows).to_csv(os.path.join(OUT, "quant_ablation.csv"), index=False)

# ---------------------------------------------------------------- 2. domain separability of features
both = np.concatenate([tests["trashnet"], tests["taco"]])
dom = np.r_[np.zeros(len(tests["trashnet"])), np.ones(len(tests["taco"]))]
grp = meta.group.values[both]
drows = []
for r in reg[reg.seed == 0].to_dict("records"):
    net = load(r)
    with torch.no_grad():
        feats = np.concatenate([net.forward_head(net.forward_features(x), pre_logits=True).numpy() for x in batches(both)])
    scores = np.zeros(len(both))
    for tr, te in GroupKFold(5).split(feats, dom, grp):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(feats[tr]), dom[tr])
        scores[te] = clf.predict_proba(sc.transform(feats[te]))[:, 1]
    drows.append(dict(model=r["model"], train_on=r["train_on"], exp_id=r["exp_id"], dim=feats.shape[1], domain_auc=roc_auc_score(dom, scores)))
    print(drows[-1], flush=True)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "domain_auc.csv"), index=False)

# ---------------------------------------------------------------- 3. reliability bins on TACO
rel = []
for tr, g in reg.groupby("train_on"):
    conf, corr = [], []
    for e in g.exp_id:
        p = pd.read_csv(os.path.join(R, "experiments", e, "predictions.csv"))
        p = p[p.set == "test_taco"]
        lo = p[["logit_%d" % k for k in range(len(CLASSES))]].values
        pr = np.exp(lo - lo.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        conf.append(pr.max(1)); corr.append((p.pred.values == p.label.values).astype(float))
    conf, corr = np.concatenate(conf), np.concatenate(corr)
    edges = np.linspace(0, 1, 16)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        rel.append(dict(train_on=tr, lo=a, hi=b, n=int(m.sum()), conf=conf[m].mean() if m.any() else np.nan, acc=corr[m].mean() if m.any() else np.nan))
pd.DataFrame(rel).to_csv(os.path.join(OUT, "reliability.csv"), index=False)
print("EXTRA_DONE")
