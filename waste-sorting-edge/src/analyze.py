"""waste-sorting-edge analysis, part 1: per-run test metrics, bootstrap intervals over photographs and pooled confusion matrices.

Reads experiments/experiment_registry.csv and every finished run's predictions.csv (logits of both fixed test sets).
Writes
  results/processed/runs.csv        one row per run and test set: accuracy, balanced accuracy, macro-F1, ECE (15 bins),
                                    recall per material
  results/processed/ci.csv          95% bootstrap interval (2000 resamples of photographs) of the seed-mean balanced accuracy
                                    for every backbone, training source and test set; the same resample is applied to all seeds
  results/processed/confusion.csv   row-normalised confusion on the TACO test set, pooled over backbones and seeds, for each
                                    training source
Usage: python analyze.py <waste-sorting-edge dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

R = sys.argv[1]
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
K = len(CLASSES)
OUT = os.path.join(R, "results", "processed")
os.makedirs(OUT, exist_ok=True)
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"))
meta = pd.read_csv(os.path.join(R, "data", "processed", "meta.csv"))
SETS = {"test_trashnet": "TN", "test_taco": "TA"}


def ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


rows, preds = [], []
for r in reg.to_dict("records"):
    f = os.path.join(R, "experiments", r["exp_id"], "predictions.csv")
    if not os.path.exists(os.path.join(R, "experiments", r["exp_id"], "results.json")) or not os.path.exists(f):
        continue
    p = pd.read_csv(f)
    lo = p[["logit_%d" % c for c in range(K)]].values
    pr = np.exp(lo - lo.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    p["conf"] = pr.max(1)
    for s, key in SETS.items():
        q = p[p.set == s]
        yy, yh = q.label.values, q.pred.values
        rec = recall_score(yy, yh, average=None, labels=list(range(K)), zero_division=0)
        rows.append(dict(exp_id=r["exp_id"], model=r["model"], train_on=r["train_on"], seed=r["seed"], test=key, n=len(q),
                         accuracy=(yy == yh).mean(), bacc=balanced_accuracy_score(yy, yh), macro_f1=f1_score(yy, yh, average="macro"),
                         ece=ece(q.conf.values, (yy == yh).astype(float)), **{"recall_" + c: rec[k] for k, c in enumerate(CLASSES)}))
    p["exp_id"], p["model"], p["train_on"], p["seed"] = r["exp_id"], r["model"], r["train_on"], r["seed"]
    preds.append(p[["exp_id", "model", "train_on", "seed", "array_index", "set", "label", "pred"]])
runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(OUT, "runs.csv"), index=False)
print("runs with results:", runs.exp_id.nunique(), "of", len(reg))
if not len(runs):
    sys.exit(0)
P = pd.concat(preds)
P["group"] = meta.group.values[P.array_index.values]

# ---------------------------------------------------------------- bootstrap over photographs
rng = np.random.default_rng(0)
B = 2000
ci = []
for (m, tr, s), g in P.groupby(["model", "train_on", "set"]):
    groups = np.sort(g.group.unique())
    gi = {u: i for i, u in enumerate(groups)}
    seeds = sorted(g.seed.unique())
    per = []
    for sd in seeds:
        q = g[g.seed == sd].sort_values("array_index")
        per.append((q.label.values, q.pred.values, np.array([gi[u] for u in q.group.values])))
    est = np.mean([balanced_accuracy_score(a, b) for a, b, _ in per])
    members = [[np.where(c == i)[0] for i in range(len(groups))] for _, _, c in per]
    bs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        vals = []
        for (a, h, _), mem in zip(per, members):
            idx = np.concatenate([mem[i] for i in pick])
            vals.append(balanced_accuracy_score(a[idx], h[idx]))
        bs[b] = np.mean(vals)
    ci.append(dict(model=m, train_on=tr, test=SETS[s], seeds=len(seeds), bacc_mean=est, ci_low=np.quantile(bs, 0.025), ci_high=np.quantile(bs, 0.975),
                   n_photos=len(groups)))
pd.DataFrame(ci).to_csv(os.path.join(OUT, "ci.csv"), index=False)

# ---------------------------------------------------------------- confusion on the TACO test set
cf = []
for tr, g in P[P.set == "test_taco"].groupby("train_on"):
    M = np.zeros((K, K))
    np.add.at(M, (g.label.values, g.pred.values), 1)
    M = M / np.maximum(M.sum(1, keepdims=True), 1)
    for i in range(K):
        for j in range(K):
            cf.append(dict(train_on=tr, true=CLASSES[i], pred=CLASSES[j], share=M[i, j]))
pd.DataFrame(cf).to_csv(os.path.join(OUT, "confusion.csv"), index=False)
print("ANALYZE_DONE")
