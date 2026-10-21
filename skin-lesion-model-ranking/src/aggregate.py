"""Collect every finished run into results/all_runs.csv (one row per run, test-fold and ISIC metrics flattened).

Usage: python aggregate.py <skin-lesion-model-ranking dir>
Also writes results/post_hoc.csv: for every run, post-hoc logit adjustment (logits - tau * log prior) with tau
chosen on the validation fold from a fixed grid, and temperature scaling (T chosen on validation NLL).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, K, SHORT, classification_metrics  # noqa: E402

root = sys.argv[1]
reg = pd.read_csv(os.path.join(root, "experiments", "experiment_registry.csv"))
os.makedirs(os.path.join(root, "results"), exist_ok=True)
TAUS = np.round(np.arange(0, 2.01, 0.25), 2)
TEMPS = np.round(np.exp(np.linspace(np.log(0.5), np.log(4), 36)), 4)


def flat(prefix, m):
    out = {prefix + k: v for k, v in m.items() if not isinstance(v, list)}
    for k, c in enumerate(CLASSES):
        out["%srecall_%s" % (prefix, c)] = m["per_class_recall"][k]
        out["%sf1_%s" % (prefix, c)] = m["per_class_f1"][k]
    return out


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logits(df):
    return df[["logit_" + c for c in CLASSES]].values.astype(np.float64)


rows, post = [], []
for r in reg.to_dict("records"):
    d = os.path.join(root, "experiments", r["exp_id"])
    f = os.path.join(d, "results.json")
    if not os.path.exists(f):
        continue
    res = json.load(open(f))
    row = dict(r, short=SHORT[r["model"]], n_params=res["n_params"], best_epoch=res["best_epoch"], train_seconds=res["train_seconds"],
               peak_gpu_mb=res["peak_gpu_mb"], leak_train_test=res["split"]["lesions_shared_train_test"])
    row.update(flat("test_", res["test"]))
    row.update(flat("isic_", res["isic"]))
    row.update(flat("val_", res["val"]))
    # ISIC 2018 test partitions (Codella et al. 2019): "external" = 316 images from institutions not in HAM10000
    pi_ = pd.read_csv(os.path.join(d, "predictions_isic.csv"))
    for part, sel in [("isicint_", pi_["source"] != "external"), ("isicext_", pi_["source"] == "external")]:
        q = pi_[sel]
        row.update(flat(part, classification_metrics(q["label"].values, softmax(logits(q)))))
    rows.append(row)

    # post-hoc corrections, chosen on validation only
    pv, pt, pi = (pd.read_csv(os.path.join(d, fn)) for fn in ["predictions_val.csv", "predictions.csv", "predictions_isic.csv"])
    tr_counts = np.array(res["split"]["train_class_counts"], float)
    logp = np.log(tr_counts / tr_counts.sum())
    yv = pv["label"].values
    best_tau = max(TAUS, key=lambda t: (classification_metrics(yv, softmax(logits(pv) - t * logp))["balanced_accuracy"], -t))
    nll = [-np.log(np.clip(softmax(logits(pv) / T)[np.arange(len(yv)), yv], 1e-12, 1)).mean() for T in TEMPS]
    best_T = float(TEMPS[int(np.argmin(nll))])
    p = dict(exp_id=r["exp_id"], tau=float(best_tau), temperature=best_T)
    for name, df in [("test_", pt), ("isic_", pi)]:
        y = df["label"].values
        p.update(flat(name + "la_", classification_metrics(y, softmax(logits(df) - best_tau * logp))))
        p.update(flat(name + "ts_", classification_metrics(y, softmax(logits(df) / best_T))))
    post.append(p)

pd.DataFrame(rows).to_csv(os.path.join(root, "results", "all_runs.csv"), index=False)
pd.DataFrame(post).to_csv(os.path.join(root, "results", "post_hoc.csv"), index=False)
print("runs collected:", len(rows), "of", len(reg))
