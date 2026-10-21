"""Dataset inspection and duplicate audit for HAM10000 and the ISIC 2018 Task 3 test set.

Usage: python inspect_dataset.py <skin-lesion-model-ranking dir>
Writes data/inspection/: summary.json, class_table.csv, lesion_multiplicity.csv, source_by_class.csv,
near_duplicates_ham.csv, near_duplicates_ham_test.csv, image_split_leakage.csv
Near-duplicate rule: dHash Hamming distance <= 6 AND Pearson correlation of 32 x 32 grey thumbnails >= 0.95
(both thresholds fixed before looking at the pairs). Candidate pairs are listed for manual viewing.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, make_split  # noqa: E402

root = sys.argv[1]
proc = os.path.join(root, "data", "processed")
out = os.path.join(root, "data", "inspection")
os.makedirs(out, exist_ok=True)
ham = pd.read_csv(os.path.join(proc, "ham_meta.csv"), dtype={"dhash": str})
test = pd.read_csv(os.path.join(proc, "test_meta.csv"), dtype={"dhash": str})
chk = pd.read_csv(os.path.join(proc, "file_check.csv"))
th_h = np.load(os.path.join(proc, "ham_thumbs32.npy")).reshape(len(ham), -1).astype(np.float32)
th_t = np.load(os.path.join(proc, "test_thumbs32.npy")).reshape(len(test), -1).astype(np.float32)

S = dict()
S["files"] = dict(checked=int(len(chk)), unreadable=int((~chk["ok"].astype(bool)).sum()),
                  sizes=chk.groupby(["part", "width", "height"]).size().reset_index().values.tolist())
for name, m in [("ham", ham), ("test", test)]:
    S[name] = dict(images=int(len(m)), lesions=int(m["lesion_id"].nunique()), duplicate_image_ids=int(m["image_id"].duplicated().sum()),
                   lesions_with_mixed_dx=int((m.groupby("lesion_id")["dx"].nunique() > 1).sum()),
                   missing=m[["age", "sex", "localization"]].isna().sum().to_dict(),
                   sex=m["sex"].value_counts().to_dict(), dx_type=m["dx_type"].value_counts().to_dict(),
                   source=m["dataset"].value_counts().to_dict())
S["lesion_ids_shared_ham_test"] = int(len(set(ham["lesion_id"]) & set(test["lesion_id"])))
S["image_ids_shared_ham_test"] = int(len(set(ham["image_id"]) & set(test["image_id"])))

rows = []
for c in CLASSES:
    h, t = ham[ham["dx"] == c], test[test["dx"] == c]
    rows.append(dict(dx=c, ham_images=len(h), ham_lesions=h["lesion_id"].nunique(), ham_images_per_lesion=len(h) / max(1, h["lesion_id"].nunique()),
                     ham_share=len(h) / len(ham), test_images=len(t), test_share=len(t) / len(test),
                     ham_histo=(h["dx_type"] == "histo").mean(), test_histo=(t["dx_type"] == "histo").mean()))
pd.DataFrame(rows).to_csv(os.path.join(out, "class_table.csv"), index=False)
mult = ham.groupby("lesion_id").agg(n=("image_id", "size"), dx=("dx", "first"))
pd.crosstab(mult["n"], mult["dx"]).to_csv(os.path.join(out, "lesion_multiplicity.csv"))
pd.concat([pd.crosstab(ham["dataset"], ham["dx"]).assign(part="ham"), pd.crosstab(test["dataset"], test["dx"]).assign(part="test")]).to_csv(
    os.path.join(out, "source_by_class.csv"))


def hamming_matrix(a, b):
    x = (a[:, None] ^ b[None, :])
    cnt = np.zeros(x.shape, np.int16)
    for s in range(64):
        cnt += ((x >> np.uint64(s)) & np.uint64(1)).astype(np.int16)
    return cnt


def corr_rows(a, b):
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)
    return (a * b).mean(1)


def near_pairs(ma, ta, mb, tb, same):
    ha = np.array([int(h, 16) for h in ma["dhash"]], np.uint64)
    hb = np.array([int(h, 16) for h in mb["dhash"]], np.uint64)
    res = []
    for s in range(0, len(ha), 500):
        d = hamming_matrix(ha[s:s + 500], hb)
        ii, jj = np.where(d <= 6)
        ii = ii + s
        if same:
            keep = ii < jj
            ii, jj = ii[keep], jj[keep]
        if len(ii):
            r = corr_rows(ta[ii], tb[jj])
            for i, j, h, c in zip(ii, jj, d[ii - s, jj], r):
                res.append(dict(a=ma["image_id"].values[i], b=mb["image_id"].values[j], hamming=int(h), thumb_corr=float(c),
                                lesion_a=ma["lesion_id"].values[i], lesion_b=mb["lesion_id"].values[j],
                                dx_a=ma["dx"].values[i], dx_b=mb["dx"].values[j], near_duplicate=bool(c >= 0.95)))
    return pd.DataFrame(res, columns=["a", "b", "hamming", "thumb_corr", "lesion_a", "lesion_b", "dx_a", "dx_b", "near_duplicate"])


p_hh = near_pairs(ham, th_h, ham, th_h, True)
p_hh["same_lesion"] = p_hh["lesion_a"] == p_hh["lesion_b"]
p_hh.to_csv(os.path.join(out, "near_duplicates_ham.csv"), index=False)
p_ht = near_pairs(ham, th_h, test, th_t, False)
p_ht.to_csv(os.path.join(out, "near_duplicates_ham_test.csv"), index=False)
nd = p_hh[p_hh["near_duplicate"]]
S["near_duplicates"] = dict(
    ham_pairs_same_lesion=int(nd["same_lesion"].sum()), ham_pairs_different_lesion=int((~nd["same_lesion"]).sum()),
    ham_pairs_different_lesion_different_dx=int(((~nd["same_lesion"]) & (nd["dx_a"] != nd["dx_b"])).sum()),
    ham_test_pairs=int(p_ht["near_duplicate"].sum()),
    ham_test_test_images_involved=int(p_ht.loc[p_ht["near_duplicate"], "b"].nunique()))

# How often does a random image-level split put another image of the same lesion in training?
rows = []
for fold in range(5):
    tr, va, te = make_split(ham, "image", fold, 0)
    trl = set(ham["lesion_id"].values[tr])
    seen = ham["lesion_id"].iloc[te].isin(trl).values
    for c in CLASSES:
        sel = ham["dx"].values[te] == c
        rows.append(dict(fold=fold, dx=c, test_images=int(sel.sum()), test_images_with_lesion_in_train=int(seen[sel].sum())))
    trg, vag, teg = make_split(ham, "lesion", fold, 0)
    assert not (set(ham["lesion_id"].values[trg]) & set(ham["lesion_id"].values[teg])), "lesion split leaks"
lk = pd.DataFrame(rows)
lk.to_csv(os.path.join(out, "image_split_leakage.csv"), index=False)
g = lk.groupby("dx")[["test_images", "test_images_with_lesion_in_train"]].sum()
S["image_split_leakage"] = dict(overall=float(g.values[:, 1].sum() / g.values[:, 0].sum()),
                                by_class={c: float(g.loc[c].iloc[1] / g.loc[c].iloc[0]) for c in CLASSES})
json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
