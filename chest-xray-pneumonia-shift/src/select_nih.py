"""Select the NIH ChestX-ray14 images used in chest-xray-pneumonia-shift (fixed before any model is trained).

Usage: python select_nih.py <chest-xray-pneumonia-shift dir>
positive = 'Pneumonia' among the finding labels (any co-findings); negative = 'No Finding'.
NIH-paediatric (age < 18): all positives and all negatives.
NIH-adult (age >= 18): all positives; negatives = all 'No Finding' images of a random sample of adult patients
(seed 0), sampled until there are four negatives per positive image. Patients are never split between groups because
age is per image; a patient with images on both sides of 18 is assigned by the age at the first image.
Writes data/raw/nih_selection.csv and nih_wanted.txt (file names for remote extraction).
"""
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
raw = os.path.join(R, "data", "raw")
d = pd.read_csv(os.path.join(raw, "Data_Entry_2017_v2020.csv"))
d["pneumonia"] = d["Finding Labels"].str.contains("Pneumonia").astype(int)
d["no_finding"] = (d["Finding Labels"] == "No Finding").astype(int)
first_age = d.sort_values("Follow-up #").groupby("Patient ID")["Patient Age"].first()
d["group"] = np.where(d["Patient ID"].map(first_age) < 18, "nih_paed", "nih_adult")
keep = d[(d.pneumonia == 1) | (d.no_finding == 1)].copy()
paed = keep[keep.group == "nih_paed"]
adult = keep[keep.group == "nih_adult"]
pos_a = adult[adult.pneumonia == 1]
neg_pool = adult[adult.pneumonia == 0]
rng = np.random.default_rng(0)
pats = rng.permutation(neg_pool["Patient ID"].unique())
target, chosen, n = 4 * len(pos_a), [], 0
cnt = neg_pool.groupby("Patient ID").size()
for p in pats:
    if n >= target:
        break
    chosen.append(p); n += cnt[p]
neg_a = neg_pool[neg_pool["Patient ID"].isin(chosen)]
sel = pd.concat([paed, pos_a, neg_a])
sel = sel.rename(columns={"Image Index": "image", "Patient ID": "patient", "Patient Age": "age", "Patient Gender": "sex",
                          "View Position": "view", "Finding Labels": "findings"})
sel["label"] = sel.pneumonia
sel[["image", "patient", "age", "sex", "view", "findings", "label", "group", "Follow-up #"]].to_csv(os.path.join(raw, "nih_selection.csv"), index=False)
open(os.path.join(raw, "nih_wanted.txt"), "w").write("\n".join(sel.image) + "\n")
print(sel.groupby(["group", "label"]).agg(images=("image", "size"), patients=("patient", "nunique")))
