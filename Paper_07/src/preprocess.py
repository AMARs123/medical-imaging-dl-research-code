"""Paper 07 preprocessing: disc-centred crops from Chaksu (3 cameras), PAPILA and RIM-ONE DL, with labels.

Usage: python preprocess.py <Paper_07 dir>
Why disc-centred crops: RIM-ONE DL is only distributed as crops around the optic disc, and glaucoma is judged at the disc.
  Chaksu : crop centred on the majority-vote expert disc mask, side = 2 x the larger extent of the largest connected mask region (stray mask pixels ignored).
  PAPILA : same rule using the mean of the two experts' disc contours (txt point lists, original pixel coordinates).
  RIM-ONE DL: images are already disc crops; used as provided (union of both published partitions, de-duplicated by name).
All crops are padded with black if needed and resized to 384 x 384.
Labels:
  Chaksu : five experts each say NORMAL or GLAUCOMA SUSPECT. label = 1 if >= 3 of 5 say suspect; n_suspect (0-5) kept
           for the disagreement analysis. File names in the decision tables are matched to images by their stem.
  PAPILA : clinical diagnosis 0 healthy, 1 glaucoma, 2 suspect (per eye, from patient_data_od/os.xlsx). Main label: glaucoma
           vs healthy; suspects kept with label -1 and analysed separately.
  RIM-ONE DL: folder name (glaucoma / normal).
Outputs: data/processed/images.npy (uint8 [N,384,384,3]), meta.csv
"""
import glob
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

R = sys.argv[1]
raw = os.path.join(R, "data", "raw")
out = os.path.join(R, "data", "processed")
os.makedirs(out, exist_ok=True)
S = 384
rows = []


def stem(s):
    return os.path.basename(str(s)).split(".")[0].split("-")[0].strip()


# ---------------------------------------------------------------- Chaksu
for split in ["Train", "Test"]:
    dec = {}
    for f in glob.glob(os.path.join(raw, "chaksu", split, "6.0_Glaucoma_Decision", "*.csv")):
        cam = [c for c in ["Remidio", "Bosch", "Forus"] if c in os.path.basename(f)]
        if not cam:
            continue
        t = pd.read_csv(f, encoding="utf-8-sig")
        ex = [c for c in t.columns if re.match(r"Expert[ .]?\d", c)]
        for r in t.to_dict("records"):
            votes = [("SUS" in str(r[c]).upper()) or ("GLAUCOMA" in str(r[c]).upper()) for c in ex]
            dec[(cam[0], stem(r[t.columns[0]]))] = (int(sum(votes)), len(ex))
    for cam in ["Remidio", "Bosch", "Forus"]:
        for f in sorted(glob.glob(os.path.join(raw, "chaksu", split, "1.0_Original_Fundus_Images", cam, "*"))):
            st = os.path.splitext(os.path.basename(f))[0]
            m = os.path.join(raw, "chaksu", split, "5.0_OD_OC_Mean_Median_Majority_STAPLE", cam, "Disc", "Majority", st + ".tif")
            d = dec.get((cam, st))
            rows.append(dict(dataset="Chaksu", camera=cam, split=split.lower(), image=os.path.basename(f), path=f, disc=m,
                             n_suspect=d[0] if d else np.nan, n_experts=d[1] if d else np.nan,
                             label=int(d[0] >= 3) if d else -9))
# ---------------------------------------------------------------- PAPILA
P = glob.glob(os.path.join(raw, "papila", "PapilaDB-*"))[0]
for eye, fx in [("OD", "patient_data_od.xlsx"), ("OS", "patient_data_os.xlsx")]:
    t = pd.read_excel(os.path.join(P, "ClinicalData", fx), header=None).iloc[3:]
    for r in t.itertuples(index=False):
        pid = str(r[0]).replace("#", "").strip()
        if not pid or pid == "nan":
            continue
        dx = int(r[3])
        f = os.path.join(P, "FundusImages", "RET%s%s.jpg" % (pid, eye))
        rows.append(dict(dataset="PAPILA", camera="PAPILA", split="external", image=os.path.basename(f), path=f,
                         disc=";".join(sorted(glob.glob(os.path.join(P, "ExpertsSegmentations", "Contours", "RET%s%s_disc_exp*.txt" % (pid, eye))))),
                         n_suspect=np.nan, n_experts=np.nan, label={0: 0, 1: 1, 2: -1}[dx], patient="P" + pid))
# ---------------------------------------------------------------- RIM-ONE DL
seen = set()
for f in sorted(glob.glob(os.path.join(raw, "rimone", "RIM-ONE_DL_images", "*", "*", "*", "*"))):
    b = os.path.basename(f)
    if b in seen:
        continue
    seen.add(b)
    lab = 1 if "/glaucoma/" in f.replace("\\", "/") else 0
    rows.append(dict(dataset="RIMONE", camera="RIMONE", split="external", image=b, path=f, disc="", n_suspect=np.nan,
                     n_experts=np.nan, label=lab))
meta = pd.DataFrame(rows)
A = np.lib.format.open_memmap(os.path.join(out, "images.npy"), "w+", np.uint8, (len(meta), S, S, 3))
ok, info = [], []
for i, r in enumerate(meta.to_dict("records")):
    im = cv2.imread(r["path"], cv2.IMREAD_COLOR)
    if im is None:
        ok.append(False); info.append(""); continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    box = None
    if r["dataset"] == "Chaksu" and os.path.exists(r["disc"]):
        m = cv2.imread(r["disc"], cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != im.shape[:2]:
                m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            n_cc, cc, st_, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8))
            ys, xs = (np.where(cc == 1 + np.argmax(st_[1:, cv2.CC_STAT_AREA])) if n_cc > 1 else (np.array([]), np.array([])))  # largest region only
            if len(ys):
                box = (xs.min(), ys.min(), xs.max(), ys.max())
    elif r["dataset"] == "PAPILA" and r["disc"]:
        pts = np.concatenate([np.loadtxt(c) for c in r["disc"].split(";")])
        box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    if box is not None:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half = 1.0 * max(box[2] - box[0], box[3] - box[1])
        x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
        pad = max(0, -x0, -y0, x1 - im.shape[1], y1 - im.shape[0])
        if pad:
            im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0); x0 += pad; x1 += pad; y0 += pad; y1 += pad
        im = im[y0:y1, x0:x1]
        info.append("disc_crop")
    else:
        info.append("as_provided" if r["dataset"] == "RIMONE" else "NO_DISC_FULL_IMAGE")
    A[i] = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
    ok.append(True)
A.flush()
meta["ok"], meta["crop"] = ok, info
meta.drop(columns=["path"]).to_csv(os.path.join(out, "meta.csv"), index=False)
print(meta.groupby(["dataset", "camera", "split", "label"]).size().to_string())
print(meta.groupby(["dataset", "crop"]).size().to_string())
print("unlabelled Chaksu:", int(((meta.dataset == "Chaksu") & (meta.label == -9)).sum()), "| unreadable:", int((~meta.ok).sum()))
print("PREPROCESS_DONE")
