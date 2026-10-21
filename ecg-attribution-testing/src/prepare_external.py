"""External test set for ecg-attribution-testing: the 12-lead ECG database of Chapman University, Shaoxing People's Hospital and
Ningbo First Hospital (PhysioNet ecg-arrhythmia 1.0.0, CC BY 4.0; 45,152 10-s ECGs at 500 Hz with SNOMED CT codes).

Steps (no setting is tuned on these data):
 1. read every header (.hea) from the archive; map SNOMED codes to the five PTB-XL superclasses:
      MI   164865005 (myocardial infarction, all walls)
      STTC 428750005 (ST-T change), 429622005 (ST depression), 164930006 (ST elevation), 164931005 (ST tilt up),
           164934002 (T-wave change), 59931005 (T-wave inversion)
      CD   270492004, 195042002, 54016002, 28189009, 27885002, 233917008 (AV blocks), 164909002 (LBBB),
           59118001 (RBBB), 698252002 (intraventricular block), 74390002 (WPW), 195060002 (pre-excitation)
      HYP  164873001 (LVH), 89792004 (RVH), 446358003 (right atrial hypertrophy)
      NORM sinus rhythm (426783006) and no other code
    Records with no superclass are dropped, as in the PTB-XL benchmark.
 2. draw a label-balanced subset with seed 0: every MI record, and up to N records per other superclass
    (records chosen for one class keep all their labels);
 3. extract the chosen .hea/.mat files, verify their SHA-256 against SHA256SUMS.txt, read them with wfdb (mV),
    order the leads as in PTB-XL (I, II, III, aVR, aVL, aVF, V1-V6), resample 500 -> 100 Hz (polyphase, factor 5);
 4. delineate lead II at 500 Hz with NeuroKit2 (discrete wavelet method) and build P, QRS, ST-T and other masks
    at 100 Hz with the same pairing rules as for PTB-XL (segments.masks_from_points).
Writes data/external/processed/: signals.npy [n, 12, 1000] float32, masks.npy [n, 4, 1000] bool, meta.csv.
Usage: python prepare_external.py <ecg-attribution-testing dir> [per_class=700]
"""
import hashlib
import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import masks_from_points  # noqa: E402

R = sys.argv[1]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 700
E = os.path.join(R, "data", "external")
OUT = os.path.join(E, "processed")
os.makedirs(OUT, exist_ok=True)
MAP = {"MI": {"164865005"},
       "STTC": {"428750005", "429622005", "164930006", "164931005", "164934002", "59931005"},
       "CD": {"270492004", "195042002", "54016002", "28189009", "27885002", "233917008", "164909002", "59118001", "698252002",
              "74390002", "195060002"},
       "HYP": {"164873001", "89792004", "446358003"}}
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

z = zipfile.ZipFile(os.path.join(E, "ecg-arrhythmia-1.0.0.zip"))
names = z.namelist()
hea = [n for n in names if n.endswith(".hea") and "WFDBRecords" in n]
rows = []
for n in hea:
    txt = z.read(n).decode("utf8", "ignore")
    dx = re.search(r"#\s*Dx:\s*([\d,]+)", txt)
    codes = set(dx.group(1).split(",")) if dx else set()
    age = re.search(r"#\s*Age:\s*(\S+)", txt)
    sex = re.search(r"#\s*Sex:\s*(\S+)", txt)
    lab = {k: int(bool(codes & v)) for k, v in MAP.items()}
    lab["NORM"] = int(codes == {"426783006"})
    rows.append(dict(name=n[:-4], codes=",".join(sorted(codes)), age=age.group(1) if age else "", sex=sex.group(1) if sex else "", **lab))
H = pd.DataFrame(rows)
H["n_super"] = H[SUPER].sum(1)
print("headers", len(H), "with a superclass", int((H.n_super > 0).sum()), H[SUPER].sum().to_dict(), flush=True)
n_headers, n_with_super = len(H), int((H.n_super > 0).sum())
H = H[H.n_super > 0].reset_index(drop=True)
rng = np.random.default_rng(0)
chosen = set()
for c in ["MI", "HYP", "CD", "STTC", "NORM"]:  # rarest first, so their records are not crowded out
    cand = H.index[(H[c] == 1) & ~H.index.isin(list(chosen))].values
    k = len(cand) if c == "MI" else min(PER, len(cand))
    chosen |= set(rng.choice(cand, k, replace=False).tolist())
S = H.loc[sorted(chosen)].reset_index(drop=True)
print("subset", len(S), S[SUPER].sum().to_dict(), flush=True)

sums = {}
for line in open(os.path.join(E, "SHA256SUMS.txt")):
    h, f = line.split()[:2]
    sums[f.lstrip("*")] = h
prefix = names[0].split("/")[0] + "/" if names[0].count("/") else ""

import neurokit2 as nk  # noqa: E402

sig, masks, ok, bad_hash, checked = [], [], [], 0, 0
for i, r in S.iterrows():
    recs = {}
    for ext in (".hea", ".mat"):
        b = z.read(r["name"] + ext)
        full = r["name"] + ext
        rel = full[full.index("WFDBRecords/"):]  # SHA256SUMS paths start at WFDBRecords/
        if rel in sums:
            checked += 1
            if hashlib.sha256(b).hexdigest() != sums[rel]:
                bad_hash += 1
        recs[ext] = b
    tmp = os.path.join(OUT, "_tmp")
    open(tmp + ".hea", "wb").write(recs[".hea"].replace(os.path.basename(r["name"]).encode(), b"_tmp"))
    open(tmp + ".mat", "wb").write(recs[".mat"])
    rec = wfdb.rdrecord(tmp)
    x = rec.p_signal[:, [rec.sig_name.index(l) for l in LEADS]].T.astype(np.float64)  # [12, 5000] mV
    x = np.nan_to_num(x)
    sig.append(resample_poly(x, 1, 5, axis=1)[:, :1000].astype(np.float32))
    try:
        clean = nk.ecg_clean(x[1], sampling_rate=500)
        _, rp = nk.ecg_peaks(clean, sampling_rate=500)
        _, w = nk.ecg_delineate(clean, rp["ECG_R_Peaks"], sampling_rate=500, method="dwt")
        pts, aux = [], []
        for key, name in [("ECG_P_Onsets", "p-wave onset"), ("ECG_P_Offsets", "p-wave offset"), ("ECG_R_Onsets", "QRS onset"),
                          ("ECG_R_Offsets", "QRS offset"), ("ECG_T_Offsets", "t-wave offset")]:
            v = np.asarray([p for p in w.get(key, []) if p == p], float)
            pts += list(np.round(v / 5).astype(int))
            aux += [name] * len(v)
        mk = masks_from_points(np.asarray(pts, int), np.asarray(aux))
        ok.append(bool(mk[1].any()))
    except Exception:  # noqa: BLE001
        mk = np.zeros((4, 1000), bool)
        mk[3] = True
        ok.append(False)
    masks.append(mk)
    if i % 500 == 0:
        print("records", i, "of", len(S), flush=True)
for f in (tmp + ".hea", tmp + ".mat"):
    os.remove(f)
S["fiducials_ok"] = ok
S["array_index"] = np.arange(len(S))
np.save(os.path.join(OUT, "signals.npy"), np.stack(sig))
np.save(os.path.join(OUT, "masks.npy"), np.stack(masks))
S.to_csv(os.path.join(OUT, "meta.csv"), index=False)
import json  # noqa: E402

json.dump(dict(headers=n_headers, with_superclass=n_with_super, records=int(len(S)), files_checked=checked, hash_mismatches=bad_hash, fiducials_ok=int(np.sum(ok)), per_class=PER),
          open(os.path.join(OUT, "prepare_info.json"), "w"), indent=1)
print("PREPARE_EXTERNAL_DONE", len(S), "fiducials ok", int(np.sum(ok)), "files checked", checked, "hash mismatches", bad_hash)
if checked < 2 * len(S) or bad_hash:
    sys.exit("checksum verification incomplete or failed")
