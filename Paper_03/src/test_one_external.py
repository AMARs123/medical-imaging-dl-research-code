"""Smoke test of prepare_external.py's per-record steps on one record downloaded directly (no zip needed)."""
import os
import re
import sys
import urllib.request

import numpy as np
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import masks_from_points  # noqa: E402
import neurokit2 as nk  # noqa: E402

base = "https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/WFDBRecords/01/010/JS00001"
tmp = "/tmp/_p03one"
recs = {ext: urllib.request.urlopen(base + ext, timeout=60).read() for ext in (".hea", ".mat")}
print(recs[".hea"].decode()[:400])
open(tmp + ".hea", "wb").write(recs[".hea"].replace(b"JS00001", os.path.basename(tmp).encode()))
open(tmp + ".mat", "wb").write(recs[".mat"])
rec = wfdb.rdrecord(tmp)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
print(rec.sig_name, rec.fs, rec.p_signal.shape, rec.units[:2])
x = rec.p_signal[:, [rec.sig_name.index(l) for l in LEADS]].T
y = resample_poly(np.nan_to_num(x), 1, 5, axis=1)[:, :1000]
print("resampled", y.shape, float(np.abs(y).max()))
dx = re.search(r"#\s*Dx:\s*([\d,]+)", recs[".hea"].decode())
print("Dx", dx.group(1) if dx else None)
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
print("segment time shares", mk.mean(1).round(3))
