"""PTB-XL+ ECGDeli global fiducial points for every record kept by preprocess.py.

Usage: python fiducials.py <ecg-attribution-testing dir>
Writes data/processed/fiducials_raw.npz (per record: positions converted from 500 Hz to 100 Hz, symbols, labels)
and fiducial_check.csv (file found, number of annotations).
"""
import os
import sys

import numpy as np
import pandas as pd
import wfdb

root = sys.argv[1]
plus = os.path.join(root, "data", "raw", "ptb-xl-plus")
out = os.path.join(root, "data", "processed")
db = pd.read_csv(os.path.join(out, "meta.csv")).set_index("ecg_id")
fid, rows = {}, []
for i, (eid, r) in enumerate(db.iterrows()):
    base = os.path.join(plus, "fiducial_points", "ecgdeli", "%05d" % (eid // 1000 * 1000), "%05d_points_global" % eid)
    row = dict(ecg_id=eid, found=os.path.exists(base + ".atr"))
    if row["found"]:
        a = wfdb.rdann(base, "atr")
        fs = a.fs if a.fs else 500
        row.update(fs=fs, n_ann=len(a.sample))
        fid[str(eid)] = dict(sample=np.round(np.asarray(a.sample) * 100.0 / fs).astype(np.int32),
                             symbol=np.asarray(a.symbol), aux=np.asarray([s.strip() for s in a.aux_note]))
    rows.append(row)
    if i % 5000 == 0:
        print("fiducials", i, flush=True)
np.savez_compressed(os.path.join(out, "fiducials_raw.npz"),
                    **{k + "_" + f: v[f] for k, v in fid.items() for f in ("sample", "symbol", "aux")})
pd.DataFrame(rows).to_csv(os.path.join(out, "fiducial_check.csv"), index=False)
print("FIDUCIALS_DONE", len(fid))
