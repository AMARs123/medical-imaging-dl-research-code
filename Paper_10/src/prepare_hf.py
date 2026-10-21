"""Paper 10: convert the Camelyon17-WILDS labelled splits from the Hugging Face mirror jxie/camelyon17 (parquet, PNG patches)
into uint8 arrays, and check them against the official WILDS split sizes (Koh et al. 2021, Sec. E.2: 302,436 training,
33,560 ID validation, 34,904 OOD validation, 85,054 OOD test patches).
The official CodaLab download was unreachable from the server on 17 Oct 2026; the mirror carries the official split names.
Hospital, patient and slide identifiers are not in the mirror; the analysis uses only the split (train / ID val / OOD val / test).
Writes <out>/{train,id_val,ood_val,test}_x.npy (N x 96 x 96 x 3 uint8), _y.npy (uint8), and summary.json.
Usage: python prepare_hf.py <parquet dir> <out dir>
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
SPL = {"train": "id_train", "id_val": "id_val", "ood_val": "ood_val", "test": "ood_test"}
OFFICIAL = {"train": 302436, "id_val": 33560, "ood_val": 34904, "test": 85054}
summ = {}
for name, pref in SPL.items():
    files = sorted(glob.glob(os.path.join(src, pref + "-*.parquet")))
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    X = np.lib.format.open_memmap(os.path.join(out, name + "_x.npy"), mode="w+", dtype=np.uint8, shape=(n, 96, 96, 3))
    Y = np.zeros(n, np.uint8)
    k = 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["image", "label"]):
            imgs, labs = b.column("image").to_pylist(), b.column("label").to_pylist()
            for im, lab in zip(imgs, labs):
                a = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                assert a.shape == (96, 96, 3), a.shape
                X[k], Y[k] = a, lab
                k += 1
    X.flush()
    np.save(os.path.join(out, name + "_y.npy"), Y)
    summ[name] = dict(n=int(n), official=OFFICIAL[name], matches=int(n) == OFFICIAL[name], tumour_share=float(Y.mean()), files=[os.path.basename(f) for f in files])
    print(name, summ[name], flush=True)
json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
print("PREPARE_DONE", all(v["matches"] for v in summ.values()))
