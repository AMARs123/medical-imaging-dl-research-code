"""Inspect the Figshare brain tumour dataset file by file. Writes an index CSV and a summary JSON.

Nothing here is assumed: every count in the dataset report must come from this script output.
"""
import glob, hashlib, json, os, sys
import h5py
import numpy as np
import pandas as pd

RAW = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rows, errors = [], []
files = sorted(glob.glob(os.path.join(RAW, "mat", "**", "*.mat"), recursive=True), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
for path in files:
    idx = int(os.path.splitext(os.path.basename(path))[0])
    try:
        with h5py.File(path, "r") as f:
            g = f["cjdata"]
            label = int(np.array(g["label"]).squeeze())
            pid_raw = np.array(g["PID"]).squeeze()
            pid = "".join(chr(int(c)) for c in np.atleast_1d(pid_raw)) if pid_raw.size else ""
            img = np.array(g["image"])
            mask = np.array(g["tumorMask"]) if "tumorMask" in g else None
            border = np.array(g["tumorBorder"]) if "tumorBorder" in g else None
        rows.append(dict(
            idx=idx, file=os.path.relpath(path, RAW), label=label, pid=pid,
            height=img.shape[0], width=img.shape[1], dtype=str(img.dtype),
            img_min=float(img.min()), img_max=float(img.max()),
            has_mask=mask is not None, mask_pixels=int(mask.sum()) if mask is not None else -1,
            mask_shape_ok=bool(mask is not None and mask.shape == img.shape),
            mask_fraction=float(mask.mean()) if mask is not None else float("nan"),
            has_border=border is not None,
            md5=hashlib.md5(img.tobytes()).hexdigest(),
        ))
    except Exception as e:  # recorded, never skipped silently
        errors.append(dict(idx=idx, file=path, error=repr(e)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "dataset_index.csv"), index=False)
s = dict(
    files_found=len(files), files_read=len(df), read_errors=errors,
    label_counts=df["label"].value_counts().sort_index().to_dict(),
    n_unique_pid=int(df["pid"].nunique()), n_empty_pid=int((df["pid"].str.strip() == "").sum()),
    pid_examples=df["pid"].drop_duplicates().head(12).tolist(),
    images_per_pid=df.groupby("pid").size().describe().to_dict(),
    pids_with_more_than_one_label=int((df.groupby("pid")["label"].nunique() > 1).sum()),
    patients_per_label={int(k): int(v) for k, v in df.groupby("label")["pid"].nunique().items()},
    image_sizes=df.groupby(["height", "width"]).size().reset_index(name="n").to_dict("records"),
    dtypes=df["dtype"].value_counts().to_dict(),
    missing_mask=int((~df["has_mask"]).sum()), empty_mask=int((df["mask_pixels"] == 0).sum()),
    mask_shape_mismatch=int((~df["mask_shape_ok"]).sum()),
    mask_fraction_by_label=df.groupby("label")["mask_fraction"].describe().round(5).to_dict("index"),
    exact_duplicate_images=int(df["md5"].duplicated().sum()),
    duplicate_groups_spanning_pids=int((df[df["md5"].duplicated(keep=False)].groupby("md5")["pid"].nunique() > 1).sum()),
)
cv = os.path.join(RAW, "cvind.mat")
if os.path.exists(cv):
    try:
        with h5py.File(cv, "r") as f:
            s["cvind"] = {k: dict(shape=list(np.array(f[k]).shape), unique=sorted(set(np.array(f[k]).ravel().tolist()))[:20]) for k in f.keys() if not k.startswith("#")}
            arr = np.array(f[[k for k in f.keys() if not k.startswith("#")][0]]).ravel().astype(int)
    except OSError:
        from scipy.io import loadmat
        m = loadmat(cv); keys = [k for k in m if not k.startswith("__")]
        s["cvind"] = {k: dict(shape=list(m[k].shape), unique=sorted(set(m[k].ravel().tolist()))[:20]) for k in keys}
        arr = m[keys[0]].ravel().astype(int)
    if len(arr) == len(df):
        d2 = df.sort_values("idx").assign(fold=arr)
        d2.to_csv(os.path.join(OUT, "dataset_index.csv"), index=False)
        s["cvind_fold_sizes"] = d2["fold"].value_counts().sort_index().to_dict()
        s["cvind_pids_in_more_than_one_fold"] = int((d2.groupby("pid")["fold"].nunique() > 1).sum())
        s["cvind_label_by_fold"] = d2.groupby(["fold", "label"]).size().unstack(fill_value=0).to_dict("index")
    else:
        s["cvind_length_mismatch"] = [int(len(arr)), int(len(df))]
json.dump(s, open(os.path.join(OUT, "inspection_summary.json"), "w"), indent=1, default=str)
print(json.dumps(s, indent=1, default=str))
