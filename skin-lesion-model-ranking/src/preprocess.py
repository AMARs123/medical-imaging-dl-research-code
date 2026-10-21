"""Unpack HAM10000 and the ISIC 2018 Task 3 test set, check every file, and store 256 x 256 RGB arrays.

Usage: python preprocess.py <skin-lesion-model-ranking dir>
Outputs in data/processed/: ham_images.npy, ham_meta.csv, test_images.npy, test_meta.csv, file_check.csv
The whole image is resized to 256 x 256 (no crop), so no part of a lesion is cut away; most images are
600 x 450, so the aspect ratio changes from 4:3 to 1:1 identically for every image.
A 64-bit difference hash (dHash) of every image is stored for the duplicate audit (inspect_dataset.py).
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES  # noqa: E402

root = sys.argv[1]
raw = os.path.join(root, "data", "raw")
img_dir = os.path.join(raw, "images")
out = os.path.join(root, "data", "processed")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(out, exist_ok=True)
SIZE = 256

for z in ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip", "ISIC2018_Task3_Test_Images.zip"]:
    with zipfile.ZipFile(os.path.join(raw, z)) as f:
        for n in f.namelist():
            if n.lower().endswith(".jpg"):
                dst = os.path.join(img_dir, os.path.basename(n))
                if not os.path.exists(dst):
                    with open(dst, "wb") as o:
                        o.write(f.read(n))
    print("unpacked", z, flush=True)


def dhash(img):
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


checks = []
for part, tab in [("ham", "HAM10000_metadata.tab"), ("test", "ISIC2018_Task3_Test_GroundTruth.tab")]:
    meta = pd.read_csv(os.path.join(raw, tab))
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)})
    assert meta["label"].notna().all(), "unknown diagnosis code"
    # images listed in the label file but absent from the archives are excluded and recorded (2026-10-15: one test
    # image, ISIC_0035068, is listed in ISIC2018_Task3_Test_GroundTruth but not contained in ISIC2018_Task3_Test_Images.zip)
    present = meta["image_id"].apply(lambda i: os.path.exists(os.path.join(img_dir, i + ".jpg")))
    meta[~present].assign(part=part, reason="listed in label file, not in image archive").to_csv(
        os.path.join(out, "%s_excluded.csv" % part), index=False)
    meta = meta[present].reset_index(drop=True)
    arr = np.lib.format.open_memmap(os.path.join(out, "%s_images.npy" % part), "w+", np.uint8, (len(meta), SIZE, SIZE, 3))
    hashes, thumbs = [], []
    for i, iid in enumerate(meta["image_id"]):
        path = os.path.join(img_dir, iid + ".jpg")
        row = dict(part=part, image_id=iid, exists=os.path.exists(path))
        try:
            with Image.open(path) as im:
                im.load()
                row.update(width=im.width, height=im.height, mode=im.mode, ok=True)
                im = im.convert("RGB")
                arr[i] = np.asarray(im.resize((SIZE, SIZE), Image.BICUBIC))
                hashes.append(dhash(im))
                thumbs.append(np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), np.uint8))
        except Exception as e:  # recorded, never silently skipped
            row.update(ok=False, error=repr(e))
            hashes.append(-1)
            thumbs.append(np.zeros((32, 32), np.uint8))
        checks.append(row)
        if i % 2000 == 0:
            print(part, i, flush=True)
    arr.flush()
    meta["dhash"] = [format(h, "016x") if h >= 0 else "" for h in hashes]
    meta["array_index"] = np.arange(len(meta))
    meta.to_csv(os.path.join(out, "%s_meta.csv" % part), index=False)
    np.save(os.path.join(out, "%s_thumbs32.npy" % part), np.stack(thumbs))
pd.DataFrame(checks).to_csv(os.path.join(out, "file_check.csv"), index=False)
print("PREPROCESS_DONE")
