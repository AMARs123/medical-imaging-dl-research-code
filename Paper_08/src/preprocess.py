"""Paper 08 preprocessing: TrashNet photos and TACO object crops, five materials, 224 x 224 arrays.

Usage: python preprocess.py <Paper_08 dir>
Classes: cardboard, glass, metal, paper, plastic. TrashNet's 'trash' class is not used (a heterogeneous catch-all).
TACO categories are mapped to a material only when the material is unambiguous (MAP below); composites and mixed or
non-recyclable items (drink cartons, paper cups, blister packs, crisp packets, foam, food waste, rope, shoes, cigarettes,
unlabelled litter, batteries) are excluded. Each TACO object is cropped from its photograph with a 15% margin around the
bounding box, padded to a square with the image's mean colour, and resized; objects whose box is smaller than 32 px on
its shorter side (in the original resolution) are excluded. For the 670 TACO photographs that were available only in the
640-px Flickr version, box coordinates are scaled by the size ratio. 'group' = source photograph (crops of one photo stay
together in splits). TrashNet images: whole photo resized (objects on a plain background).
Outputs: data/processed/images.npy (uint8 [N,224,224,3]), meta.csv, excluded_categories.csv
"""
import json
import os
import sys
import zipfile
import io

import numpy as np
import pandas as pd
from PIL import Image

R = sys.argv[1]
raw = os.path.join(R, "data", "raw")
out = os.path.join(R, "data", "processed"); os.makedirs(out, exist_ok=True)
S = 224
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
MAP = {
    "cardboard": ["Toilet tube", "Other carton", "Egg carton", "Corrugated carton", "Meal carton", "Pizza box"],
    "glass": ["Glass bottle", "Broken glass", "Glass cup", "Glass jar"],
    "metal": ["Aluminium foil", "Metal bottle cap", "Food Can", "Aerosol", "Drink can", "Metal lid", "Pop tab", "Scrap metal"],
    "paper": ["Magazine paper", "Tissues", "Wrapping paper", "Normal paper", "Paper bag", "Paper straw"],
    "plastic": ["Other plastic bottle", "Clear plastic bottle", "Plastic bottle cap", "Disposable plastic cup", "Other plastic cup",
                "Plastic lid", "Other plastic", "Plastic film", "Six pack rings", "Garbage bag", "Other plastic wrapper",
                "Single-use carrier bag", "Polypropylene bag", "Spread tub", "Tupperware", "Disposable food container",
                "Other plastic container", "Plastic glooves", "Plastic utensils", "Squeezable tube", "Plastic straw"]}
name2cls = {n: c for c, ns in MAP.items() for n in ns}


def square(im):
    w, h = im.size; s = max(w, h)
    bg = Image.new("RGB", (s, s), tuple(int(v) for v in np.asarray(im).reshape(-1, 3).mean(0)))
    bg.paste(im, ((s - w) // 2, (s - h) // 2))
    return bg.resize((S, S), Image.BICUBIC)


imgs, rows = [], []
z = zipfile.ZipFile(os.path.join(raw, "trashnet", "data", "dataset-resized.zip"))
for n in sorted(z.namelist()):
    parts = n.split("/")
    if len(parts) == 3 and parts[0] == "dataset-resized" and parts[1] in CLASSES and n.lower().endswith(".jpg"):
        im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        imgs.append(np.asarray(square(im)))
        rows.append(dict(source="TrashNet", cls=parts[1], label=CLASSES.index(parts[1]), group="tn:" + parts[2], image=parts[2], taco_category=""))
print("TrashNet", len(rows), flush=True)

a = json.load(open(os.path.join(raw, "taco", "data", "annotations.json")))
cats = {c["id"]: c["name"] for c in a["categories"]}
ims = {i["id"]: i for i in a["images"]}
excl = {}
cache = {}
for an in a["annotations"]:
    nm = cats[an["category_id"]]
    if nm not in name2cls:
        excl[nm] = excl.get(nm, 0) + 1; continue
    info = ims[an["image_id"]]
    x, y, w, h = an["bbox"]
    if min(w, h) < 32:
        excl["(box < 32 px) " + nm] = excl.get("(box < 32 px) " + nm, 0) + 1; continue
    fn = os.path.join(raw, "taco", "data", info["file_name"])
    if fn not in cache:
        cache.clear()
        try:
            from PIL import ImageOps
            cache[fn] = ImageOps.exif_transpose(Image.open(fn)).convert("RGB")
        except Exception:
            cache[fn] = None
    im = cache[fn]
    if im is None:
        excl["(unreadable image)"] = excl.get("(unreadable image)", 0) + 1; continue
    sx, sy = im.width / info["width"], im.height / info["height"]
    if abs(sx - sy) > 0.05:  # orientation mismatch between annotation and file: swap
        sx, sy = im.width / info["height"], im.height / info["width"]
    m = 0.15
    box = (max(0, (x - m * w) * sx), max(0, (y - m * h) * sy), min(im.width, (x + w * (1 + m)) * sx), min(im.height, (y + h * (1 + m)) * sy))
    crop = im.crop(tuple(int(round(v)) for v in box))
    imgs.append(np.asarray(square(crop)))
    c = name2cls[nm]
    rows.append(dict(source="TACO", cls=c, label=CLASSES.index(c), group="taco:%d" % an["image_id"], image=info["file_name"],
                     taco_category=nm, scale=round(sx, 4)))
print("TACO crops", sum(r["source"] == "TACO" for r in rows), flush=True)
np.save(os.path.join(out, "images.npy"), np.stack(imgs))
meta = pd.DataFrame(rows); meta.to_csv(os.path.join(out, "meta.csv"), index=False)
pd.DataFrame(sorted(excl.items(), key=lambda kv: -kv[1]), columns=["category", "objects"]).to_csv(os.path.join(out, "excluded_categories.csv"), index=False)
print(meta.groupby(["source", "cls"]).size().unstack(0))
print("TACO photos used:", meta[meta.source == "TACO"].group.nunique(), "| excluded objects:", sum(excl.values()))
print("PREPROCESS_DONE")
