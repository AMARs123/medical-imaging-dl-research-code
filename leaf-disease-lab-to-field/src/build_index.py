"""leaf-disease-lab-to-field: build the image index (no image arrays; images are read from disk during training).

Usage: python build_index.py <leaf-disease-lab-to-field dir>
Classes: the 27 classes present in both PlantVillage (lab photographs) and PlantDoc (field images from the web).
The mapping below is fixed by hand from the class names; 'Tomato two spotted spider mites leaf' (2 PlantDoc images)
is left out. PlantVillage leaf identity: leaf-map.json maps the lower-cased file-name suffix (after '___', without the
extension) to '<class>:::<leaf number>'; images of one leaf share a group. Images without an entry get their own group
and the coverage is reported.
Outputs: data/index.csv (source, split, path, seg_path, class, label, group), data/class_map.csv, data/index_summary.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
raw = os.path.join(R, "data", "raw")
MAP = {  # PlantDoc folder -> PlantVillage folder
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple rust leaf": "Apple___Cedar_apple_rust", "Apple leaf": "Apple___healthy",
    "Blueberry leaf": "Blueberry___healthy", "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "grape leaf black rot": "Grape___Black_rot", "grape leaf": "Grape___healthy",
    "Peach leaf": "Peach___healthy", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot", "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Potato leaf early blight": "Potato___Early_blight", "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": "Squash___Powdery_mildew", "Strawberry leaf": "Strawberry___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato Early blight leaf": "Tomato___Early_blight", "Tomato leaf": "Tomato___healthy",
    "Tomato leaf late blight": "Tomato___Late_blight", "Tomato mold leaf": "Tomato___Leaf_Mold", "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus", "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"}
classes = sorted(MAP.values())
lab = {c: i for i, c in enumerate(classes)}
leafmap = json.load(open(os.path.join(raw, "plantvillage", "leaf-map.json")))
rows = []
for c in classes:
    for f in sorted(glob.glob(os.path.join(raw, "plantvillage", "raw", "color", c, "*"))):
        b = os.path.basename(f)
        key = b.split("___", 1)[-1].rsplit(".", 1)[0].lower()
        seg = os.path.join(raw, "plantvillage", "raw", "segmented", c, b.rsplit(".", 1)[0] + "_final_masked.jpg")
        v = leafmap.get(key)
        grp = None
        if v:
            cls_, num = v[0].split(":::")
            grp = "%s:::%s" % (cls_, num) if cls_ == c else None
        rows.append(dict(source="PV", split="", path=os.path.relpath(f, R), seg_path=os.path.relpath(seg, R) if os.path.exists(seg) else "",
                         cls=c, label=lab[c], group=grp or ("img:" + key), has_leaf_id=grp is not None))
for split in ["train", "test"]:
    for pd_cls, c in MAP.items():
        for f in sorted(glob.glob(os.path.join(raw, "plantdoc", split, pd_cls, "*"))):
            rows.append(dict(source="PD", split=split, path=os.path.relpath(f, R), seg_path="", cls=c, label=lab[c],
                             group="pd:" + os.path.basename(f), has_leaf_id=False))
idx = pd.DataFrame(rows)
os.makedirs(os.path.join(R, "data"), exist_ok=True)
idx.to_csv(os.path.join(R, "data", "index.csv"), index=False)
pd.DataFrame([dict(plantdoc=k, plantvillage=v, label=lab[v]) for k, v in MAP.items()]).sort_values("label").to_csv(os.path.join(R, "data", "class_map.csv"), index=False)
pv = idx[idx.source == "PV"]
summ = dict(classes=len(classes), pv_images=int(len(pv)), pv_with_leaf_id=int(pv.has_leaf_id.sum()), pv_with_segmented=int((pv.seg_path != "").sum()),
            pv_leaf_groups=int(pv[pv.has_leaf_id].group.nunique()), pv_images_per_leaf=float(pv[pv.has_leaf_id].groupby("group").size().mean()),
            pd_train=int(((idx.source == "PD") & (idx.split == "train")).sum()), pd_test=int(((idx.source == "PD") & (idx.split == "test")).sum()),
            per_class=idx.groupby(["cls", "source"]).size().unstack(fill_value=0).to_dict("index"))
json.dump(summ, open(os.path.join(R, "data", "index_summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "per_class"}, indent=1))
print(pv.groupby("cls").agg(n=("path", "size"), leaf=("has_leaf_id", "mean"), groups=("group", "nunique")).round(2).to_string())
