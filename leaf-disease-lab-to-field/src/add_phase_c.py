"""Append group C to the leaf-disease-lab-to-field registry (16 Oct 2026, before any leaf-disease-lab-to-field run had started).

Reason: a study published in 2026 (Nguyen et al., Smart Agricultural Technology 102531) reported that a frozen DINOv2 backbone
with a linear probe transfers from PlantVillage to PlantDoc much better than fine-tuned networks. Group C tests this under our
leaf-grouped evaluation and extends it with field images:
  C (18): linear probe (frozen backbone, learning rate 1e-3) of ConvNeXt-T and DINOv2 ViT-S on colour and mixed photos,
          and of DINOv2 ViT-S with 5 and 20 PlantDoc training images per class; leaf split; seeds 0-2.
Existing rows are not changed; new rows get new IDs. A copy of the registry before the change is kept.
Usage: python add_phase_c.py <leaf-disease-lab-to-field dir>"""
import os
import shutil
import sys

import pandas as pd

root = sys.argv[1]
reg = os.path.join(root, "experiments", "experiment_registry.csv")
df = pd.read_csv(reg)
assert "C" not in set(df.phase), "group C already registered"
shutil.copy(reg, reg.replace(".csv", "_before_C.csv"))
rows = []
for m in ["convnext_tiny.fb_in1k", "vit_small_patch14_dinov2.lvd142m"]:
    for src in ["color", "mixed"]:
        for s in range(3):
            rows.append(dict(phase="C", model=m, split="leaf", source=src, aug="base", fewshot=0, seed=s, freeze=1, lr=1e-3))
for k in [5, 20]:
    for s in range(3):
        rows.append(dict(phase="C", model="vit_small_patch14_dinov2.lvd142m", split="leaf", source="color", aug="base", fewshot=k, seed=s,
                         freeze=1, lr=1e-3))
new = pd.DataFrame(rows)
start = len(df) + 1
new.insert(0, "exp_id", ["EXP%03d" % (start + i) for i in range(len(new))])
out = pd.concat([df, new], ignore_index=True)
out["freeze"] = out["freeze"].fillna(0).astype(int)
out.to_csv(reg, index=False)
print("added", len(new), "runs:", new.exp_id.iloc[0], "-", new.exp_id.iloc[-1])
