"""Data and plan tables of Paper 06 and the matching numbers (paper/numbers_data.tex), from stored files only.
The PlantVillage splits are rebuilt with exactly the code of train.py (split seed 0), so the counts match every run.
Usage: python make_tables_data.py <Paper_06 dir>"""
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1]
T = os.path.join(R, "paper", "tables")
os.makedirs(T, exist_ok=True)
idx = pd.read_csv(os.path.join(R, "data", "index.csv"), keep_default_na=False)
pv = idx[idx.source == "PV"].copy()
pdd = idx[idx.source == "PD"].copy()
known = pv.has_leaf_id.astype(str) == "True"


def split(kind):
    rs = np.random.default_rng(0)
    part = {}
    for c, g in pv.groupby("label"):
        units = g.group.unique() if kind == "leaf" else g.index.values
        units = units[rs.permutation(len(units))]
        n = len(units)
        for k, u in enumerate(units):
            part[(c, u)] = "train" if k < 0.7 * n else ("val" if k < 0.8 * n else "test")
    return pd.Series([part[(r.label, r.group if kind == "leaf" else i)] for i, r in pv.iterrows()], index=pv.index)


def fmt(n):
    return "{:,}".format(int(n))


nums = {}
nums["nClasses"] = str(idx.label.max() + 1)
nums["nPV"] = fmt(len(pv))
nums["nPVLeafKnown"] = fmt(known.sum())
nums["nPVLeafKnownPct"] = "%.1f" % (100 * known.mean())
nums["nPVLeafUnknown"] = fmt((~known).sum())
nums["nPVLeaves"] = fmt(pv[known].group.nunique())
nums["nPVPerLeaf"] = "%.1f" % pv[known].groupby("group").size().mean()
nums["nPVPerLeafMax"] = fmt(pv[known].groupby("group").size().max())
nums["nPVSeg"] = fmt((pv.seg_path != "").sum())
nums["nPD"] = fmt(len(pdd))
nums["nPDTrain"] = fmt((pdd.split == "train").sum())
nums["nPDTest"] = fmt((pdd.split == "test").sum())
nums["nCrops"] = str(pv.cls.str.split("___").str[0].nunique())
for kind, tag in [("leaf", "Leaf"), ("random", "Rand")]:
    p = split(kind)
    for s in ["train", "val", "test"]:
        nums["n%s%s" % (tag, s.capitalize())] = fmt((p == s).sum())
    te = pv[(p == "test") & known]
    trg = set(pv[(p == "train") & known].group)
    shared = te.group.isin(trg)
    nums["n%sTestShared" % tag] = fmt(shared.sum())
    nums["n%sTestSharedPct" % tag] = "%.1f" % (100 * shared.sum() / max((p == "test").sum(), 1))
cnt = pdd.groupby("label").size()
nums["nPDMin"], nums["nPDMax"] = fmt(cnt.min()), fmt(cnt.max())
cpv = pv.groupby("label").size()
nums["nPVMin"], nums["nPVMax"] = fmt(cpv.min()), fmt(cpv.max())
trc = pdd[pdd.split == "train"].groupby("label").size()
nums["nPDTrainMin"] = fmt(trc.min())
nums["nFewTwenty"] = fmt(np.minimum(trc, 20).sum())
nums["nFewFive"] = fmt(np.minimum(trc, 5).sum())


def name(c):
    crop, dis = c.split("___")
    crop = crop.replace("_(maize)", "").replace("_(including_sour)", "").replace(",_bell", " (bell)").replace("_", " ")
    dis = dis.replace("Cercospora_leaf_spot Gray_leaf_spot", "Grey leaf spot").replace("Tomato_Yellow_Leaf_Curl_Virus", "Yellow leaf curl virus")
    dis = dis.replace("Tomato_mosaic_virus", "Mosaic virus").replace("_", " ").strip()
    return crop, dis[0].upper() + dis[1:].lower()


rows = []
for c, g in idx.groupby("cls"):
    q = g[g.source == "PV"]
    d = g[g.source == "PD"]
    crop, dis = name(c)
    kn = q.has_leaf_id.astype(str) == "True"
    rows.append("%s & %s & %s & %.0f & %s & %s & %s \\\\" % (crop, dis, fmt(len(q)), 100 * kn.mean(), fmt(q[kn].group.nunique()),
                                                     fmt((d.split == "train").sum()), fmt((d.split == "test").sum())))
rows.append("\\midrule\nAll & & %s & %.0f & %s & %s & %s \\\\" % (nums["nPV"], float(nums["nPVLeafKnownPct"]), nums["nPVLeaves"], nums["nPDTrain"], nums["nPDTest"]))
open(os.path.join(T, "tab_data.tex"), "w", encoding="utf8", newline="\n").write(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{The %s classes shared by PlantVillage (laboratory) and PlantDoc (field). Known: share of PlantVillage images whose leaf is identified; Leaves: number of distinct identified leaves. Images without an identified leaf form groups of one, so for classes with 0\%% known the leaf split equals the random split. PlantDoc counts follow its official split.}
\label{tab:data}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrr}
\toprule
Crop & Class & \multicolumn{3}{c}{PlantVillage} & \multicolumn{2}{c}{PlantDoc} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
 & & Images & Known (\%%) & Leaves & Train & Test \\
\midrule
""" % nums["nClasses"] + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
reg = pd.read_csv(os.path.join(R, "experiments", "experiment_registry.csv"), keep_default_na=False)
MN = {"resnet50.a1_in1k": "ResNet-50", "efficientnet_b0.ra_in1k": "EfficientNet-B0", "convnext_tiny.fb_in1k": "ConvNeXt-T",
      "vit_small_patch14_dinov2.lvd142m": "DINOv2 ViT-S"}
PH = {"A": "A: split and backbone (RQ1)", "B": "B: images, augmentation, field labels (RQ2, RQ3)", "C": "C: linear probes (RQ2, RQ3)"}
rows = []
for ph, g in reg.groupby("phase", sort=True):
    models = ", ".join(MN[m] for m in g.model.unique())
    conds = []
    for _, r in g.drop_duplicates(["split", "source", "aug", "fewshot"]).iterrows():
        conds.append("%s/%s/%s/$k$=%s" % (r.split, r.source.replace("color", "colour"), r.aug, r.fewshot))
    rows.append(r"%s & %s & \makecell[l]{%s} & %s & %d \\" % (PH[ph], models, r" \\ ".join(conds),
                                                                          "fine-tuned" if str(g.freeze.iloc[0]) == "0" else "frozen", len(g)))
open(os.path.join(T, "tab_plan.tex"), "w", encoding="utf8", newline="\n").write(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Registered runs, fixed before training; every condition has seeds 0, 1 and 2. Conditions are written split/training images/augmentation/field images per class. In group C the probes with field images ($k=5, 20$) use DINOv2 ViT-S only. Group C was added after the literature check and before any run.}
\label{tab:plan}
\begin{tabular}{>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{3.6cm}llr}
\toprule
Group & Backbones & Conditions & Training & Runs \\
\midrule
""" + "\n\\midrule\n".join(rows) + r"""
\midrule
All & & & & %d \\
\bottomrule
\end{tabular}
\end{table}
""" % len(reg))
nums["nRuns"] = str(len(reg))
nocls = [name(c) for c, g in pv.groupby("cls") if (g.has_leaf_id.astype(str) == "True").sum() == 0]
nums["nClassNoLeaf"] = str(len(nocls))
nums["nImgNoLeafClass"] = fmt(pv[pv.cls.isin([c for c, g in pv.groupby("cls") if (g.has_leaf_id.astype(str) == "True").sum() == 0])].shape[0])
with open(os.path.join(R, "paper", "numbers_data.tex"), "w", encoding="utf8") as f:
    f.write("% generated by src/make_tables_data.py\n")
    for k, v in sorted(nums.items()):
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print("wrote", nums)
