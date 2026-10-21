"""Result tables of skin-lesion-model-ranking, generated from results/statistics (stats.py) and results/all_runs.csv only.
Usage: python make_tables_results.py <skin-lesion-model-ranking dir>
  tab_inflation  RQ1: image-split vs lesion-split means per metric, inflation over the nine backbones, Wilcoxon + Holm
  tab_seen       RQ1: accuracy on test images whose lesion was / was not in the training folds (image split), per class
  tab_scores     RQ2: balanced accuracy per backbone in every setting (mean +- SD over runs)
  tab_kendall    RQ2: Kendall tau between settings with bootstrap intervals
  tab_remedy     RQ3: paired differences of each remedy to cross-entropy on the lesion split (pooled over 3 backbones x 5 folds)
  tab_remedy_isic RQ3: paired image-level bootstrap on the ISIC 2018 test set, per backbone
Missing inputs give a PENDING row; nothing is typed by hand. At most 8 columns per table."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, MODELS, NAMES, SHORT  # noqa: E402

R = sys.argv[1]
S = os.path.join(R, "results", "statistics")
T = os.path.join(R, "paper", "tables")
os.makedirs(T, exist_ok=True)
MET = {"balanced_accuracy": "Balanced accuracy", "accuracy": "Accuracy", "macro_f1": "Macro-F1", "mcc": "MCC", "macro_auc": "Macro AUC",
       "mel_auc": "Melanoma AUC", "ece": "Calibration error (ECE)"}
SET = {"image_cv": "Image split, test folds", "lesion_cv": "Lesion split, test folds", "isic_lesion_trained": "ISIC 2018 test (lesion-trained)",
       "isic_image_trained": "ISIC 2018 test (image-trained)", "isic_internal": "ISIC 2018 internal part", "isic_external": "ISIC 2018 external part"}
REM = {"wce": "Weighted CE", "focal": "Focal loss", "la": "Logit-adjusted loss", "sampler": "Balanced sampling", "posthoc_la": "Post-hoc adjustment"}


def write(name, s):
    open(os.path.join(T, name + ".tex"), "w", encoding="utf8", newline="\n").write(s)
    print("wrote", name)


def pv(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else "%.3f" % p


def table(name, caption, label, colspec, header, rows, size="small"):
    write(name, "\\begin{table}[!htbp]\n\\centering\\%s\n\\caption{%s}\n\\label{%s}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
          % (size, caption, label, colspec, header, "\n".join(rows) if rows else "\\multicolumn{%d}{c}{PENDING} \\\\" % (colspec.count("l") + colspec.count("r") + colspec.count("c"))))


def read(f):
    p = os.path.join(S, f)
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------------------------------------------------------- RQ1 inflation
inf = read("rq1_inflation_summary.csv")
bym = read("rq1_inflation_by_model.csv")
rows = []
if inf is not None and bym is not None:
    means = bym.groupby("metric")[["image_cv", "lesion_cv"]].mean()
    for _, r in inf.set_index("metric").reindex(list(MET)).dropna(how="all").iterrows():
        rows.append("%s & %.3f & %.3f & %+.3f (%+.3f to %+.3f) & %d/%d & %s \\\\" % (
            MET[r.name], means.loc[r.name, "image_cv"], means.loc[r.name, "lesion_cv"],
            r.mean_inflation, r["min"], r["max"], r.models_inflated, r.n_models, pv(r.p_holm)))
table("tab_inflation", "Inflation caused by the image split (RQ1): mean over the five test folds and the nine backbones under each split, the "
      "difference image minus lesion (mean and range over backbones), the number of backbones with a positive difference, and the Wilcoxon "
      "signed-rank test over backbones, Holm-adjusted over metrics. For ECE a positive difference means worse calibration under the image split.",
      "tab:inflation", "lrrcrr", "Metric & Image split & Lesion split & Inflation (range) & Backbones & $p_{\\mathrm{Holm}}$", rows)

# ---------------------------------------------------------------- RQ1 seen vs unseen lesions
seen = read("rq1_seen_unseen.csv")
rows = []
if seen is not None:
    for c in CLASSES + ["all"]:
        q = seen[seen.dx == c].set_index("lesion_in_train")
        if True in q.index and False in q.index:
            rows.append("%s & %s & %.3f & %s & %.3f & %+.3f \\\\" % ("All classes" if c == "all" else NAMES[c], "{:,}".format(int(q.loc[True, "n"])),
                                                                  q.loc[True, "accuracy"], "{:,}".format(int(q.loc[False, "n"])),
                                                                  q.loc[False, "accuracy"], q.loc[True, "accuracy"] - q.loc[False, "accuracy"]))
table("tab_seen", "Recall of image-split test images whose lesion had another image in the training folds (seen) and of the others (unseen), pooled over "
      "the nine backbones and five folds. $n$ counts test predictions. The pooled row mixes classes: most naevi are unseen and easy, so it "
      "reverses the within-class pattern.", "tab:seen", "lrrrrr",
      "Class & $n$ seen & Recall seen & $n$ unseen & Recall unseen & Difference", rows)

# ---------------------------------------------------------------- RQ2 scores per backbone
sc = read("rq2_scores.csv")
rows = []
if sc is not None:
    sc = sc[sc.metric == "balanced_accuracy"]
    cols = ["lesion_cv", "image_cv", "isic_lesion_trained", "isic_internal", "isic_external"]
    for fam in ["CNN", "Transformer", "Hybrid"]:
        for m in [k for k, v in MODELS.items() if v == fam]:
            q = sc[sc.model == m].set_index("setting")
            cells = []
            for c in cols:
                if c in q.index:
                    r = q.loc[c]
                    cells.append("%.3f $\\pm$ %.3f" % (r["mean"], r["sd"]) if r["n"] > 1 else "%.3f" % r["mean"])
                else:
                    cells.append("PENDING")
            rows.append("%s & %s & %s \\\\" % (fam if m == [k for k, v in MODELS.items() if v == fam][0] else "", SHORT[m], " & ".join(cells)))
        rows.append("\\addlinespace")
    rows = rows[:-1]
table("tab_scores", "Balanced accuracy per backbone (mean $\\pm$ SD over runs): lesion split (5 folds $\\times$ 3 seeds), image split (5 folds), and the "
      "ISIC 2018 test set (lesion-trained models) with its internal and external parts.", "tab:scores", "llccccc",
      "Family & Backbone & Lesion split $\\uparrow$ & Image split $\\uparrow$ & ISIC 2018 $\\uparrow$ & ISIC internal $\\uparrow$ & ISIC external $\\uparrow$",
      rows, size="footnotesize")

# ---------------------------------------------------------------- RQ2 Kendall
kt = read("rq2_kendall.csv")
rows = []
if kt is not None:
    for _, r in kt[kt.metric == "balanced_accuracy"].iterrows():
        rows.append("%s & %s & %.2f & [%.2f, %.2f] & %s & %s \\\\" % (SET.get(r.a, r.a), SET.get(r.b, r.b), r.tau, r.ci_low, r.ci_high, r.top_a, r.top_b))
table("tab_kendall", "Agreement between the rankings of the nine backbones by balanced accuracy (Kendall's $\\tau$, Eq.~\\ref{eq:kendall}) with 95\\% "
      "bootstrap intervals over runs (Eq.~\\ref{eq:boot}), and the best backbone in each setting.", "tab:kendall",
      ">{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{3.6cm}rcll",
      "Setting A & Setting B & $\\tau$ & 95\\% CI & Best in A & Best in B", rows, size="footnotesize")

# ---------------------------------------------------------------- RQ3 remedies on the lesion split
r3 = read("rq3_remedies_pooled.csv")
rows = []
if r3 is not None:
    t = r3[r3.set == "test"]
    for rem in REM:
        q = t[t.remedy == rem].set_index("metric")
        if "balanced_accuracy" not in q.index:
            continue
        cells = []
        for met in ["balanced_accuracy", "accuracy", "macro_f1", "ece"]:
            if met in q.index:
                r = q.loc[met]
                cells.append("%+.3f (%d/%d)" % (r.mean_diff, r.pairs_improved, r.n_pairs))
            else:
                cells.append("--")
        b = q.loc["balanced_accuracy"]
        rows.append("%s & %s & %s & %.2f \\\\" % (REM[rem], " & ".join(cells), pv(b.get("p_holm_within_metric", np.nan)), b.get("rank_biserial", np.nan)))
table("tab_remedy", "Class-imbalance remedies on the lesion split (RQ3): mean paired difference to cross-entropy over 3 backbones $\\times$ 5 folds, and in "
      "brackets the pairs in which the remedy was higher (for ECE, higher is worse). $p_{\\mathrm{Holm}}$ and $r_{\\mathrm{rb}}$ refer to balanced accuracy "
      "(Wilcoxon signed-rank, Holm over the five remedies, Eqs.~\\ref{eq:holm}).", "tab:remedy", "lccccrr",
      "Remedy & Balanced accuracy & Accuracy & Macro-F1 & ECE & $p_{\\mathrm{Holm}}$ & $r_{\\mathrm{rb}}$", rows, size="footnotesize")

# ---------------------------------------------------------------- RQ3 ISIC bootstrap
bt = read("rq3_isic_bootstrap.csv")
rows = []
if bt is not None:
    for rem in REM:
        q = bt[bt.remedy == rem].set_index("model")
        if len(q) == 0:
            continue
        cells = []
        for m in ["convnext_tiny.fb_in1k", "swin_tiny_patch4_window7_224.ms_in1k", "coatnet_0_rw_224.sw_in1k"]:
            if m in q.index:
                r = q.loc[m]
                cells.append("%+.3f [%+.3f, %+.3f]" % (r.isic_bacc_diff, r.ci_low, r.ci_high))
            else:
                cells.append("--")
        rows.append("%s & %s \\\\" % (REM[rem], " & ".join(cells)))
table("tab_remedy_isic", "Remedies on the ISIC 2018 test set: difference in balanced accuracy to cross-entropy (mean of the five fold models) with 95\\% "
      "paired bootstrap intervals over the 1,511 test images (Eq.~\\ref{eq:pairboot}).", "tab:remedy_isic", "lccc",
      "Remedy & ConvNeXt-T & Swin-T & CoAtNet-0", rows, size="footnotesize")
