"""Shared matplotlib style for all journal-paper figures (Papers 02-10).

Categorical slots (validated with the dataviz palette validator, light mode, white surface: lightness band, chroma,
CVD separation >= 8 Delta-E for adjacent pairs, normal-vision floor all PASS; slot 3 is below 3:1 contrast, so every
use carries a direct text label or a legend):
  slot 1 blue #2a78d6, slot 2 orange #eb6834, slot 3 aqua #1baf7a, slot 4 yellow #eda100, slot 5 magenta #e87ba4
Sequential: one blue ramp (light -> dark). Diverging: blue <-> orange with a grey midpoint.
Text is always dark grey, never a series colour. Thin marks, recessive grid, no dual axes.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#222222", "#555555", "#e6e6e6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_bo", ["#1c5cab", "#86b6ef", "#e8e8e6", "#f4a582", "#c2410c"])


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 5, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png", dpi=300)
    plt.close(fig)
