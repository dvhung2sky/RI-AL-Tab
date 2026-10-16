"""Shared plot style for the A8L2 paper: Times New Roman 12 pt, STIX math, figures drawn near print width (7 in) so the
text stays ~11-12 pt after Word scales them to the 6.3 in text block. Palette = validated dataviz reference slots 1-4."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # AL-Tab, RI-AL-Tab, accents
G0, G2, GS = "#a6a6a6", "#6b6b6b", "#262626"                               # AK-MCS L0 / L2 / per regime
W = 7.0                                                                      # figure width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})

def save(fig, name):
    fig.tight_layout(); fig.savefig(name + ".png", dpi=300); fig.savefig(name + ".pdf"); plt.close(fig)
