"""Fig. 8: Study-2 confirmatory results from the raw run records (p_label=0.1, 10 partitions x 2 seeds).
Top row: partition-level MCC (mean of the two seeds) per method.
Bottom row: paired per-partition differences FG - comparator (the quantity the Wilcoxon tests use).
Boxes: median and interquartile range; whiskers: 1.5 x IQR (matplotlib default)."""
import json, os, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/claude/work/p3bres/combined"
runs = []
for f in glob.glob(f"{SRC}/*.jsonl"):
    runs += [json.loads(l) for l in open(f) if l.strip()]
runs = [r for r in runs if r.get("tag") in ("phase2", "phase3b") and r["p_label"] == 0.1]
cell = defaultdict(list)
for r in runs:
    cell[(r["dataset"], r["config"], r["partition_seed"])].append(r["MCC"])
PM = lambda d, m: np.array([np.mean(cell[(d, m, p)]) for p in range(101, 111)])
DS = [("wustl", "WUSTL-IIoT-2021"), ("toniot", "TON_IoT-Network"), ("edgeiiot", "Edge-IIoTset (10%)"),
      ("cicapt", "CICAPT-IIoT2024\n(independent test 1)"), ("xiiotid", "X-IIoTID\n(independent test 2)")]
M = ["FG", "B1", "B2", "B3", "B4", "B5"]
CONF = {("wustl", "B4"), ("toniot", "B4"), ("edgeiiot", "B4"), ("cicapt", "B4"), ("toniot", "B5"), ("edgeiiot", "B5"),
        ("xiiotid", "B4"), ("xiiotid", "B5")}      # comparisons inside a confirmatory test (H_A-H_C, H_E1-H_E3)
C_A, C_B, C_D, C_E = "#2a78d6", "#eb6834", "#3b3b3b", "#9a9a96"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED})
fig, axes = plt.subplots(2, 5, figsize=(15, 7.4), sharey="row", gridspec_kw=dict(height_ratios=[1, 0.9]))
rng = np.random.default_rng(1)


def box(ax, i, v, col):
    ax.boxplot([v], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=col + "33", edgecolor=col, lw=1.2), medianprops=dict(color=col, lw=2),
               whiskerprops=dict(color=col, lw=1), capprops=dict(color=col, lw=1))
    ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=15, color=col, edgecolor="white", linewidth=0.6, zorder=3)


for k, (d, title) in enumerate(DS):
    ax = axes[0, k]; ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for i, m in enumerate(M):
        box(ax, i, PM(d, m), C_A if m == "FG" else C_B)
    ax.set_xticks(range(6)); ax.set_xticklabels(M, fontsize=8.5); ax.set_title(title, fontsize=9.5, color=INK, pad=8)
    ax.set_ylim(-0.15, 1.05)
    ax2 = axes[1, k]; ax2.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    fg = PM(d, "FG")
    for i, b in enumerate(M[1:]):
        box(ax2, i, fg - PM(d, b), C_D if (d, b) in CONF else C_E)
    ax2.set_xticks(range(5)); ax2.set_xticklabels([f"FG−{b}" for b in M[1:]], fontsize=8)
    ax2.set_ylim(-0.75, 1.05)
    for a in (ax, ax2):
        a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
axes[0, 0].set_ylabel("Partition-level MCC (mean of 2 seeds)")
axes[1, 0].set_ylabel("Paired difference per partition\n(FG − comparator)")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_A, label="FedGTCL (FG)"),
                    Line2D([], [], marker="o", ls="", color=C_B, label="Tuned baselines B1-B5"),
                    Line2D([], [], marker="o", ls="", color=C_D, label="Difference in a confirmatory test"),
                    Line2D([], [], marker="o", ls="", color=C_E, label="Difference, exploratory only"),
                    Line2D([], [], ls=(0, (4, 3)), color=MUTED, label="Zero")],
           loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(os.path.join(HERE, "fig8_study2_confirmatory.png"), dpi=300); print("ok")
