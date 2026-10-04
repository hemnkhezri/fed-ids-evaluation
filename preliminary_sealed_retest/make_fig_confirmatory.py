"""Fig. 7: per-seed MCC distribution for all seven arms, three datasets (confirmatory cell)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results_v2r.jsonl"))]
rows = [r for r in rows if r["partition"] == "noniid" and r["p_label"] == 0.5]
ARMS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
LBL = {"A1": "A1\nFull", "A2": "A2\nno con.", "A3": "A3\nno \u03b3", "A4": "A4\nno FAO",
       "B1": "B1\nGCN-GRU", "B2": "B2\nLSTM", "B3": "B3\nFedAdam"}
DS = [("wustl", "WUSTL-IIoT-2021"), ("toniot", "TON_IoT-Network"), ("edgeiiot", "Edge-IIoTset (10%)")]
C_A, C_B = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
rng = np.random.default_rng(0)
for ax, (ds, title) in zip(axes, DS):
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for i, arm in enumerate(ARMS):
        v = np.array([r["MCC"] for r in rows if r["dataset"] == ds and r["arm"] == arm])
        col = C_A if arm.startswith("A") else C_B
        ax.boxplot([v], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=col + "33", edgecolor=col, lw=1.2),
                   medianprops=dict(color=col, lw=2), whiskerprops=dict(color=col, lw=1),
                   capprops=dict(color=col, lw=1))
        x = i + rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(x, v, s=18, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        k = int((v > 0).sum())
        ax.text(i, 1.07, f"{k}/{len(v)}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=7.8)
    ax.set_title(title, fontsize=10, color=INK, pad=16)
    lo=min(r["MCC"] for r in rows); ax.set_ylim(min(-0.08, lo-0.05), 1.2); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("MCC on validation windows (per seed)")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_A, label="FedGTCL and its ablations (A1-A4)"),
                    Line2D([], [], marker="o", ls="", color=C_B, label="Federated baselines (B1-B3)"),
                    Line2D([], [], ls=(0, (4, 3)), color=MUTED, label="MCC = 0 (no discrimination)")],
           loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, 0.955, "Numbers above boxes: seeds with MCC > 0 (discriminative) out of 10", ha="center",
         fontsize=7.8, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
out = os.path.join(HERE, "fig7_confirmatory_mcc.png")
fig.savefig(out, dpi=300)
print(out)
