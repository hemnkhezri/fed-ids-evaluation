"""Figures for the evaluation paper, drawn from numbers.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = json.load(open(os.path.join(ROOT, "analysis", "numbers.json")))
OUT = os.path.join(ROOT, "figures")
import os; os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
C = {"FG": "#2a78d6", "B4": "#eb6834", "B1": "#1baf7a", "B2": "#eda100"}
NAME = {"FG": "FedGTCL", "B4": "FedProx", "B1": "FedAvg-GCN-GRU", "B2": "FedAvg-LSTM"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------- Fig. 1: the sequence of studies
steps = [("Study 1", "Pilot", "Random split of\noverlapping sequences;\n30 optimiser steps;\nbaselines with all labels",
          "FedGTCL ahead;\nbaselines collapse"),
         ("Study 2", "Development and\nheld-out tests", "Configurations tuned\non an inner split;\nCICAPT-IIoT2024 and\nX-IIoTID held out",
          "No method learns on\neither held-out set"),
         ("Study 3", "Window-disjoint,\nequal budget (sealed)", "Chronological regions;\n1,000 steps; equal labels;\nFedProx and\npseudo-labelling added",
          "Not better than FedProx;\nits own components hurt"),
         ("Study 4", "Revised variant,\nexternal test (sealed)", "Fixed-rule selection;\nre-test on new partitions;\ntest on UNSW-NB15",
          "Wins internally,\nloses externally"),
         ("Study 5", "Controlled factors\n(sealed)", "Split x budget x\nbaseline labels\non four datasets",
          "Budget flips the ranking;\nleakage raises IIoT results"),
         ("Study 6", "Budget curve and\nre-tuning (sealed)", "Up to 3,000 steps with\ncheckpoints; learning rate\nre-tuned on an inner split",
          "FedGTCL ahead at 200–400\nsteps, behind from 1,000"),
         ("Study 7", "Per-record classifiers\n(sealed)", "Random vs temporal\nrecord split; LR and MLP;\nthree datasets",
          "Random split: MCC +0.15,\naccuracy +0.03")]
fig, ax = plt.subplots(figsize=(8.6, 6.3)); ax.axis("off"); ax.set_xlim(0, 8.6); ax.set_ylim(0, 6.6)
for i, (s, t, what, res) in enumerate(steps):
    row, col = divmod(i, 4); x = 0.1 + col * 2.1 + (1.05 if row == 1 else 0); y0 = 3.3 * (1 - row)
    ax.add_patch(FancyBboxPatch((x, y0 + 0.95), 1.85, 2.2, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#f4f3f0", ec=MUTED, lw=0.8))
    ax.text(x + 0.925, y0 + 2.95, s, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + 0.925, y0 + 2.55, t, ha="center", va="center", fontsize=8, color=C["FG"], fontweight="bold")
    ax.text(x + 0.925, y0 + 1.75, what, ha="center", va="center", fontsize=7.2, color=INK2, linespacing=1.3)
    ax.text(x + 0.925, y0 + 0.45, res, ha="center", va="center", fontsize=7.4, color=INK)
    ax.plot([x + 0.925, x + 0.925], [y0 + 0.95, y0 + 0.72], color=MUTED, lw=0.8)
    if col < (3 if row == 0 else 2):
        ax.annotate("", xy=(x + 2.08, y0 + 2.05), xytext=(x + 1.87, y0 + 2.05), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
save(fig, "fig1_studies")
import sys; sys.exit() if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "studies" else None

# ---------------- Fig. 2: Study 5, mean MCC and collapse rate per protocol
T = N["s5_table"]
COND = [("random", "pilot"), ("chrono", "pilot"), ("random", "full"), ("chrono", "full")]
XL = ["Random split\n30 steps\n(pilot)", "Chronological\n30 steps", "Random split\n1,000 steps", "Chronological\n1,000 steps\n(corrected)"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, kind in zip(axes, ("mean", "deg")):
    for m in ("FG", "B4", "B1", "B2"):
        ys = []
        for sp, b in COND:
            e = T[f"{sp}|{b}|{m}|pooled"]
            if kind == "mean":
                ys.append(e["mean"])
            else:
                k, n = e["degenerate"].split("/"); ys.append(100 * int(k) / int(n))
        ax.plot([0, 1], ys[:2], color=C[m], lw=2, marker="o", ms=6, mec="white", mew=1.2, label=NAME[m], zorder=3)
        ax.plot([2, 3], ys[2:], color=C[m], lw=2, marker="o", ms=6, mec="white", mew=1.2, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(XL, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.axvspan(1.5, 3.5, color="#f4f3f0", zorder=0)
    if kind == "mean":
        ax.set_ylabel("Mean MCC (4 datasets, 40 partitions)"); ax.set_ylim(0, 0.85); ax.set_title("a  Detection (MCC)", loc="left", fontsize=9.5, color=INK)
    else:
        ax.set_ylabel("Runs that collapsed to one class (%)"); ax.set_ylim(0, 100); ax.set_title("b  Collapsed models", loc="left", fontsize=9.5, color=INK)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.5, -0.02, "Shaded: 1,000 optimiser steps. All methods with 50% labels.", ha="center", fontsize=7.5, color=MUTED)
plt.tight_layout(); save(fig, "fig2_study5")

# ---------------- Fig. 3: confirmatory differences, Studies 3 and 4 (forest plot)
rows = []
lab = {"B4": "FedProx", "B5": "pseudo-labelling"}
for c in N["s3_conf"]:
    rows.append(("Study 3: FedGTCL", f"vs {lab[c['comparator']]}, {int(c['p_label']*100)}% labels", c["mean_diff"], c["ci95"], c["p_sup_holm"]))
for fam, title in (("s4_internal", "Study 4 internal: variant C4"), ("s4_external", "Study 4 external (UNSW-NB15): C4")):
    for c in N[fam]["confirmatory"]:
        rows.append((title, f"vs {lab[c['comparator']]}, {int(c['p_label']*100)}% labels", c["mean_diff"], c["ci95"], c["p_sup_holm"]))
fig, ax = plt.subplots(figsize=(8.2, 5.0))
y = 0; yt, yl = [], []; last = None
for g, l, m, ci, p in rows:
    if g != last:
        y -= 0.6; ax.text(-0.62, y, g, fontsize=8.5, fontweight="bold", color=INK, va="center"); y -= 0.9; last = g
    col = C["FG"] if (p < 0.05 and m > 0) else (C["B4"] if p < 0.05 else MUTED)
    ax.plot(ci, [y, y], color=col, lw=2, solid_capstyle="round"); ax.plot(m, y, "o", color=col, ms=7, mec="white", mew=1.2, zorder=3)
    ax.text(0.58, y, f"{m:+.2f}  (p = {p:.3f})" if p >= 0.001 else f"{m:+.2f}  (p < 0.001)", fontsize=7.5, color=INK2, va="center")
    yt.append(y); yl.append(l); y -= 0.75
ax.axvline(0, color=INK2, lw=0.8); ax.axvspan(-0.05, 0.05, color="#f4f3f0", zorder=0)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.8); ax.set_xlim(-0.62, 0.95); ax.set_ylim(y, 0)
ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4]); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("Difference in MCC (method minus comparator), mean with bootstrap 95% CI; p Holm-adjusted")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
save(fig, "fig3_forest")

# ---------------- Fig. 4: Study 5 confirmatory effects
fig, ax = plt.subplots(figsize=(8.2, 3.4))
short = ["Leakage inflation: FedGTCL", "Leakage inflation: FedAvg-GCN-GRU", "Leakage inflation: FedAvg-LSTM", "Leakage inflation: FedProx",
         "Budget effect on FedGTCL - FedProx gap", "Pilot protocol effect on FedGTCL - FedAvg gap"]
for i, (c, s) in enumerate(zip(N["s5_conf"], short)):
    yy = -i
    ax.plot(c["ci95"], [yy, yy], color=C["FG"], lw=2, solid_capstyle="round"); ax.plot(c["mean"], yy, "o", color=C["FG"], ms=7, mec="white", mew=1.2, zorder=3)
    ax.text(0.70, yy, f"{c['mean']:+.2f}  (p = {c['p_holm']:.3f})" if c["p_holm"] >= 0.001 else f"{c['mean']:+.2f}  (p < 0.001)", fontsize=7.5, color=INK2, va="center")
ax.set_yticks([-i for i in range(6)]); ax.set_yticklabels(short, fontsize=7.8)
ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-0.05, 0.92); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("Difference in MCC, mean with bootstrap 95% CI (n = 40); p Holm-adjusted over six tests")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
save(fig, "fig4_study5_tests")

# ---------------- Fig. S1: attention cost vs host count
B = N["bench"]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
V = [b["V"] for b in B]
for key, lab_, col in (("ref_dense_ms", "Dense-masked (paper's implementation)", C["B4"]), ("batched_dense_ms", "Dense-masked, batched", C["B2"]),
                       ("sparse_ms", "Sparse (top-m neighbours only)", C["FG"])):
    ax.plot(V, [b[key] for b in B], marker="o", ms=5, lw=2, color=col, mec="white", mew=1, label=lab_)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Hosts per window |V|"); ax.set_ylabel("ms per T = 5 sequence (one CPU thread)")
ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=7.5)
save(fig, "figS1_sparse")
print("done")
