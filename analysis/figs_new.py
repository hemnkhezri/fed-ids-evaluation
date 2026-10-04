"""Three added figures: leakage schematic, audit matrix, Study 3 per-unit distribution."""
import json, random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f4f3f0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GOOD, WARN, CRIT, NEUTRAL = "#0ca30c", "#fab219", "#d03b3b", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ leakage schematic
def leakage():
    W, T = 14, 5
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), gridspec_kw={"hspace": 0.32})
    seqs = [(s, s + T) for s in range(W - T + 1)]          # 10 sequences, stride 1
    rnd = random.Random(4)
    val_r = set(rnd.sample(range(len(seqs)), 2))            # random 20% of sequences for validation
    for ax, mode in zip(axes, ("random", "chrono")):
        ax.set_xlim(-0.6, W + 0.2); ax.axis("off")
        nrow = len(seqs)
        ytop0 = nrow * 0.42 + 0.55; ax.set_ylim(ytop0 - 0.5 - (nrow - 1) * 0.42 - 0.25, ytop0 + 1.25)
        ytop = nrow * 0.42 + 0.55
        if mode == "chrono":
            regions = [(0, 9, "train"), (9, 14, "val")]
            for a, b, kind in regions:
                ax.add_patch(Rectangle((a, ytop - 0.08), b - a, 0.62, fc=BLUE if kind == "train" else ORANGE, alpha=0.18, ec="none"))
                ax.text((a + b) / 2, ytop + 0.72, "training region" if kind == "train" else "validation region",
                        ha="center", va="bottom", fontsize=8, color=INK2)
            ax.plot([9, 9], [ytop - 0.5 - (nrow - 1) * 0.42 - 0.1, ytop + 0.6], color=INK2, lw=1, ls=(0, (3, 2)))
        for w in range(W):
            ax.add_patch(Rectangle((w + 0.04, ytop), 0.92, 0.46, fc="white", ec=MUTED, lw=0.8))
            ax.text(w + 0.5, ytop + 0.23, f"w{w + 1}", ha="center", va="center", fontsize=7, color=INK2)
        for i, (a, b) in enumerate(seqs):
            y = ytop - 0.5 - i * 0.42
            if mode == "random":
                kind = "val" if i in val_r else "train"
            else:
                kind = "train" if b <= 9 else ("val" if a >= 9 else "drop")
            col = {"train": BLUE, "val": ORANGE, "drop": "#c9c8c3"}[kind]
            ax.add_patch(FancyBboxPatch((a + 0.08, y), (b - a) - 0.16, 0.26, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=col if kind != "drop" else "white", ec=col, lw=1.2,
                                        ls="-" if kind != "drop" else (0, (2, 2))))
            label = {"train": "training", "val": "validation", "drop": "dropped (crosses boundary)"}[kind]
            ax.text(b + 0.1, y + 0.13, f"s{i + 1}: {label}", va="center", fontsize=6.8, color=INK2)
        if mode == "random":
            # overlap highlight for one validation sequence
            vi = sorted(val_r)[0]; a, b = seqs[vi]
            ax.add_patch(Rectangle((a, ytop - 0.02), b - a, 0.5, fc="none", ec=ORANGE, lw=1.8))
            ax.set_title("(a) Random split of overlapping sequences (pilot protocol): each validation sequence shares "
                         "four of its five windows\nwith training sequences", loc="left", fontsize=8.6, color=INK, pad=4)
        else:
            ax.set_title("(b) Window-disjoint chronological split (corrected protocol): the timeline is divided first, "
                         "sequences are formed\ninside each region, and those crossing the boundary are dropped",
                         loc="left", fontsize=8.6, color=INK, pad=16)
    save(fig, "fig_leakage")


# ------------------------------------------------------------------ audit matrix
def audit():
    A = json.load(open(os.path.join(ROOT, "audit", "final_codes.json")))
    items = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"]
    names = ["Temporal\nor capture\nsplit", "No\nsequence-\nleakage risk", "Scaling\nfitted on\ntraining", "Rounds\nand steps\nreported",
             "Equal\nbaseline\nbudgets", "Baselines\nre-run", "MCC or\nbalanced\naccuracy", "Non-IID\nclients",
             "Repeats\nwith\ndispersion", "External\ntest set", "Public\ncode"]
    G, P, B, M = "met", "partly", "not met", "not reported"
    rule = {
        "A1": {"temporal_or_capture": G, "provided_split": P, "random": B, "not_reported": M},
        "A2": {"no": G, "yes": B, "unclear": M},
        "A3": {"train_only": G, "all_data": B, "not_reported": M},
        "A4": {"both": G, "one": P, "neither": M},
        "A5": {"same": G, "copied": B, "different_or_unstated": M},
        "A6": {"rerun": G, "copied": B, "none": M},
        "A7": {"yes": G, "no": B},
        "A8": {"yes": G, "no": B, "unclear": M},
        "A9": {"runs_with_dispersion_or_test": G, "runs_no_dispersion": P, "single_or_not_reported": M},
        "A10": {"yes": G, "no": B},
        "A11": {"yes": G, "no": B},
    }
    col = {G: GOOD, P: WARN, B: CRIT, M: NEUTRAL}
    glyph = {G: "✓", P: "~", B: "✗", M: "–"}
    rows = sorted(A, key=lambda r: (-sum(rule[k][r["codes"][k]] == G for k in items), r["id"]))
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for i, r in enumerate(rows):
        for j, k in enumerate(items):
            st = rule[k][r["codes"][k]]
            ax.add_patch(Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, fc=col[st], ec="white", lw=1.5))
            ax.text(j + 0.5, i + 0.52, glyph[st], ha="center", va="center", fontsize=8,
                    color="white" if st in (G, B) else INK, fontweight="bold")
    ax.set_xlim(0, len(items)); ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(items))]); ax.set_xticklabels(names, fontsize=6.8, linespacing=1.1)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks([i + 0.5 for i in range(len(rows))]); ax.set_yticklabels([r["id"] for r in rows], fontsize=7.2)
    for s in ax.spines.values(): s.set_visible(False)
    tot = [sum(rule[k][r["codes"][k]] == G for r in rows) for k in items]
    for j, t in enumerate(tot):
        ax.text(j + 0.5, len(rows) + 0.75, f"{t}/24", ha="center", va="center", fontsize=7.5, color=INK)
    ax.text(-0.15, len(rows) + 0.75, "met", ha="right", va="center", fontsize=7.5, color=INK2)
    handles = [Rectangle((0, 0), 1, 1, fc=col[s]) for s in (G, P, B, M)]
    ax.legend(handles, ["✓ practice met", "~ partly met", "✗ not met", "– not reported"], loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False, fontsize=7.8)
    save(fig, "fig_audit")


# ------------------------------------------------------------------ Study 3 distribution per unit
def s3dist():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "study3", "results", "results_s3.jsonl"))]
    unit = defaultdict(list)
    for r in rows:
        if r.get("stage") == "main" and r["regime"] == "noniid" and r["config"] in ("FG", "B1", "B2", "B4", "B5"):
            unit[(r["p_label"], r["dataset"], r["config"], r["partition_seed"])].append(r["MCC"])
    meths = ["FG", "B4", "B1", "B2", "B5"]
    mname = {"FG": "FedGTCL", "B4": "FedProx", "B1": "FedAvg-\nGCN-GRU", "B2": "FedAvg-\nLSTM", "B5": "Pseudo-\nlabelling"}
    mcol = {"FG": BLUE, "B4": ORANGE, "B1": AQUA, "B2": YELLOW, "B5": MAGENTA}
    dsets = [("wustl", "WUSTL-IIoT-2021"), ("toniot", "TON_IoT-Network"), ("edgeiiot", "Edge-IIoTset")]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2), sharey=True, gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    rng = np.random.default_rng(0)
    for i, pl in enumerate((0.1, 0.5)):
        for j, (d, dn) in enumerate(dsets):
            ax = axes[i, j]
            ax.axhline(0, color=MUTED, lw=0.8)
            for x, m in enumerate(meths):
                v = np.array([np.mean(unit[(pl, d, m, ps)]) for ps in range(101, 111)])
                xs = x + rng.uniform(-0.16, 0.16, len(v))
                ax.scatter(xs, v, s=16, color=mcol[m], edgecolor="white", linewidth=0.6, zorder=3)
                ax.plot([x - 0.28, x + 0.28], [v.mean()] * 2, color=INK, lw=1.8, zorder=4)
            ax.set_xticks(range(len(meths))); ax.set_xticklabels([mname[m] for m in meths], fontsize=6.6)
            ax.set_ylim(-0.25, 1.02); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.set_title(f"{dn}, $p_L$ = {pl}", fontsize=8.4, color=INK, loc="left")
            if j == 0: ax.set_ylabel("MCC (mean of two seeds)")
    save(fig, "fig_s3_units")




def curves():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "study3", "results", "results_s3.jsonl"))]
    g = defaultdict(list)
    for r in rows:
        if r.get("stage") == "main" and r["regime"] == "noniid" and r["config"] in ("B1", "B2", "B4", "B5"):
            g[(r["dataset"], r["p_label"], r["config"])].append([np.nan if x is None else x for x in r["loss_curve"]])
    meths = ["B4", "B1", "B2", "B5"]
    mname = {"B4": "FedProx", "B1": "FedAvg-GCN-GRU", "B2": "FedAvg-LSTM", "B5": "Pseudo-labelling"}
    mcol = {"B4": ORANGE, "B1": AQUA, "B2": YELLOW, "B5": MAGENTA}
    dsets = [("wustl", "WUSTL-IIoT-2021"), ("toniot", "TON_IoT-Network"), ("edgeiiot", "Edge-IIoTset")]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.0), sharey=True, sharex=True, gridspec_kw={"hspace": 0.35, "wspace": 0.08})
    x = np.arange(1, 51)
    for i, pl in enumerate((0.1, 0.5)):
        for j, (d, dn) in enumerate(dsets):
            ax = axes[i, j]
            ax.axvline(1.5, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
            for m in meths:
                a = np.nanmedian(np.array(g[(d, pl, m)], dtype=float), 0)
                ax.plot(x, a, color=mcol[m], lw=1.6, label=mname[m])
            ax.set_ylim(0, 1.0); ax.set_xlim(0.5, 50.5); ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{dn}, $p_L$ = {pl}", fontsize=8.4, color=INK, loc="left")
            if j == 0: ax.set_ylabel("Training loss (median of 20 runs)")
            if i == 1: ax.set_xlabel("Round (20 local steps each)")
    axes[0, 2].legend(frameon=False, fontsize=7.2, loc="upper right")
    save(fig, "figS2_curves")


if __name__ == "__main__":
    import sys
    for fn in sys.argv[1:] or ["curves"]: globals()[fn]()
