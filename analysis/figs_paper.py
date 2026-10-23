"""Figures 1-9 and S1-S2 of the paper, as they appear in the manuscript. Run from the repository root:
    python analysis/figs_paper.py                       # Figs. 1-9
    python analysis/figs_paper.py supp_curves supp_sparse   # Figs. S1-S2

High-resolution, print-size versions of the manuscript figures.

Every figure is drawn at its printed width (6.5 in = full text width), so the
font sizes below are the sizes a reader sees on the page.  Smallest text is 8 pt,
most text 9-10 pt.  Output: 600 dpi PNG and vector PDF in figures/.
"""
import json, random, sys, textwrap
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.transforms as mtrans

OUT = "figures"
import os; os.makedirs(OUT, exist_ok=True)
N = json.load(open("analysis/numbers.json"))

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#3f3e3b", "#7a7974", "#e1e0dc", "#f2f1ee"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GOOD, WARN, CRIT, NEUTRAL = "#0ca30c", "#fab219", "#d03b3b", "#e1e0dc"
C = {"FG": BLUE, "B4": ORANGE, "B1": AQUA, "B2": YELLOW, "B5": MAGENTA}
NAME = {"FG": "FedGTCL", "B4": "FedProx", "B1": "FedAvg-GCN-GRU", "B2": "FedAvg-LSTM", "B5": "Pseudo-labeling"}
NAMES = {"wustl": "WUSTL-IIoT-2021", "edgeiiot": "Edge-IIoTset", "xiiotid": "X-IIoTID", "toniot": "TON_IoT-Network"}

W = 6.5            # printed width in inches
FS, TICK, TITLE, SMALL = 9.5, 9, 10, 8.5
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": FS,
    "axes.titlesize": TITLE, "axes.labelsize": FS, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "legend.fontsize": TICK, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
SIZES = {}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    from PIL import Image
    w, h = Image.open(f"{OUT}/{name}.png").size
    SIZES[name] = (w / 600, h / 600)
    print(f"{name}: {w / 600:.2f} x {h / 600:.2f} in  (font scale in a 6.5 in column: {min(1, W / (w / 600)):.2f})")
    plt.close(fig)


def ptext(v, p):
    t = f"{v:+.2f}  (p = {p:.3f})" if p >= 0.001 else f"{v:+.2f}  (p < 0.001)"
    return t.replace("-", "\u2212", 1) if v < 0 else t


# ------------------------------------------------------------------ Fig. 1  leakage schematic
def leakage():
    Wn, T = 14, 5
    seqs = [(s, s + T) for s in range(Wn - T + 1)]
    val_r = set(random.Random(4).sample(range(len(seqs)), 2))
    fig, axes = plt.subplots(2, 1, figsize=(W, 7.4), gridspec_kw={"hspace": 0.30})
    titles = {
        "random": ("(a) Random split of overlapping sequences (pilot protocol)",
                   "Each validation sequence shares four of its five windows with training sequences."),
        "chrono": ("(b) Window-disjoint chronological split (corrected protocol)",
                   "The timeline is divided first, sequences are formed inside each region, and sequences "
                   "crossing the boundary are dropped."),
    }
    for ax, mode in zip(axes, ("random", "chrono")):
        nrow = len(seqs); dy = 0.5
        ytop = 0
        ax.set_xlim(-1.0, Wn + 0.1); ax.axis("off")
        ax.set_ylim(ytop - 0.55 - (nrow - 1) * dy - 0.2, ytop + 1.05)
        if mode == "chrono":
            for a, b, kind in ((0, 9, "train"), (9, 14, "val")):
                ax.add_patch(Rectangle((a, ytop - 0.08), b - a, 0.66, fc=BLUE if kind == "train" else ORANGE, alpha=0.16, ec="none"))
                ax.text((a + b) / 2, ytop + 0.68, "training region" if kind == "train" else "validation region",
                        ha="center", va="bottom", fontsize=FS, color=INK2)
            ax.plot([9, 9], [ytop - 0.55 - (nrow - 1) * dy - 0.12, ytop + 0.62], color=INK2, lw=1.2, ls=(0, (3, 2)))
        for w in range(Wn):
            ax.add_patch(Rectangle((w + 0.04, ytop), 0.92, 0.5, fc="white", ec=MUTED, lw=0.9))
            ax.text(w + 0.5, ytop + 0.25, f"w{w + 1}", ha="center", va="center", fontsize=SMALL, color=INK2)
        for i, (a, b) in enumerate(seqs):
            y = ytop - 0.55 - i * dy
            kind = ("val" if i in val_r else "train") if mode == "random" else ("train" if b <= 9 else ("val" if a >= 9 else "drop"))
            col = {"train": BLUE, "val": ORANGE, "drop": "#a9a8a3"}[kind]
            ax.add_patch(FancyBboxPatch((a + 0.08, y), (b - a) - 0.16, 0.32, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=col if kind != "drop" else "white", ec=col, lw=1.4,
                                        ls="-" if kind != "drop" else (0, (2.5, 2))))
            ax.text(a - 0.05, y + 0.16, f"s{i + 1}", ha="right", va="center", fontsize=SMALL, color=INK2)
        if mode == "random":
            a, b = seqs[sorted(val_r)[0]]
            ax.add_patch(Rectangle((a, ytop - 0.03), b - a, 0.56, fc="none", ec=ORANGE, lw=2.2))
        t1, t2 = titles[mode]
        ax.set_title(t1 + "\n" + "\n".join(textwrap.wrap(t2, 86)), loc="left", fontsize=TITLE, color=INK,
                     pad=18 if mode == "chrono" else 6, linespacing=1.35)
    h = [Rectangle((0, 0), 1, 1, fc=BLUE, ec=BLUE), Rectangle((0, 0), 1, 1, fc=ORANGE, ec=ORANGE),
         Rectangle((0, 0), 1, 1, fc="white", ec="#a9a8a3", ls=(0, (2.5, 2)), lw=1.4),
         Rectangle((0, 0), 1, 1, fc="none", ec=ORANGE, lw=2.2)]
    fig.legend(h, ["training sequence", "validation sequence", "dropped (crosses boundary)",
                   "windows of one validation sequence"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=TICK, handlelength=1.6, columnspacing=1.8)
    fig.subplots_adjust(bottom=0.09)
    save(fig, "Fig1_leakage")


# ------------------------------------------------------------------ Fig. 2  the seven studies
def studies():
    steps = [("Study 1", "Pilot", "Random split of overlapping sequences; 30 optimizer steps; baselines with all labels",
              "FedGTCL ahead; baselines collapse"),
             ("Study 2", "Development and held-out tests", "Configurations tuned on an inner split; CICAPT-IIoT2024 and X-IIoTID held out",
              "No method learns on either held-out set"),
             ("Study 3", "Window-disjoint, equal budget (sealed)", "Chronological regions; 1,000 steps; equal labels; FedProx and pseudo-labeling added",
              "Not better than FedProx; its own components hurt"),
             ("Study 4", "Revised variant, external test (sealed)", "Fixed-rule selection; re-test on new partitions; test on UNSW-NB15",
              "Wins internally, loses externally"),
             ("Study 5", "Controlled factors (sealed)", "Split × budget × baseline labels on four datasets",
              "Budget flips the ranking; leakage raises IIoT results"),
             ("Study 6", "Budget curve and re-tuning (sealed)", "Up to 3,000 steps with checkpoints; learning rate re-tuned on an inner split",
              "FedGTCL ahead at 200–400 steps, behind from 1,000"),
             ("Study 7", "Per-record classifiers (sealed)", "Random vs temporal record split; LR and MLP; three datasets",
              "Random split: MCC +0.15, accuracy +0.03")]
    bw, gap = 1.5, 0.1667
    bh, rh = 1.82, 2.72          # box height, row pitch
    H = 0.05 + rh + bh + 0.62
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    for i, (s, t, what, res) in enumerate(steps):
        row, col = divmod(i, 4)
        x = col * (bw + gap) + (0 if row == 0 else (bw + gap) / 2)
        top = H - 0.05 - row * rh
        ax.add_patch(FancyBboxPatch((x + 0.02, top - bh), bw - 0.04, bh, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=SURF, ec=MUTED, lw=0.9))
        cx = x + bw / 2
        ax.text(cx, top - 0.1, s, ha="center", va="top", fontsize=TITLE, fontweight="bold", color=INK)
        ax.text(cx, top - 0.36, "\n".join(textwrap.wrap(t, 20, break_on_hyphens=False)), ha="center", va="top", fontsize=SMALL + 0.5,
                color=BLUE, fontweight="bold", linespacing=1.15)
        ax.text(cx, top - 0.92, "\n".join(textwrap.wrap(what, 21, break_on_hyphens=False)), ha="center", va="top", fontsize=SMALL,
                color=INK2, linespacing=1.2)
        ax.plot([cx, cx], [top - bh, top - bh - 0.14], color=MUTED, lw=0.9)
        ax.text(cx, top - bh - 0.2, "\n".join(textwrap.wrap(res, 20, break_on_hyphens=False)), ha="center", va="top", fontsize=SMALL + 0.5,
                color=INK, linespacing=1.15)
        if col < (3 if row == 0 else 2):
            yy = top - bh / 2
            ax.annotate("", xy=(x + bw + gap + 0.01, yy), xytext=(x + bw - 0.01, yy),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, mutation_scale=11))
    save(fig, "Fig2_studies")


# ------------------------------------------------------------------ Fig. 3  audit matrix
def audit():
    A = json.load(open("audit/final_codes.json"))
    items = [f"A{k}" for k in range(1, 12)]
    names = ["Temporal\nor\ncapture\nsplit", "No\nsequence\nleakage\nrisk", "Scaling\nfitted on\ntraining", "Rounds\nand\nsteps\nreported",
             "Equal\nbaseline\nbudgets", "Base-\nlines\nre-run", "MCC or\nbalanced\naccuracy", "Non-IID\nclients",
             "Repeats\nwith\ndisper-\nsion", "External\ntest set", "Public\ncode"]
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
    fig, ax = plt.subplots(figsize=(W, 8.3))
    fig.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.05)
    for i, r in enumerate(rows):
        for j, k in enumerate(items):
            st = rule[k][r["codes"][k]]
            ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc=col[st], ec="white", lw=1.2))
            ax.text(j + 0.5, i + 0.52, glyph[st], ha="center", va="center", fontsize=TICK,
                    color="white" if st in (G, B) else INK, fontweight="bold")
    ax.set_xlim(0, len(items)); ax.set_ylim(len(rows) + 1.1, 0)
    ax.set_xticks([j + 0.5 for j in range(len(items))]); ax.set_xticklabels(names, fontsize=8, linespacing=1.08)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks([i + 0.5 for i in range(len(rows))]); ax.set_yticklabels([r["id"] for r in rows], fontsize=TICK)
    for s in ax.spines.values(): s.set_visible(False)
    tot = [sum(rule[k][r["codes"][k]] == G for r in rows) for k in items]
    for j, t in enumerate(tot):
        ax.text(j + 0.5, len(rows) + 0.6, f"{t}/24", ha="center", va="center", fontsize=TICK, color=INK, fontweight="bold")
    ax.text(-0.12, len(rows) + 0.6, "Met", ha="right", va="center", fontsize=TICK, color=INK)
    handles = [Rectangle((0, 0), 1, 1, fc=col[s]) for s in (G, P, B, M)]
    fig.legend(handles, ["✓ practice met", "~ partly met", "✗ not met", "– not reported"], loc="lower center",
               bbox_to_anchor=(0.55, 0.0), ncol=4, frameon=False, fontsize=TICK, columnspacing=1.4, handlelength=1.3)
    save(fig, "Fig3_audit")


# ------------------------------------------------------------------ Fig. 4  Study 5 MCC and collapse
def study5():
    T = N["s5_table"]
    COND = [("random", "pilot"), ("chrono", "pilot"), ("random", "full"), ("chrono", "full")]
    XL = ["Random\nsplit,\n30 steps\n(pilot)", "Chrono-\nlogical,\n30 steps", "Random\nsplit,\n1,000\nsteps",
          "Chrono-\nlogical,\n1,000\nsteps\n(corrected)"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.9), gridspec_kw={"wspace": 0.36})
    for ax, kind in zip(axes, ("mean", "deg")):
        ax.axvspan(1.5, 3.5, color=SURF, zorder=0)
        for m in ("FG", "B4", "B1", "B2"):
            ys = []
            for sp, b in COND:
                e = T[f"{sp}|{b}|{m}|pooled"]
                if kind == "mean":
                    ys.append(e["mean"])
                else:
                    k, n = e["degenerate"].split("/"); ys.append(100 * int(k) / int(n))
            kw = dict(color=C[m], lw=2.2, marker="o", ms=7, mec="white", mew=1.2, zorder=3)
            ax.plot([0, 1], ys[:2], label=NAME[m], **kw); ax.plot([2, 3], ys[2:], **kw)
        ax.set_xticks(range(4)); ax.set_xticklabels(XL, fontsize=SMALL, linespacing=1.1); ax.set_xlim(-0.45, 3.45)
        ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
        if kind == "mean":
            ax.set_ylabel("Mean MCC\n(4 datasets, 40 partitions)"); ax.set_ylim(0, 0.85); ax.set_title("(a) Detection (MCC)", loc="left", color=INK)
        else:
            ax.set_ylabel("Runs collapsed to one class (%)"); ax.set_ylim(0, 100); ax.set_title("(b) Collapsed models", loc="left", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04), columnspacing=1.4, handlelength=1.8)
    fig.text(0.5, -0.16, "Shaded: 1,000 optimizer steps. All methods with 50% labels.", ha="center", fontsize=SMALL, color=INK2)
    save(fig, "Fig4_study5")


# ------------------------------------------------------------------ Fig. 5  Study 5 confirmatory tests
def study5_tests():
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.5))
    short = ["Leakage inflation:\nFedGTCL", "Leakage inflation:\nFedAvg-GCN-GRU", "Leakage inflation:\nFedAvg-LSTM",
             "Leakage inflation:\nFedProx", "Budget effect on\nFedGTCL − FedProx gap", "Pilot protocol effect on\nFedGTCL − FedAvg gap"]
    for i, (c, s) in enumerate(zip(N["s5_conf"], short)):
        yy = -i
        ax.plot(c["ci95"], [yy, yy], color=BLUE, lw=2.4, solid_capstyle="round")
        ax.plot(c["mean"], yy, "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=3)
        ax.text(1.0, yy, ptext(c["mean"], c["p_holm"]), fontsize=SMALL + 0.5, color=INK2, va="center",
                transform=mtrans.blended_transform_factory(ax.transAxes, ax.transData))
    ax.set_yticks([-i for i in range(6)]); ax.set_yticklabels(short, fontsize=SMALL + 0.5, linespacing=1.1)
    ax.axvline(0, color=INK2, lw=0.9); ax.set_xlim(-0.05, 0.8); ax.set_ylim(-5.6, 0.6)
    ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("Difference in MCC, mean with bootstrap 95% CI (n = 40);\np Holm-adjusted over six tests", fontsize=SMALL + 0.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "Fig5_study5_tests")


# ------------------------------------------------------------------ Fig. 6  Study 6 budget curves
def study6():
    a = json.load(open("study6/results/analysis_s6.json"))["curve"]
    CK = [1, 2, 5, 10, 20, 50, 100, 150]
    M = [("FG", "FedGTCL"), ("B4", "FedProx"), ("B1", "FedAvg-GCN-GRU"), ("B2", "FedAvg-LSTM"), ("B5", "Pseudo-labeling")]
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.6), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08, "hspace": 0.28})
    for ax, ds in zip(axes.flat, ["pooled", "wustl", "toniot", "edgeiiot"]):
        for m, lab in M:
            ks = [k for k in CK if not (m == "FG" and k <= 5)]
            y = [a[f"{m}|{k}|{ds}"]["mean"] for k in ks]
            big = m in ("FG", "B4")
            ax.plot([k * 20 for k in ks], y, "-o", color=C[m], lw=2.4 if big else 1.5, ms=4.5 if big else 3.5, label=lab, zorder=3 if big else 2)
        for x, t, ha, f in ((30, "pilot", "left", 1.08), (1000, "Studies 3–5", "right", 0.92)):
            ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (3, 2))); ax.text(x * f, 0.97, t, fontsize=SMALL, color=INK2, va="top", ha=ha)
        ax.set_xscale("log"); ax.set_xticks([20, 100, 1000, 3000]); ax.set_xticklabels(["20", "100", "1,000", "3,000"])
        ax.minorticks_off()
        ax.set_title("Pooled (3 datasets)" if ds == "pooled" else NAMES[ds], color=INK, loc="left")
        ax.set_ylim(-0.02, 1.0); ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for ax in axes[1]: ax.set_xlabel("Local steps per client (log scale)")
    fig.supylabel("Mean validation MCC (non-IID, $p_L$ = 0.5)", fontsize=FS, color=INK2, x=0.0)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, -0.07), columnspacing=1.5)
    save(fig, "Fig6_study6_curves")


# ------------------------------------------------------------------ Fig. 7  forest plot Studies 3 and 4
def forest():
    rows = []
    lab = {"B4": "FedProx", "B5": "pseudo-labeling"}
    for c in N["s3_conf"]:
        rows.append(("Study 3: FedGTCL", f"vs {lab[c['comparator']]}, {int(c['p_label'] * 100)}% labels", c["mean_diff"], c["ci95"], c["p_sup_holm"]))
    for fam, title in (("s4_internal", "Study 4 internal: variant C4"), ("s4_external", "Study 4 external (UNSW-NB15): C4")):
        for c in N[fam]["confirmatory"]:
            rows.append((title, f"vs {lab[c['comparator']]}, {int(c['p_label'] * 100)}% labels", c["mean_diff"], c["ci95"], c["p_sup_holm"]))
    fig, ax = plt.subplots(figsize=(W * 0.7, 5.6))
    y = 0; yt, yl = [], []; last = None
    for g, l, m, ci, p in rows:
        if g != last:
            y -= 0.55; ax.text(-0.6, y, g, fontsize=FS, fontweight="bold", color=INK, va="center",
                                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5); y -= 0.85; last = g
        col = BLUE if (p < 0.05 and m > 0) else (ORANGE if p < 0.05 else MUTED)
        ax.plot(ci, [y, y], color=col, lw=2.4, solid_capstyle="round")
        ax.plot(m, y, "o", color=col, ms=8, mec="white", mew=1.2, zorder=3)
        ax.text(1.0, y, ptext(m, p), fontsize=SMALL + 0.5, color=INK2, va="center",
                transform=mtrans.blended_transform_factory(ax.transAxes, ax.transData))
        yt.append(y); yl.append(l); y -= 0.75
    ax.axvline(0, color=INK2, lw=0.9); ax.axvspan(-0.05, 0.05, color=SURF, zorder=0)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=SMALL + 0.5); ax.set_xlim(-0.6, 0.5); ax.set_ylim(y + 0.2, 0)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4]); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("Difference in MCC (method minus comparator),\nmean with bootstrap 95% CI; p Holm-adjusted", fontsize=SMALL + 0.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "Fig7_forest")


# ------------------------------------------------------------------ Fig. 8  Study 3 per-unit MCC
def s3units():
    rows = [json.loads(l) for l in open("study3/results/results_s3.jsonl")]
    unit = defaultdict(list)
    for r in rows:
        if r.get("stage") == "main" and r["regime"] == "noniid" and r["config"] in ("FG", "B1", "B2", "B4", "B5"):
            unit[(r["p_label"], r["dataset"], r["config"], r["partition_seed"])].append(r["MCC"])
    meths = ["FG", "B4", "B1", "B2", "B5"]
    mname = {"FG": "FedGTCL", "B4": "FedProx", "B1": "FedAvg-\nGCN-GRU", "B2": "FedAvg-\nLSTM", "B5": "Pseudo-\nlabeling"}
    dsets = [("wustl", "WUSTL-IIoT-2021"), ("toniot", "TON_IoT-Network"), ("edgeiiot", "Edge-IIoTset")]
    fig, axes = plt.subplots(3, 2, figsize=(W, 7.4), sharey=True, sharex=True, gridspec_kw={"hspace": 0.24, "wspace": 0.07})
    rng = np.random.default_rng(0)
    for i, pl in enumerate((0.1, 0.5)):          # same jitter order as the original figure
        for j, (d, dn) in enumerate(dsets):
            ax = axes[j, i]
            ax.axhline(0, color=MUTED, lw=0.9)
            for x, m in enumerate(meths):
                v = np.array([np.mean(unit[(pl, d, m, ps)]) for ps in range(101, 111)])
                xs = x + rng.uniform(-0.17, 0.17, len(v))
                ax.scatter(xs, v, s=30, color=C[m], edgecolor="white", linewidth=0.7, zorder=3)
                ax.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color=INK, lw=2.2, zorder=4)
            ax.set_xticks(range(len(meths)))
            ax.set_xticklabels([NAME[m] for m in meths], fontsize=SMALL, rotation=30, ha="right", rotation_mode="anchor")
            ax.set_ylim(-0.25, 1.02); ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
            ax.set_title(f"{dn}, $p_L$ = {pl}", color=INK, loc="left")
    fig.supylabel("MCC (mean of two seeds)", fontsize=FS, color=INK2, x=0.01)
    save(fig, "Fig8_s3_units")


# ------------------------------------------------------------------ Fig. 9  Study 7 per-record classifiers
def study7():
    a = json.load(open("study7/results/analysis_s7.json"))["table"]
    M = [("LR", "LR-\nFedAvg"), ("MLP", "MLP-\nFedAvg"), ("MLPprox", "MLP-\nFedProx")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, ds in zip(axes, ["wustl", "edgeiiot", "xiiotid"]):
        for i, (m, _) in enumerate(M):
            for met, col, dx in (("MCC", BLUE, -0.15), ("Accuracy", ORANGE, 0.15)):
                t = a[f"temporal|full|{m}|{ds}"][met]; r = a[f"random|full|{m}|{ds}"][met]
                x = i + dx
                if abs(r - t) > 0.02:
                    ax.annotate("", xy=(x, r), xytext=(x, t), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, shrinkA=4, shrinkB=4, mutation_scale=11))
                ax.plot([x], [t], "o", ms=7, mfc="white", mec=col, mew=1.7, zorder=3)
                ax.plot([x], [r], "o", ms=7, color=col, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in M], linespacing=1.1)
        ax.set_title(NAMES[ds], color=INK); ax.set_ylim(0.3, 1.04); ax.set_xlim(-0.5, 2.5)
        ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    axes[0].set_ylabel("Score on validation records\n(mean of 20 runs)")
    h = [Line2D([], [], color=BLUE, marker="o", lw=1.8, label="MCC"), Line2D([], [], color=ORANGE, marker="o", lw=1.8, label="Accuracy"),
         Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, label="temporal split"), Line2D([], [], color=INK2, marker="o", lw=0, label="random split")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1), columnspacing=1.5)
    save(fig, "Fig9_study7_records")


# ------------------------------------------------------------------ Supplementary Fig. S1  training-loss curves
def supp_curves():
    rows = [json.loads(l) for l in open("study3/results/results_s3.jsonl")]
    g = defaultdict(list)
    for r in rows:
        if r.get("stage") == "main" and r["regime"] == "noniid" and r["config"] in ("B1", "B2", "B4", "B5"):
            g[(r["dataset"], r["p_label"], r["config"])].append([np.nan if x is None else x for x in r["loss_curve"]])
    meths = ["B4", "B1", "B2", "B5"]
    dsets = [("wustl", "WUSTL-IIoT-2021"), ("toniot", "TON_IoT-Network"), ("edgeiiot", "Edge-IIoTset")]
    fig, axes = plt.subplots(3, 2, figsize=(W, 7.0), sharey=True, sharex=True, gridspec_kw={"hspace": 0.3, "wspace": 0.07})
    x = np.arange(1, 51)
    for i, pl in enumerate((0.1, 0.5)):
        for j, (d, dn) in enumerate(dsets):
            ax = axes[j, i]
            ax.axvline(1.5, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
            for m in meths:
                a = np.nanmedian(np.array(g[(d, pl, m)], dtype=float), 0)
                ax.plot(x, a, color=C[m], lw=2.0, label=NAME[m])
            ax.set_ylim(0, 1.0); ax.set_xlim(0.5, 50.5); ax.grid(axis="y", color=GRID, lw=0.7)
            ax.set_title(f"{dn}, $p_L$ = {pl}", color=INK, loc="left")
    for ax in axes[2]: ax.set_xlabel("Round (20 local steps each)")
    fig.supylabel("Training loss (median of 20 runs)", fontsize=FS, color=INK2, x=0.01)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 2)))); l.append("pilot budget (30 steps)")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, -0.035), columnspacing=1.5)
    save(fig, "FigS1_training_loss")


# ------------------------------------------------------------------ Supplementary Fig. S2  attention cost
def supp_sparse():
    B = N["bench"]
    fig, ax = plt.subplots(figsize=(W * 0.72, 3.7))
    V = [b["V"] for b in B]
    for key, lab_, col in (("ref_dense_ms", "Dense-masked (paper's implementation)", ORANGE),
                           ("batched_dense_ms", "Dense-masked, batched", YELLOW),
                           ("sparse_ms", "Sparse (top-m neighbors only)", BLUE)):
        ax.plot(V, [b[key] for b in B], marker="o", ms=6.5, lw=2.2, color=col, mec="white", mew=1, label=lab_)
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500, 1000, 2000])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FixedFormatter(["10", "20", "50", "100", "200", "500", "1,000", "2,000"]))
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100])); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10", "100"]))
    ax.set_xlabel("Hosts per window |V|"); ax.set_ylabel("ms per T = 5 sequence\n(one CPU thread)")
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper left")
    save(fig, "FigS2_sparse_attention")


ALL = ["leakage", "studies", "audit", "study5", "study5_tests", "study6", "forest", "s3units", "study7"]
if __name__ == "__main__":
    for f in sys.argv[1:] or ALL:
        globals()[f]()
