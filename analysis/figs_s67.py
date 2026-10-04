"""Figures for Studies 6 and 7."""
import json, sys
import numpy as np
ARGS = sys.argv[1:]; sys.argv = [sys.argv[0]]
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
exec(open(HERE + "/figs_new.py").read().replace("__file__", "HERE + '/figs_new.py'").split("# ------------------------------------------------------------------ leakage schematic")[0])
NAMES = {"wustl": "WUSTL-IIoT-2021", "edgeiiot": "Edge-IIoTset", "xiiotid": "X-IIoTID", "toniot": "TON_IoT-Network"}

def s7():
    a = json.load(open(os.path.join(ROOT, "study7", "results", "analysis_s7.json")))["table"]
    M = [("LR", "LR-FedAvg"), ("MLP", "MLP-FedAvg"), ("MLPprox", "MLP-FedProx")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, ds in zip(axes, ["wustl", "edgeiiot", "xiiotid"]):
        for i, (m, lab) in enumerate(M):
            for met, col, dx in (("MCC", BLUE, -0.13), ("Accuracy", ORANGE, 0.13)):
                t = a[f"temporal|full|{m}|{ds}"][met]; r = a[f"random|full|{m}|{ds}"][met]
                x = i + dx
                if abs(r - t) > 0.02: ax.annotate("", xy=(x, r), xytext=(x, t), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, shrinkA=3, shrinkB=3))
                ax.plot([x], [t], "o", ms=6, mfc="white", mec=col, mew=1.5, zorder=3)
                ax.plot([x], [r], "o", ms=6, color=col, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in M], fontsize=8)
        ax.set_title(NAMES[ds], fontsize=9, color=INK); ax.set_ylim(0.3, 1.04); ax.set_xlim(-0.5, 2.5)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Score on validation records (mean of 20 runs)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, marker="o", lw=1.6, label="MCC"), Line2D([], [], color=ORANGE, marker="o", lw=1.6, label="Accuracy"),
         Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, label="temporal split"), Line2D([], [], color=INK2, marker="o", lw=0, label="random split")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig_s7_records")

def s6():
    a = json.load(open(os.path.join(ROOT, "study6", "results", "analysis_s6.json")))["curve"]
    CK = [1, 2, 5, 10, 20, 50, 100, 150]
    M = [("FG", "FedGTCL", BLUE), ("B4", "FedProx", ORANGE), ("B1", "FedAvg-GCN-GRU", AQUA), ("B2", "FedAvg-LSTM", YELLOW), ("B5", "Pseudo-labelling", MAGENTA)]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, ds in zip(axes, ["pooled", "wustl", "toniot", "edgeiiot"]):
        for m, lab, col in M:
            ks = [k for k in CK if not (m == "FG" and k <= 5)]
            y = [a[f"{m}|{k}|{ds}"]["mean"] for k in ks]
            ax.plot([k * 20 for k in ks], y, "-o", color=col, lw=2 if m in ("FG", "B4") else 1.2, ms=3.5 if m in ("FG", "B4") else 2.5, label=lab)
        for x, t, ha, f in ((30, "pilot", "left", 1.08), (1000, "Studies 3–5", "right", 0.93)):
            ax.axvline(x, color=MUTED, lw=0.8, ls=(0, (3, 2))); ax.text(x * f, 0.97, t, fontsize=6.5, color=INK2, va="top", ha=ha)
        ax.set_xscale("log"); ax.set_xticks([20, 100, 1000, 3000]); ax.set_xticklabels(["20", "100", "1,000", "3,000"])
        ax.set_title("Pooled (3 datasets)" if ds == "pooled" else NAMES[ds], fontsize=9, color=INK); ax.set_ylim(-0.02, 1.0)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel("Local steps per client", fontsize=8)
    axes[0].set_ylabel("Mean validation MCC (non-IID, $p_L$ = 0.5)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig_s6_curves")

if __name__ == "__main__":
    for f in ARGS or ["s7"]:
        globals()[f]()
