"""Study 5 analysis (PLAN_STUDY5.md). Usage: python code/analyze_s5.py [results/results_s5.jsonl] [out_dir]
Merges the new runs with the reused chronological/full-budget cells of Studies 3-4, runs the six confirmatory tests
(Holm), and writes analysis_s5.json, tables_s5.md and fig_s5.png."""
import json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "results_s5.jsonl")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results")
DS = ["wustl", "toniot", "edgeiiot", "unsw"]
NAMES = {"wustl": "WUSTL-IIoT-2021", "toniot": "TON_IoT-Network", "edgeiiot": "Edge-IIoTset", "unsw": "UNSW-NB15"}
CONFIGS = ["FG", "B1", "B2", "B4", "B1F", "B2F", "B4F"]
LABEL = {"FG": "FedGTCL", "B1": "FedAvg-GCN-GRU", "B2": "FedAvg-LSTM", "B4": "FedProx", "B1F": "FedAvg-GCN-GRU (all labels)",
         "B2F": "FedAvg-LSTM (all labels)", "B4F": "FedProx (all labels)"}
ALPHA = 0.05

rows = []
for r in (json.loads(l) for l in open(res)):
    rows.append(dict(split=r["split"], budget=r["budget"], dataset=r["dataset"], ps=r["partition_seed"], seed=r["seed"],
                     config=r["config"], MCC=r["MCC"], degenerate=r["degenerate"], shared=r.get("val_share_with_train_windows"),
                     source="study5"))
n_new = len(rows)
for r in (json.loads(l) for l in open(os.path.join(ROOT, "reused", "results_s3.jsonl"))):
    if r["regime"] != "noniid":
        continue
    c = "B1F" if r["config"] == "REF" else r["config"]
    if (c in ("FG", "B1", "B2", "B4") and r["p_label"] == 0.5) or c == "B1F":
        rows.append(dict(split="chrono", budget="full", dataset=r["dataset"], ps=r["partition_seed"], seed=r["seed"], config=c,
                         MCC=r["MCC"], degenerate=r["degenerate"], shared=0.0, source="study3"))
for r in (json.loads(l) for l in open(os.path.join(ROOT, "reused", "results_s4.jsonl"))):
    if r["stage"] != "external":
        continue
    c = "B1F" if r["config"] == "REF" else r["config"]
    if (c in ("FG", "B1", "B2", "B4") and r["p_label"] == 0.5) or c == "B1F":
        rows.append(dict(split="chrono", budget="full", dataset="unsw", ps=r["partition_seed"], seed=r["seed"], config=c,
                         MCC=r["MCC"], degenerate=r["degenerate"], shared=0.0, source="study4"))

cell = defaultdict(list)
for r in rows:
    cell[(r["split"], r["budget"], r["config"], r["dataset"], r["ps"])].append(r)
unit = {k: float(np.mean([x["MCC"] for x in v])) for k, v in cell.items()}
UNITS = [(d, p) for d in DS for p in range(101, 111)]


def vec(sp, b, c):
    return np.array([unit.get((sp, b, c, d, p), np.nan) for d, p in UNITS])


def boot(d, n=10000):
    m = np.random.default_rng(1).choice(d, size=(n, len(d)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def wil(d):
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="zsplit").pvalue)


tests = []
for c in ("FG", "B1", "B2", "B4"):
    tests.append((f"H1 {LABEL[c]}: random minus chronological (1,000 steps, 50% labels)", vec("random", "full", c) - vec("chrono", "full", c)))
tests.append(("H2 (FedGTCL - FedProx) at 30 steps minus at 1,000 steps (chronological, equal labels)",
              (vec("chrono", "pilot", "FG") - vec("chrono", "pilot", "B4")) - (vec("chrono", "full", "FG") - vec("chrono", "full", "B4"))))
tests.append(("H3 (FedGTCL - FedAvg-GCN-GRU): pilot protocol minus corrected protocol",
              (vec("random", "pilot", "FG") - vec("random", "pilot", "B1F")) - (vec("chrono", "full", "FG") - vec("chrono", "full", "B1"))))
conf = []
for name, d in tests:
    ok = ~np.isnan(d); d = d[ok]
    conf.append(dict(test=name, n=int(len(d)), mean=float(d.mean()) if len(d) else None, median=float(np.median(d)) if len(d) else None,
                     ci95=boot(d) if len(d) else None, positive=int((d > 0).sum()), p=wil(d) if len(d) else None))
ps_ = [c["p"] if c["p"] is not None else 1.0 for c in conf]
order = sorted(range(len(ps_)), key=lambda i: ps_[i]); run = 0.0
for rk, i in enumerate(order):
    run = max(run, min(1.0, (len(ps_) - rk) * ps_[i])); conf[i]["p_holm"] = run; conf[i]["significant"] = run < ALPHA

COND = [("random", "pilot"), ("random", "full"), ("chrono", "pilot"), ("chrono", "full")]
CNAME = {("random", "pilot"): "random split, 30 steps (pilot)", ("random", "full"): "random split, 1,000 steps",
         ("chrono", "pilot"): "chronological, 30 steps", ("chrono", "full"): "chronological, 1,000 steps (corrected)"}
table = {}
for sp, b in COND:
    for c in CONFIGS:
        for d in DS + ["pooled"]:
            dl = DS if d == "pooled" else [d]
            v = [unit[(sp, b, c, x, p)] for x in dl for p in range(101, 111) if (sp, b, c, x, p) in unit]
            raw = [r for x in dl for p in range(101, 111) for r in cell.get((sp, b, c, x, p), [])]
            table[f"{sp}|{b}|{c}|{d}"] = dict(mean=float(np.mean(v)) if v else None, n=len(v),
                                              degenerate=f"{sum(r['degenerate'] != 'no' for r in raw)}/{len(raw)}")
shared = defaultdict(list)
for r in rows:
    if r["split"] == "random" and r["shared"] is not None:
        shared[r["dataset"]].append(r["shared"])
leak = {d: float(np.mean(v)) for d, v in shared.items()}

out = dict(n_new_runs=n_new, n_rows=len(rows), confirmatory=conf, table=table, val_sequences_sharing_train_windows=leak)
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, "analysis_s5.json"), "w"), indent=1)

L = ["# Study 5 results", "", f"{n_new} new runs + {len(rows) - n_new} reused runs (Studies 3-4).", "",
     "## Confirmatory tests (n = 40 units: 4 datasets x 10 partitions; Wilcoxon, Holm over 6)", "",
     "| Test | n | mean | median | positive | 95% CI | p (Holm) |", "|---|---|---|---|---|---|---|"]
for c in conf:
    if c["n"]:
        L.append(f"| {c['test']} | {c['n']} | {c['mean']:+.3f} | {c['median']:+.3f} | {c['positive']}/{c['n']} | "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['p_holm']:.4f}{' *' if c['significant'] else ''} |")
L += ["", "Share of validation sequences that share a window with a training sequence (random split): " +
      ", ".join(f"{NAMES[d]} {v:.1%}" for d, v in leak.items())]
for d in ["pooled"] + DS:
    L += ["", f"## Mean MCC, {NAMES.get(d, 'pooled over 4 datasets')} (degenerate runs in brackets)", "",
          "| Condition | " + " | ".join(LABEL[c] for c in CONFIGS) + " |", "|---" * (len(CONFIGS) + 1) + "|"]
    for sp, b in COND:
        cells = []
        for c in CONFIGS:
            e = table[f"{sp}|{b}|{c}|{d}"]
            cells.append("-" if e["mean"] is None else f"{e['mean']:.3f} ({e['degenerate']})")
        L.append(f"| {CNAME[(sp, b)]} | " + " | ".join(cells) + " |")
open(os.path.join(out_dir, "tables_s5.md"), "w").write("\n".join(L) + "\n")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(COND)); w = 0.2
    for i, c in enumerate(("FG", "B1", "B4", "B1F")):
        m = [table[f"{sp}|{b}|{c}|pooled"]["mean"] or np.nan for sp, b in COND]
        ax.bar(xs + (i - 1.5) * w, m, w, label=LABEL[c])
    ax.set_xticks(xs); ax.set_xticklabels([CNAME[k].replace(", ", "\n") for k in COND], fontsize=8)
    ax.set_ylabel("Mean MCC (non-IID, 4 datasets)"); ax.legend(fontsize=8); ax.axhline(0, color="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_s5.png"), dpi=200)
except Exception as ex:
    print("figure skipped:", ex)
print(open(os.path.join(out_dir, "tables_s5.md")).read())
