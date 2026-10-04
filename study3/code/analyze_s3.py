"""Study 3 analysis (PLAN_STUDY3.md). Usage: python code/analyze_s3.py [results/results_s3.jsonl] [out_dir]
Writes analysis_s3.json, tables_s3.md and fig_s3_mcc.png, and prints the pre-specified decision."""
import json, math, os, random, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
res_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "results_s3.jsonl")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results")
DELTA, ALPHA_SUP, ALPHA_NI = 0.05, 0.05, 0.025
DATASETS = ["wustl", "toniot", "edgeiiot"]; NAMES = {"wustl": "WUSTL-IIoT-2021", "toniot": "TON_IoT-Network", "edgeiiot": "Edge-IIoTset"}
rows = [json.loads(l) for l in open(res_path)]

# unit of analysis: mean over model seeds of one (dataset, partition, regime, p_label, config) cell
cell = defaultdict(list)
for r in rows:
    cell[(r["dataset"], r["partition_seed"], r["regime"], r["p_label"], r["config"])].append(r)
unit = {k: float(np.mean([x["MCC"] for x in v])) for k, v in cell.items()}


def paired(regime, pl, a, b, datasets=DATASETS, pl_b=None):
    pl_b = pl if pl_b is None else pl_b
    d = []
    for ds in datasets:
        for ps in range(101, 111):
            ka, kb = (ds, ps, regime, pl, a), (ds, ps, regime, pl_b, b)
            if ka in unit and kb in unit:
                d.append(unit[ka] - unit[kb])
    return np.array(d)


def boot_ci(d, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(d, alternative="two-sided"):
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, zero_method="zsplit", alternative=alternative).pvalue)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


# ---------------- confirmatory family
fam = [(pl, b) for pl in (0.1, 0.5) for b in ("B4", "B5")]
conf = []
for pl, b in fam:
    d = paired("noniid", pl, "FG", b)
    conf.append(dict(p_label=pl, comparator=b, n=len(d), mean_diff=float(d.mean()) if len(d) else None,
                     median_diff=float(np.median(d)) if len(d) else None, ci95=boot_ci(d) if len(d) else None,
                     p_sup=wil(d), p_ni=wil(d + DELTA, "greater")))
for c, a in zip(conf, holm([c["p_sup"] for c in conf])):
    c["p_sup_holm"] = a; c["superior"] = a < ALPHA_SUP and c["median_diff"] > 0; c["inferior"] = a < ALPHA_SUP and c["median_diff"] < 0
for c, a in zip(conf, holm([c["p_ni"] for c in conf])):
    c["p_ni_holm"] = a; c["noninferior"] = a < ALPHA_NI

if any(c["superior"] for c in conf):
    outcome = "A"; text = "FedGTCL is superior in: " + ", ".join(f"p_L={c['p_label']} vs {c['comparator']}" for c in conf if c["superior"])
elif all(c["noninferior"] for c in conf):
    outcome = "B"; text = f"FedGTCL is non-inferior (margin {DELTA} MCC) to FedProx and pseudo-labelling at both label fractions"
elif any(c["noninferior"] for c in conf):
    outcome = "B-partial"; text = "Non-inferiority holds only in: " + ", ".join(f"p_L={c['p_label']} vs {c['comparator']}" for c in conf if c["noninferior"])
else:
    outcome = "C"; text = "FedGTCL is neither superior nor non-inferior to the tuned baselines"

# ---------------- exploratory
expl = {}
for regime in ("noniid", "iid"):
    for pl in (0.1, 0.5):
        for ds in DATASETS + ["pooled"]:
            dsl = DATASETS if ds == "pooled" else [ds]
            for m in ("FG", "B1", "B2", "B4", "B5"):
                v = [unit[(d, ps, regime, pl, m)] for d in dsl for ps in range(101, 111) if (d, ps, regime, pl, m) in unit]
                raw = [x for (d, ps, rg, p, cf), xs in cell.items() if d in dsl and rg == regime and p == pl and cf == m for x in xs]
                deg = sum(x["degenerate"] != "no" for x in raw)
                expl[f"{regime}|{pl}|{ds}|{m}"] = dict(n_units=len(v), mean=float(np.mean(v)) if v else None, sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                                     degenerate=f"{deg}/{len(raw)}")
ref = {}
for regime in ("noniid", "iid"):
    for ds in DATASETS:
        v = [unit[(ds, ps, regime, 1.0, "REF")] for ps in range(101, 111) if (ds, ps, regime, 1.0, "REF") in unit]
        ref[f"{regime}|{ds}"] = float(np.mean(v)) if v else None
abl = {}
for m in ("FG_noCon", "FG_noOpt"):
    d = paired("noniid", 0.5, "FG", m)
    abl[m] = dict(n=len(d), mean_diff=float(d.mean()) if len(d) else None, p_two_sided=wil(d) if len(d) else None)

out = dict(outcome=outcome, decision=text, delta=DELTA, confirmatory=conf, exploratory=expl, full_label_reference=ref, ablation=abl,
           n_rows=len(rows))
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, "analysis_s3.json"), "w"), indent=1)

L = ["# Study 3 results", "", f"Outcome **{outcome}**: {text}.", "", "## Confirmatory tests (non-IID, pooled over 3 datasets, unit = partition)", "",
     "| p_L | FG vs | n | mean diff | median diff | 95% CI (bootstrap) | p superiority (Holm) | p non-inferiority, margin 0.05 (Holm) |",
     "|---|---|---|---|---|---|---|---|"]
for c in conf:
    L.append(f"| {c['p_label']} | {c['comparator']} | {c['n']} | {c['mean_diff']:+.3f} | {c['median_diff']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
             f"{c['p_sup_holm']:.4f} | {c['p_ni_holm']:.4f} |")
for regime in ("noniid", "iid"):
    for pl in (0.1, 0.5):
        L += ["", f"## Mean MCC, {regime}, p_L={pl} (degenerate runs in brackets)", "", "| Dataset | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | Pseudo-label | full-label reference |", "|---|---|---|---|---|---|---|"]
        for ds in DATASETS + ["pooled"]:
            cells = []
            for m in ("FG", "B1", "B2", "B4", "B5"):
                e = expl[f"{regime}|{pl}|{ds}|{m}"]
                cells.append("-" if e["mean"] is None else f"{e['mean']:.3f} ({e['degenerate']})")
            r_ = ref.get(f"{regime}|{ds}") if ds != "pooled" else None
            L.append(f"| {NAMES.get(ds, 'Pooled')} | " + " | ".join(cells) + f" | {'-' if r_ is None else f'{r_:.3f}'} |")
L += ["", "## Ablation (non-IID, p_L=0.5): FedGTCL minus variant", ""] + [f"- {m}: mean diff {v['mean_diff']:+.3f}, n={v['n']}, p={v['p_two_sided']:.4f}" for m, v in abl.items() if v["n"]]
open(os.path.join(out_dir, "tables_s3.md"), "w").write("\n".join(L) + "\n")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        data, labels = [], []
        for pl in (0.1, 0.5):
            for m in ("FG", "B4", "B5", "B1"):
                data.append([unit[(ds, ps, "noniid", pl, m)] for ps in range(101, 111) if (ds, ps, "noniid", pl, m) in unit]); labels.append(f"{m}\n{pl}")
        ax.boxplot(data, tick_labels=labels, showmeans=True); ax.set_title(NAMES[ds]); ax.axhline(0, color="gray", lw=0.8)
    axes[0].set_ylabel("MCC (non-IID, mean of 2 seeds per partition)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_s3_mcc.png"), dpi=200)
except Exception as ex:
    print("figure skipped:", ex)
print(f"Outcome {outcome}: {text}")
print(open(os.path.join(out_dir, "tables_s3.md")).read())
