"""Study 4 analysis (PLAN_STUDY4.md). Usage: python code/analyze_s4.py [results/results_s4.jsonl] [out_dir]
Writes analysis_s4.json, tables_s4.md and fig_s4_mcc.png, and prints the pre-specified decision."""
import json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
res_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "results_s4.jsonl")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results")
DELTA, ALPHA_SUP, ALPHA_NI = 0.05, 0.05, 0.025
NAMES = {"wustl": "WUSTL-IIoT-2021", "toniot": "TON_IoT-Network", "edgeiiot": "Edge-IIoTset", "unsw": "UNSW-NB15"}
LABEL = {"C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4", "FG": "FedGTCL (original)", "B1": "FedAvg-GCN-GRU",
         "B2": "FedAvg-LSTM", "B4": "FedProx", "B5": "Pseudo-label", "REF": "Full-label reference"}
rows = [json.loads(l) for l in open(res_path)]
sel_path = os.path.join(out_dir, "selected.json")
SEL = json.load(open(sel_path))["selected"] if os.path.exists(sel_path) else None

cell = defaultdict(list)
for r in rows:
    cell[(r["stage"], r["dataset"], r["partition_seed"], r["p_label"], r["config"])].append(r)
unit = {k: float(np.mean([x["MCC"] for x in v])) for k, v in cell.items()}


def paired(stage, datasets, parts, pl, a, b):
    d = [unit[(stage, ds, ps, pl, a)] - unit[(stage, ds, ps, pl, b)] for ds in datasets for ps in parts
         if (stage, ds, ps, pl, a) in unit and (stage, ds, ps, pl, b) in unit]
    return np.array(d)


def boot_ci(d, n=10000, seed=1):
    m = np.random.default_rng(seed).choice(d, size=(n, len(d)), replace=True).mean(1)
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


def family(stage, datasets, parts):
    fam = []
    for pl in (0.1, 0.5):
        for b in ("B4", "B5"):
            d = paired(stage, datasets, parts, pl, SEL, b)
            if len(d) == 0:
                continue
            fam.append(dict(p_label=pl, comparator=b, n=len(d), mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                            ci95=boot_ci(d), wins=int((d > 0).sum()), p_sup=wil(d), p_ni=wil(d + DELTA, "greater")))
    if not fam:
        return fam, None, "not run"
    for c, a in zip(fam, holm([c["p_sup"] for c in fam])):
        c["p_sup_holm"] = a; c["superior"] = a < ALPHA_SUP and c["median_diff"] > 0
        c["inferior"] = a < ALPHA_SUP and c["median_diff"] < 0
    for c, a in zip(fam, holm([c["p_ni"] for c in fam])):
        c["p_ni_holm"] = a; c["noninferior"] = a < ALPHA_NI
    lab = lambda c: f"p_L={c['p_label']} vs {LABEL[c['comparator']]}"
    if len(fam) < 4:
        return fam, "incomplete", f"only {len(fam)} of 4 comparisons available"
    if any(c["superior"] for c in fam):
        return fam, "A", "superior in: " + ", ".join(lab(c) for c in fam if c["superior"])
    if all(c["noninferior"] for c in fam):
        return fam, "B", f"non-inferior (margin {DELTA} MCC) in all four comparisons"
    if any(c["noninferior"] for c in fam):
        return fam, "B-partial", "non-inferior only in: " + ", ".join(lab(c) for c in fam if c["noninferior"])
    return fam, "C", "neither superior nor non-inferior"


out = dict(selected=SEL, selection=json.load(open(sel_path)) if SEL else None)
for fname, stage, dsl, parts in (("internal", "internal", ["wustl", "toniot", "edgeiiot"], range(111, 121)),
                                 ("external", "external", ["unsw"], range(101, 111))):
    fam, oc, txt = family(stage, dsl, parts) if SEL else ([], None, "no selection")
    out[fname] = dict(confirmatory=fam, outcome=oc, decision=txt)
gate_p = os.path.join(out_dir, "gate.json")
out["gate"] = json.load(open(gate_p)) if os.path.exists(gate_p) else None
el_p = os.path.join(out_dir, "unsw_eligibility.json")
out["eligibility"] = {k: v for k, v in json.load(open(el_p)).items() if k != "partitions"} if os.path.exists(el_p) else None


def table(stage, datasets, parts, configs, pls):
    L = []
    for pl in pls:
        L += ["", f"### {stage}, p_L = {pl} (mean MCC over partitions; degenerate runs in brackets)", "",
              "| Dataset | " + " | ".join(LABEL[c] for c in configs) + " |", "|---" * (len(configs) + 1) + "|"]
        for ds in datasets + (["pooled"] if len(datasets) > 1 else []):
            dsl = datasets if ds == "pooled" else [ds]; cells = []
            for c in configs:
                v = [unit[(stage, d, ps, pl, c)] for d in dsl for ps in parts if (stage, d, ps, pl, c) in unit]
                raw = [x for d in dsl for ps in parts for x in cell.get((stage, d, ps, pl, c), [])]
                deg = sum(x["degenerate"] != "no" for x in raw)
                cells.append("-" if not v else f"{np.mean(v):.3f} ({deg}/{len(raw)})")
            L.append(f"| {NAMES.get(ds, 'Pooled')} | " + " | ".join(cells) + " |")
    return L


L = ["# Study 4 results", ""]
if SEL:
    s = out["selection"]
    L += [f"Selected candidate: **{SEL}** ({s['rule']}).", "", "| Candidate | mean MCC (60 units) | degenerate runs |",
          "|---|---|---|"] + [f"| {c} | {s['scores'][c]:.3f} | {s['degenerate'].get(c, 0)}/120 |" for c in ("C1", "C2", "C3", "C4")]
    L += table("select", ["wustl", "toniot", "edgeiiot"], range(101, 111), ["C1", "C2", "C3", "C4"], (0.1, 0.5))
for fname in ("external", "internal"):
    f = out[fname]
    L += ["", f"## {fname.capitalize()} confirmatory family: outcome **{f['outcome']}** ({f['decision']})", "",
          "| p_L | vs | n | mean diff | median diff | wins | 95% CI | p sup. (Holm) | p NI, margin 0.05 (Holm) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in f["confirmatory"]:
        L.append(f"| {c['p_label']} | {LABEL[c['comparator']]} | {c['n']} | {c['mean_diff']:+.3f} | {c['median_diff']:+.3f} | "
                 f"{c['wins']}/{c['n']} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['p_sup_holm']:.4f} | {c['p_ni_holm']:.4f} |")
if SEL:
    L += table("internal", ["wustl", "toniot", "edgeiiot"], range(111, 121), [SEL, "B4", "B5"], (0.1, 0.5))
    L += table("external", ["unsw"], range(101, 111), [SEL, "B4", "B5", "FG", "B1", "B2"], (0.1, 0.5))
    ref = [unit[k] for k in unit if k[0] == "external" and k[4] == "REF"]
    if ref:
        L += ["", f"Full-label reference on UNSW-NB15 (non-IID): mean MCC {np.mean(ref):.3f} over {len(ref)} partitions."]
if out["gate"]:
    L += ["", f"Learnability gate: mean MCC {out['gate']['mean_MCC']:.3f} (threshold {out['gate']['threshold']}), "
          f"{'passed' if out['gate']['passed'] else 'NOT passed'}."]
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, "analysis_s4.json"), "w"), indent=1)
open(os.path.join(out_dir, "tables_s4.md"), "w").write("\n".join(L) + "\n")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels = [("internal", d, range(111, 121)) for d in ("wustl", "toniot", "edgeiiot")] + [("external", "unsw", range(101, 111))]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (st, ds, parts) in zip(axes, panels):
        data, labels = [], []
        for pl in (0.1, 0.5):
            for m in (SEL, "B4", "B5"):
                data.append([unit[(st, ds, ps, pl, m)] for ps in parts if (st, ds, ps, pl, m) in unit]); labels.append(f"{m}\n{pl}")
        if any(data):
            ax.boxplot([d if d else [np.nan] for d in data], tick_labels=labels, showmeans=True)
        ax.set_title(f"{NAMES[ds]} ({st})"); ax.axhline(0, color="gray", lw=0.8)
    axes[0].set_ylabel("MCC (non-IID, mean of 2 seeds per partition)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_s4_mcc.png"), dpi=200)
except Exception as ex:
    print("figure skipped:", ex)
print(open(os.path.join(out_dir, "tables_s4.md")).read())
