"""Phase-2/3 analysis exactly as PHASE2_PLAN.md. Usage: python phase2_analyze.py results_dir"""
import sys, glob, json, math
from collections import defaultdict
import numpy as np
from scipy import stats

rows = []
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    rows += [json.loads(l) for l in open(f)]
rows = [r for r in rows if r.get("tag") == "phase2"]
cell = defaultdict(list)
for r in rows:
    cell[(r["dataset"], r["p_label"], r["partition_seed"], r["config"])].append(r["MCC"])


def pmean(ds_list, p, cfg):
    out = {}
    for (ds, pl, ps, c), v in cell.items():
        if ds in ds_list and pl == p and c == cfg and len(v) == 2:
            out[(ds, ps)] = float(np.mean(v))
    return out


def test(ds_list, p=0.1, a="FG", b="B4"):
    A = pmean(ds_list, p, a); B = pmean(ds_list, p, b); ks = sorted(set(A) & set(B))
    x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks]); d = x - y
    pv = 1.0 if len(ks) == 0 or np.all(d == 0) else stats.wilcoxon(x, y, zero_method="zsplit").pvalue
    return dict(n=len(ks), mean_a=float(x.mean()) if len(ks) else None, mean_b=float(y.mean()) if len(ks) else None,
                median_diff=float(np.median(d)) if len(ks) else None, p=float(pv))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


H = {"H_A": test(["wustl"]), "H_B": test(["toniot", "edgeiiot"]), "H_C": test(["cicapt"])}
exp_n = {"H_A": 10, "H_B": 20, "H_C": 10}
complete = all(H[h]["n"] == exp_n[h] for h in H)
adj = holm(np.array([H[h]["p"] for h in H]))
for (h, r), q in zip(H.items(), adj):
    r["p_holm"] = float(q)
    r["decision"] = ("FG > B4" if q < 0.05 and r["median_diff"] > 0 else "FG < B4" if q < 0.05 and r["median_diff"] < 0
                     else "no detectable difference")
print("COMPLETE" if complete else "INCOMPLETE - confirmatory family not final", {h: H[h]["n"] for h in H})
for h, r in H.items():
    print(h, r)
expl = {}
for ds in ("wustl", "toniot", "edgeiiot", "cicapt"):
    for b in ("B1", "B2", "B3", "B4"):
        for p in (0.1, 0.5):
            t = test([ds], p, "FG", b)
            if t["n"]:
                expl[f"{ds}|p{p}|FG_vs_{b}"] = t
    for p in (0.1, 0.5):
        for c in ("FG", "B1", "B2", "B3", "B4"):
            v = [m for (d_, pl, ps, cc), vs in cell.items() if d_ == ds and pl == p and cc == c for m in vs]
            if v:
                k = sum(m > 0 for m in v); n = len(v)
                expl[f"{ds}|p{p}|{c}|summary"] = dict(n_runs=n, mean_mcc=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if n > 1 else 0,
                                                      discriminative=k)
json.dump(dict(complete=complete, confirmatory=H, exploratory=expl), open("phase2_analysis.json", "w"), indent=1)
for k, v in expl.items():
    print("EXPL", k, v)
