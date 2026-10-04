"""PHASE3B_PLAN.md analysis. Usage: python phase3b_analyze.py <dir with phase2 + phase3b result jsonl files>"""
import sys, glob, json
from collections import defaultdict
import numpy as np
from scipy import stats
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/**/*.jsonl", recursive=True) for l in open(f)]
rows = [r for r in rows if r.get("tag") in ("phase2", "phase3b") and r["p_label"] == 0.1]
cell = defaultdict(list)
seen = set()
for r in rows:
    k = (r["dataset"], r["partition_seed"], r["seed"], r["config"])
    if k in seen:
        continue
    seen.add(k); cell[(r["dataset"], r["partition_seed"], r["config"])].append(r["MCC"])
def pm(dss, cfg):
    return {(d, p): float(np.mean(v)) for (d, p, c), v in cell.items() if d in dss and c == cfg and len(v) == 2}
def test(dss, a, b):
    A, B = pm(dss, a), pm(dss, b); ks = sorted(set(A) & set(B))
    x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
    p = 1.0 if not ks or np.all(x == y) else stats.wilcoxon(x, y, zero_method="zsplit").pvalue
    return dict(n=len(ks), mean_a=float(x.mean()) if ks else None, mean_b=float(y.mean()) if ks else None,
                median_diff=float(np.median(x - y)) if ks else None, p=float(p))
H = {"H_E1": test(["toniot", "edgeiiot"], "FG", "B5"), "H_E2": test(["xiiotid"], "FG", "B4"), "H_E3": test(["xiiotid"], "FG", "B5")}
exp = {"H_E1": 20, "H_E2": 10, "H_E3": 10}
ps = np.array([H[h]["p"] for h in H]); o = np.argsort(ps); adj = np.empty(3); run = 0
for r_, i in enumerate(o):
    run = max(run, min(1, (3 - r_) * ps[i])); adj[i] = run
for (h, r), q in zip(H.items(), adj):
    r["p_holm"] = float(q)
    r["decision"] = "FG > comp" if q < 0.05 and r["median_diff"] > 0 else "FG < comp" if q < 0.05 else "no detectable difference"
complete = all(H[h]["n"] == exp[h] for h in H)
print("COMPLETE" if complete else "INCOMPLETE", {h: H[h]["n"] for h in H})
for h, r in H.items():
    print(h, r)
expl = {}
for ds in ("wustl", "toniot", "edgeiiot", "cicapt", "xiiotid"):
    for b in ("B1", "B2", "B3", "B4", "B5"):
        t = test([ds], "FG", b)
        if t["n"]:
            expl[f"{ds}|FG_vs_{b}"] = t; print("EXPL", ds, b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
json.dump(dict(complete=complete, confirmatory=H, exploratory=expl), open("phase3b_analysis.json", "w"), indent=1)
