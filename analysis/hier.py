"""Post hoc robustness analysis: confirmatory effects of Studies 3-5 re-estimated with the dataset as a cluster.
(a) linear mixed model d ~ 1 + (1 | dataset) (REML); (b) two-stage cluster bootstrap (datasets, then partitions);
(c) per-dataset means. Unit = mean MCC over two seeds of one (dataset, partition) cell, as in the sealed plans."""
import json, warnings
from collections import defaultdict
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
import os, tempfile
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
def units(rows, key):
    c = defaultdict(list)
    for r in rows: c[key(r)].append(r["MCC"])
    return {k: float(np.mean(v)) for k, v in c.items()}

# ---- Study 5 (reuse the analysis script's merge by importing its unit dict)
import runpy, sys
sys.argv = ["x", f"{R}/study5/results/results_s5.jsonl", tempfile.mkdtemp()]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(f"{R}/study5/code/analyze_s5.py")
unit5 = g["unit"]; DS5 = ["wustl", "toniot", "edgeiiot", "unsw"]
def v5(sp, b, c): return {(d, p): unit5.get((sp, b, c, d, p)) for d in DS5 for p in range(101, 111)}
def sub(a, b): return {k: a[k] - b[k] for k in a if a.get(k) is not None and b.get(k) is not None}
T = {}
for c, n in [("FG", "FedGTCL"), ("B1", "FedAvg-GCN-GRU"), ("B2", "FedAvg-LSTM"), ("B4", "FedProx")]:
    T[f"S5 H1 leakage, {n}"] = sub(v5("random", "full", c), v5("chrono", "full", c))
T["S5 H2 budget: (FG-FedProx) 30 minus 1,000 steps"] = sub(sub(v5("chrono", "pilot", "FG"), v5("chrono", "pilot", "B4")), sub(v5("chrono", "full", "FG"), v5("chrono", "full", "B4")))
T["S5 H3 pilot vs corrected protocol"] = sub(sub(v5("random", "pilot", "FG"), v5("random", "pilot", "B1F")), sub(v5("chrono", "full", "FG"), v5("chrono", "full", "B1")))

# ---- Study 3 confirmatory (non-IID)
r3 = [json.loads(l) for l in open(f"{R}/study3/results/results_s3.jsonl")]
u3 = units([r for r in r3 if r["regime"] == "noniid"], lambda r: (r["config"], r["p_label"], r["dataset"], r["partition_seed"]))
def v3(c, pl): return {(d, p): u3.get((c, pl, d, p)) for d in ["wustl", "toniot", "edgeiiot"] for p in range(101, 111)}
for pl in (0.1, 0.5):
    T[f"S3 FedGTCL - FedProx, p_L={pl}"] = sub(v3("FG", pl), v3("B4", pl))
    T[f"S3 FedGTCL - pseudo-labelling, p_L={pl}"] = sub(v3("FG", pl), v3("B5", pl))
# ---- Study 4 internal
r4 = [json.loads(l) for l in open(f"{R}/study4/results/results_s4.jsonl")]
u4 = units([r for r in r4 if r["stage"] == "internal"], lambda r: (r["config"], r["p_label"], r["dataset"], r["partition_seed"]))
def v4(c, pl): return {(d, p): u4.get((c, pl, d, p)) for d in ["wustl", "toniot", "edgeiiot"] for p in range(111, 121)}
for pl in (0.1, 0.5):
    T[f"S4 internal C4 - FedProx, p_L={pl}"] = sub(v4("C4", pl), v4("B4", pl))
    T[f"S4 internal C4 - pseudo-labelling, p_L={pl}"] = sub(v4("C4", pl), v4("B5", pl))

rng = np.random.default_rng(7)
out = []
for name, dct in T.items():
    df = pd.DataFrame([(d, p, x) for (d, p), x in dct.items()], columns=["ds", "ps", "d"])
    ds = sorted(df.ds.unique()); per = {k: df[df.ds == k].d.values for k in ds}
    try:
        m = smf.mixedlm("d ~ 1", df, groups=df["ds"]).fit(reml=True)
        est, (lo, hi), p, sd = float(m.params["Intercept"]), m.conf_int().loc["Intercept"].tolist(), float(m.pvalues["Intercept"]), float(np.sqrt(max(m.cov_re.iloc[0, 0], 0)))
    except Exception as e:
        est = lo = hi = p = sd = float("nan")
    B = []
    for _ in range(10000):
        pick = rng.choice(ds, len(ds), replace=True)
        B.append(np.mean([rng.choice(per[k], len(per[k]), replace=True).mean() for k in pick]))
    out.append(dict(test=name, n=len(df), n_datasets=len(ds), mean=float(df.d.mean()), lmm=est, lmm_ci=[lo, hi], lmm_p=p, sd_dataset=sd,
                    cboot_ci=[float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))],
                    per_dataset={k: round(float(per[k].mean()), 3) for k in ds}, same_sign=int(sum(np.sign(per[k].mean()) == np.sign(df.d.mean()) for k in ds))))
json.dump(out, open(f"{R}/analysis/hier.json", "w"), indent=1)
for o in out:
    print(f"{o['test'][:52]:52s} n={o['n']:2d} mean={o['mean']:+.3f} LMM={o['lmm']:+.3f} [{o['lmm_ci'][0]:+.3f},{o['lmm_ci'][1]:+.3f}] p={o['lmm_p']:.4f} "
          f"cboot=[{o['cboot_ci'][0]:+.3f},{o['cboot_ci'][1]:+.3f}] sign {o['same_sign']}/{o['n_datasets']} {o['per_dataset']}")
