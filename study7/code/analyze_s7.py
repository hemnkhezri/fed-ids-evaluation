"""Study 7 analysis (PLAN_STUDY7.md). Usage: python code/analyze_s7.py -> results/analysis_s7.json, tables_s7.md"""
import json, os, warnings
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); RES = os.path.join(ROOT, "results")
DS = ["wustl", "edgeiiot", "xiiotid"]; UNITS = [(d, p) for d in DS for p in range(101, 111)]
NAME = {"LR": "LR-FedAvg", "MLP": "MLP-FedAvg", "MLPprox": "MLP-FedProx"}
rows = [json.loads(l) for l in open(os.path.join(RES, "results_s7.jsonl"))]
cell = defaultdict(list)
for r in rows: cell[(r["split"], r["budget"], r["method"], r["dataset"], r["partition_seed"])].append(r)
def vec(sp, b, m, met="MCC"):
    return np.array([np.mean([x[met] for x in cell[(sp, b, m, d, p)]]) if len(cell.get((sp, b, m, d, p), [])) == 2 else np.nan for d, p in UNITS])
def boot(d, n=10000):
    mm = np.random.default_rng(1).choice(d, size=(n, len(d)), replace=True).mean(1); return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]
def wil(d): return 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="zsplit").pvalue)
def lmm(d):
    try:
        import pandas as pd, statsmodels.formula.api as smf
        df = pd.DataFrame(dict(d=d, g=[u[0] for u in UNITS])).dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); m = smf.mixedlm("d ~ 1", df, groups=df["g"]).fit(reml=True)
        return dict(est=float(m.params["Intercept"]), ci=m.conf_int().loc["Intercept"].tolist(), p=float(m.pvalues["Intercept"]))
    except Exception as e: return dict(error=str(e))
def summ(name, d):
    full = d; d = d[~np.isnan(d)]
    return dict(test=name, n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), positive=int((d > 0).sum()), ci95=boot(d),
                p=wil(d), per_dataset={ds: float(np.nanmean(x)) for ds, x in zip(DS, np.array_split(full, 3))}, lmm=lmm(full))
conf = [summ(f"H7.{i + 1} {NAME[m]}: MCC random minus temporal (1,000 steps)", vec("random", "full", m) - vec("temporal", "full", m))
        for i, m in enumerate(["MLP", "MLPprox", "LR"])]
ps_ = [c["p"] for c in conf]; order = sorted(range(3), key=lambda i: ps_[i]); run = 0.0
for rk, i in enumerate(order):
    run = max(run, min(1.0, (3 - rk) * ps_[i])); conf[i]["p_holm"] = run; conf[i]["significant"] = run < 0.05
desc = {}
for met in ("MCC", "Accuracy", "F1"):
    for b in ("short", "full"):
        for m in NAME:
            desc[f"{met}|{b}|{m}"] = summ(f"{met} random-temporal {b} {m}", vec("random", b, m, met) - vec("temporal", b, m, met))
table = {}
for sp in ("random", "temporal"):
    for b in ("short", "full"):
        for m in NAME:
            for d in DS + ["pooled"]:
                rr = [r for r in rows if r["split"] == sp and r["budget"] == b and r["method"] == m and (d == "pooled" or r["dataset"] == d)]
                if rr:
                    table[f"{sp}|{b}|{m}|{d}"] = {k: float(np.mean([r[k] for r in rr])) for k in ("MCC", "Accuracy", "F1")} | dict(
                        collapsed=f"{sum(r['degenerate'] != 'no' for r in rr)}/{len(rr)}", n=len(rr))
dup = {f"{sp}|{d}": float(np.mean([r["val_near_duplicate_share"] for r in rows if r["split"] == sp and r["dataset"] == d]))
       for sp in ("random", "temporal") for d in DS if any(r["split"] == sp and r["dataset"] == d for r in rows)}
rank = {f"{sp}|{b}": sorted(NAME, key=lambda m: -table.get(f"{sp}|{b}|{m}|pooled", {"MCC": -9})["MCC"]) for sp in ("random", "temporal") for b in ("short", "full")}
json.dump(dict(n_runs=len(rows), confirmatory=conf, descriptive=desc, table=table, near_duplicate=dup, ranking=rank),
          open(os.path.join(RES, "analysis_s7.json"), "w"), indent=1)
L = ["# Study 7 results", "", "| Test | n | mean | 95% CI | positive | p (Holm) | per dataset |", "|---|---|---|---|---|---|---|"]
for c in conf:
    L.append(f"| {c['test']} | {c['n']} | {c['mean']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['positive']} | {c['p_holm']:.4f} | {c['per_dataset']} |")
L += ["", f"Near-duplicate share: {dup}", f"Ranking: {rank}", "", "| split | budget | method | dataset | MCC | Acc | F1 | collapsed |", "|---|---|---|---|---|---|---|---|"]
for k, v in table.items():
    L.append("| " + " | ".join(k.split("|")) + f" | {v['MCC']:.3f} | {v['Accuracy']:.3f} | {v['F1']:.1f} | {v['collapsed']} |")
open(os.path.join(RES, "tables_s7.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L[:8]))
