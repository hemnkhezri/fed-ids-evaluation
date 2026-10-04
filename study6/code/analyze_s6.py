"""Study 6 analysis (PLAN_STUDY6.md): three confirmatory tests (Holm), curves, grid, mixed-model robustness check.
Usage: python code/analyze_s6.py  -> results/analysis_s6.json, results/tables_s6.md"""
import json, os, warnings
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DS = ["wustl", "toniot", "edgeiiot"]; UNITS = [(d, p) for d in DS for p in range(101, 111)]
CK = [1, 2, 5, 10, 20, 50, 100, 150]
rd = lambda f: [json.loads(l) for l in open(os.path.join(RES, f))] if os.path.exists(os.path.join(RES, f)) else []
cur, sel, con = rd("results_s6_curves.jsonl"), rd("results_s6_select.jsonl"), rd("results_s6_confirm.jsonl")

ck = defaultdict(list); ckdeg = defaultdict(list)
for r in cur:
    for k, v in r["checkpoints"].items():
        ck[(r["config"], int(k), r["dataset"], r["partition_seed"])].append(v["MCC"])
        ckdeg[(r["config"], int(k), r["dataset"])].append(v["degenerate"] != "no")
U = {k: float(np.mean(v)) for k, v in ck.items() if len(v) == 2}
def vec(m, k): return np.array([U.get((m, k, d, p), np.nan) for d, p in UNITS])

# Part B selection
best = {}
for m in ("FG", "B4", "B1"):
    sc = {lr: float(np.mean([r["MCC"] for r in sel if r["config"] == m and r["lr"] == lr])) for lr in (3e-4, 1e-3, 3e-3)
          if any(r["config"] == m and r["lr"] == lr for r in sel)}
    if sc:
        top = max(sc.values()); c = [lr for lr in sc if sc[lr] == top]
        best[m] = dict(lr=1e-3 if 1e-3 in c else c[0], inner_mean=sc)
cu = defaultdict(list)
for r in con:
    cu[(r["config"], r["dataset"], r["partition_seed"])].append(r["MCC"])
def tuned(m):
    if m in best and best[m]["lr"] != 1e-3:
        return np.array([np.mean(cu[(m, d, p)]) if len(cu.get((m, d, p), [])) == 2 else np.nan for d, p in UNITS])
    return vec(m, 50)

def boot(d, n=10000):
    mm = np.random.default_rng(1).choice(d, size=(n, len(d)), replace=True).mean(1)
    return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]
def wil(d): return 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="zsplit").pvalue)
def lmm(d):
    try:
        import pandas as pd, statsmodels.formula.api as smf
        df = pd.DataFrame(dict(d=d, g=[u[0] for u in UNITS]))
        m = smf.mixedlm("d ~ 1", df.dropna(), groups=df.dropna()["g"]).fit(reml=True)
        return dict(est=float(m.params["Intercept"]), ci=m.conf_int().loc["Intercept"].tolist(), p=float(m.pvalues["Intercept"]))
    except Exception as e:
        return dict(error=str(e))

T = [("H6.1 [FG-FedProx] at 3,000 minus at 1,000 steps", (vec("FG", 150) - vec("B4", 150)) - (vec("FG", 50) - vec("B4", 50))),
     ("H6.2 FG - FedProx at 3,000 steps", vec("FG", 150) - vec("B4", 150)),
     ("H6.3 FG - FedProx at 1,000 steps, re-tuned learning rates", tuned("FG") - tuned("B4"))]
conf = []
for name, d in T:
    d = d[~np.isnan(d)]
    if not len(d):
        conf.append(dict(test=name, n=0)); continue
    conf.append(dict(test=name, n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), positive=int((d > 0).sum()),
                     ci95=boot(d), p=wil(d), per_dataset={ds: float(np.nanmean(dd)) for ds, dd in
                     zip(DS, np.array_split(np.array(T[[t[0] for t in T].index(name)][1]), 3))}, lmm=lmm(T[[t[0] for t in T].index(name)][1])))
ps_ = [c.get("p", 1.0) for c in conf]; order = sorted(range(len(ps_)), key=lambda i: ps_[i]); run = 0.0
for rk, i in enumerate(order):
    run = max(run, min(1.0, (len(ps_) - rk) * ps_[i])); conf[i]["p_holm"] = run; conf[i]["significant"] = run < 0.05
if conf[0].get("n"):
    conf[0]["stable"] = bool(-0.10 < conf[0]["ci95"][0] and conf[0]["ci95"][1] < 0.10)

curve = {}
for m in ("FG", "B1", "B2", "B4", "B5"):
    for k in CK:
        for d in DS + ["pooled"]:
            dl = DS if d == "pooled" else [d]
            v = [U[(m, k, x, p)] for x in dl for p in range(101, 111) if (m, k, x, p) in U]
            dg = [z for x in dl for z in ckdeg.get((m, k, x), [])]
            curve[f"{m}|{k}|{d}"] = dict(mean=float(np.mean(v)) if v else None, n=len(v), collapsed=f"{sum(dg)}/{len(dg)}")
gap = {k: float(np.nanmean(vec("FG", k) - vec("B4", k))) for k in CK if not np.all(np.isnan(vec("FG", k) - vec("B4", k)))}
first_neg = next((k * 20 for k in CK if k > 5 and gap.get(k, 1) < 0), None)
change = {m: dict(mean=float(np.nanmean(vec(m, 150) - vec(m, 50))), p=wil((vec(m, 150) - vec(m, 50))[~np.isnan(vec(m, 150) - vec(m, 50))]))
          for m in ("FG", "B1", "B2", "B4", "B5") if not np.all(np.isnan(vec(m, 150)))}
out = dict(n_runs=dict(curves=len(cur), select=len(sel), confirm=len(con)), confirmatory=conf, selected=best, curve=curve,
           gap_FG_minus_FedProx=gap, first_negative_gap_steps=first_neg, change_1000_to_3000=change)
json.dump(out, open(os.path.join(RES, "analysis_s6.json"), "w"), indent=1)
L = ["# Study 6 results", "", "| Test | n | mean | 95% CI | positive | p (Holm) |", "|---|---|---|---|---|---|"]
for c in conf:
    if c.get("n"):
        L.append(f"| {c['test']} | {c['n']} | {c['mean']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['positive']} | {c['p_holm']:.4f} |")
L += ["", f"Selected learning rates: { {m: b['lr'] for m, b in best.items()} }", f"Gap FG-FedProx by round: {gap}",
      f"First negative gap (steps, after round 5): {first_neg}", f"Change 1000->3000: {change}", "", "| Method | " + " | ".join(str(k * 20) for k in CK) + " |", "|---" * (len(CK) + 1) + "|"]
for m in ("FG", "B1", "B2", "B4", "B5"):
    L.append(f"| {m} | " + " | ".join("-" if curve[f'{m}|{k}|pooled']['mean'] is None else f"{curve[f'{m}|{k}|pooled']['mean']:.3f}" for k in CK) + " |")
open(os.path.join(RES, "tables_s6.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
