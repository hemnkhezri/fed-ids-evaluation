"""Pre-registered confirmatory analysis (plan v2 + addendum) and labelled exploratory analyses."""
import json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results_v2r.jsonl"))]
D = defaultdict(dict)
for r in rows:
    D[(r["dataset"], r["partition"], r["p_label"], r["arm"])][r["seed"]] = r

DS = ["wustl", "toniot", "edgeiiot"]
ARMS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired(ds, a, b, part="noniid", p=0.5, seeds=range(1, 11)):
    A = D[(ds, part, p, a)]; B = D[(ds, part, p, b)]
    s = [x for x in seeds if x in A and x in B]
    da = np.array([A[x]["MCC"] for x in s]); db = np.array([B[x]["MCC"] for x in s])
    diff = da - db
    if len(s) == 0:
        return None
    if np.all(diff == 0):
        pval = 1.0
    else:
        pval = stats.wilcoxon(da, db, zero_method="zsplit", alternative="two-sided").pvalue
    return dict(n=len(s), median_diff=float(np.median(diff)), mean_diff=float(diff.mean()), p=float(pval),
                mean_a=float(da.mean()), mean_b=float(db.mean()))


def holm(ps, alpha=0.05):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def mcnemar_disc(ds, a, b, part="noniid", p=0.5, seeds=range(1, 11)):
    A = D[(ds, part, p, a)]; B = D[(ds, part, p, b)]
    s = [x for x in seeds if x in A and x in B]
    n10 = sum(A[x]["MCC"] > 0 and not B[x]["MCC"] > 0 for x in s)
    n01 = sum(B[x]["MCC"] > 0 and not A[x]["MCC"] > 0 for x in s)
    pv = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, pv


out = {"confirmatory": [], "summary": {}, "exploratory": {}}
tests = []
for ds in DS:
    for h, b in (("H1", "B1"), ("H2", "A2"), ("H3", "A4")):
        r = paired(ds, "A1", b)
        if r is None or r["n"] < 10:
            print(f"INCOMPLETE {ds} {h}: n={None if r is None else r['n']}"); continue
        tests.append((ds, h, b, r))
if len(tests) == 9:
    adj = holm(np.array([t[3]["p"] for t in tests]))
    for (ds, h, b, r), pa in zip(tests, adj):
        sig = pa < 0.05
        verdict = ("A1 > " + b if sig and r["median_diff"] > 0 else "A1 < " + b if sig and r["median_diff"] < 0
                   else "no detectable difference at n=10")
        out["confirmatory"].append(dict(dataset=ds, hypothesis=h, comparator=b, p_holm=float(pa), verdict=verdict, **r))
else:
    print("confirmatory family incomplete; not tested")

for ds in DS:
    for part, p in (("noniid", 0.5), ("noniid", 0.1), ("iid", 0.5)):
        for arm in ARMS:
            X = D[(ds, part, p, arm)]
            if not X:
                continue
            m = np.array([X[s]["MCC"] for s in sorted(X)]); ba = np.array([X[s]["BalancedAcc"] for s in sorted(X)])
            f1 = np.array([X[s]["F1"] for s in sorted(X)])
            k = int((m > 0).sum()); n = len(m); lo, hi = wilson(k, n)
            deg = {t: sum(X[s]["degenerate"] == t for s in X) for t in ("always-benign", "always-attack")}
            out["summary"][f"{ds}|{part}|{p}|{arm}"] = dict(
                n=n, mcc_mean=float(m.mean()), mcc_sd=float(m.std(ddof=1)) if n > 1 else 0.0,
                mcc_median=float(np.median(m)), balacc_mean=float(np.nanmean(ba)), f1_mean=float(f1.mean()),
                discriminative=k, wilson95=[lo, hi], **deg, per_seed_mcc=[round(float(v), 4) for v in m])

for ds in DS:
    for b in ("A3", "B2", "B3"):
        r = paired(ds, "A1", b)
        if r and r["n"] == 10:
            out["exploratory"][f"{ds}|A1_vs_{b}|noniid|0.5"] = r
    for b in ("A2", "A3", "A4", "B1", "B2", "B3"):
        n10, n01, pv = mcnemar_disc(ds, "A1", b)
        out["exploratory"][f"{ds}|mcnemar_disc_A1_vs_{b}"] = dict(A1_only=n10, other_only=n01, p_exact=pv)
    r = paired(ds, "A1", "A2", p=0.1)
    if r:
        out["exploratory"][f"{ds}|A1_vs_A2|noniid|0.1"] = r
    for b in ("B1", "B2"):
        r = paired(ds, "A1", b, part="iid", seeds=range(1, 6))
        if r:
            out["exploratory"][f"{ds}|A1_vs_{b}|iid|0.5"] = r

json.dump(out, open(os.path.join(HERE, "analysis_results.json"), "w"), indent=1)
for c in out["confirmatory"]:
    print(f"{c['dataset']:9s} {c['hypothesis']} A1 vs {c['comparator']}: median dMCC={c['median_diff']:+.3f} "
          f"p={c['p']:.4f} p_holm={c['p_holm']:.4f} -> {c['verdict']}")
for k, v in out["summary"].items():
    print(f"{k:28s} MCC {v['mcc_mean']:+.3f}±{v['mcc_sd']:.3f} disc {v['discriminative']}/{v['n']} "
          f"benign-only {v['always-benign']} attack-only {v['always-attack']} BA {v['balacc_mean']:.3f}")
for k, v in out["exploratory"].items():
    print("EXPL", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
