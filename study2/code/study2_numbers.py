"""Every Study-2 number quoted in the manuscript, computed from the raw run records.
Inputs : dev_results_local.jsonl, results_b5dev/, phase2 + phase3b results (local), cloud cross-checks.
Output : study2_numbers.json (read by the manuscript builder; nothing is typed by hand)."""
import json, glob, math, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

V3 = "/mnt/user-data/outputs/rerun_v3"
LOCAL = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/work/p3bres/combined"
OUT = f"{V3}/study2_summary/study2_numbers.json"
DS = ["wustl", "toniot", "edgeiiot", "cicapt", "xiiotid"]
M = ["FG", "B1", "B2", "B3", "B4", "B5"]


def load(files):
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0, 0)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def wil(x, y):
    x = np.asarray(x); y = np.asarray(y); d = x - y
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


# ------------------------------------------------------------------ development (phase 1 + B5 step 1)
dev = load([f"{V3}/dev_results_local.jsonl"] + sorted(glob.glob(f"{V3}/results_b5dev/*.jsonl")))
cfg_runs = defaultdict(list)
for r in dev:
    cfg_runs[r["config"]].append(r)
fam = lambda c: c.split("_")[0]
sel = json.load(open(f"{V3}/selected_configs.json")); sel["B5"] = json.load(open(f"{V3}/b5_selection.json"))["selected"]
D = {}
for f_ in M:
    cfgs = {c: v for c, v in cfg_runs.items() if fam(c) == f_}
    means = {c: float(np.mean([r["MCC"] for r in v])) for c, v in cfgs.items()}
    s = sel[f_]; rs = cfgs[s]
    per = {}
    for ds in ("wustl", "toniot", "edgeiiot"):
        for p in (0.1, 0.5):
            v = [r["MCC"] for r in rs if r["dataset"] == ds and r["p_label"] == p]
            per[f"{ds}|{p}"] = dict(n=len(v), mean=float(np.mean(v)), disc=int(sum(m > 0 for m in v)))
    D[f_] = dict(n_configs=len(cfgs), n_runs_per_config=sorted({len(v) for v in cfgs.values()}), selected=s,
                 selected_mean=means[s], selected_runs=len(rs), range_of_config_means=[min(means.values()), max(means.values())],
                 per_cell=per)

# ------------------------------------------------------------------ confirmatory runs (local = pre-registered)
L = [r for r in load(glob.glob(f"{LOCAL}/*.jsonl")) if r.get("tag") in ("phase2", "phase3b") and r["p_label"] == 0.1]
cell = defaultdict(dict)                     # (ds, method) -> {(pseed, seed): record}
for r in L:
    cell[(r["dataset"], r["config"])][(r["partition_seed"], r["seed"])] = r


def pmeans(c):
    by = defaultdict(list)
    for (ps, s), r in c.items():
        by[ps].append(r["MCC"])
    return {ps: float(np.mean(v)) for ps, v in by.items() if len(v) == 2}


summary = {}
for ds in DS:
    for m in M:
        c = cell.get((ds, m))
        if not c:
            continue
        v = [r["MCC"] for r in c.values()]; n = len(v); k = sum(x > 0 for x in v)
        pm = pmeans(c)
        summary[f"{ds}|{m}"] = dict(
            n_runs=n, mean=float(np.mean(v)), sd_run=float(np.std(v, ddof=1)),
            mean_partition=float(np.mean(list(pm.values()))), sd_partition=float(np.std(list(pm.values()), ddof=1)),
            median_partition=float(np.median(list(pm.values()))), disc=k, disc_ci=wilson(k, n),
            neg=int(sum(x < 0 for x in v)),
            benign_only=int(sum(r["degenerate"] == "always-benign" for r in c.values())),
            attack_only=int(sum(r["degenerate"] == "always-attack" for r in c.values())),
            balacc=float(np.mean([r["BalancedAcc"] for r in c.values()])),
            f1=float(np.mean([r["F1"] for r in c.values()])),
            regimes=sorted({r.get("regime") for r in c.values()}))


def test(dss, a, b, src=None):
    src = src or cell
    A, B = {}, {}
    for ds in dss:
        for ps, v in pmeans(src.get((ds, a), {})).items():
            A[(ds, ps)] = v
        for ps, v in pmeans(src.get((ds, b), {})).items():
            B[(ds, ps)] = v
    ks = sorted(set(A) & set(B))
    x = [A[k] for k in ks]; y = [B[k] for k in ks]; d = np.subtract(x, y)
    return dict(n=len(ks), mean_a=float(np.mean(x)), mean_b=float(np.mean(y)), median_diff=float(np.median(d)),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()), p=wil(x, y))


fam1 = {"H_A": test(["wustl"], "FG", "B4"), "H_B": test(["toniot", "edgeiiot"], "FG", "B4"),
        "H_C": test(["cicapt"], "FG", "B4")}
fam2 = {"H_E1": test(["toniot", "edgeiiot"], "FG", "B5"), "H_E2": test(["xiiotid"], "FG", "B4"),
        "H_E3": test(["xiiotid"], "FG", "B5")}
for fam_ in (fam1, fam2):
    for (h, r), q in zip(fam_.items(), holm([r["p"] for r in fam_.values()])):
        r["p_holm"] = q
        r["decision"] = ("FG > comparator" if q < 0.05 and r["median_diff"] > 0 else
                         "FG < comparator" if q < 0.05 and r["median_diff"] < 0 else "no detectable difference")

expl = {}
for ds in DS:
    for b in ("B1", "B2", "B3", "B4", "B5"):
        if (ds, b) in cell and (ds, "FG") in cell:
            expl[f"{ds}|FG_vs_{b}"] = test([ds], "FG", b)
for b in ("B1", "B2", "B3"):
    expl[f"toniot+edgeiiot|FG_vs_{b}"] = test(["toniot", "edgeiiot"], "FG", b)

# regime moderator (pre-specified in PHASE2_PLAN as exploratory)
regime = {}
for ds in ("wustl", "toniot", "edgeiiot", "cicapt", "xiiotid"):
    reg = {r["partition_seed"]: r.get("regime") for r in L if r["dataset"] == ds}
    fg = pmeans(cell[(ds, "FG")])
    for rg in ("extreme", "moderate"):
        ps = [p for p, g in reg.items() if g == rg]
        if ps:
            regime[f"{ds}|{rg}"] = dict(n=len(ps), FG=float(np.mean([fg[p] for p in ps])),
                                        best_baseline=float(max(np.mean([pmeans(cell[(ds, b)])[p] for p in ps])
                                                                for b in ("B1", "B2", "B3", "B4", "B5") if (ds, b) in cell)))

# ------------------------------------------------------------------ cross-platform (post hoc, not pre-registered)
PR = f"{V3}/phase2_results"
C = load(glob.glob(f"{PR}/cloud_crosscheck_*.jsonl"))
ccell = defaultdict(dict)
for r in C:
    ccell[(r["dataset"], r["config"])][(r["partition_seed"], r["seed"])] = r
cloud = {}
cl_src = dict(ccell)
# B4 on WUSTL was repeated on the Linux instance after the independent review (posthoc/, E=3 as pre-registered)
PH = [f"{V3}/posthoc/wustl_B4_E3_linux.jsonl", f"{V3}/wustl_B4_E3_linux.jsonl"]
wb4 = load([p for p in PH if os.path.exists(p)][:1])
for r in wb4:
    ccell[("wustl", "B4")][(r["partition_seed"], r["seed"])] = r
cl_src = dict(ccell)
cloud["H_A"] = test(["wustl"], "FG", "B4", cl_src); cloud["H_A"]["note"] = f"linux FG vs linux B4 ({len(wb4)} runs)"
cloud["H_A_mixed"] = test(["wustl"], "FG", "B4", {**cl_src, ("wustl", "B4"): cell[("wustl", "B4")]})
cloud["H_B"] = test(["toniot", "edgeiiot"], "FG", "B4", cl_src)
cloud["H_E1"] = test(["toniot", "edgeiiot"], "FG", "B5", cl_src)
for b in ("B1", "B2", "B3"):
    cloud[f"toniot+edgeiiot|FG_vs_{b}"] = test(["toniot", "edgeiiot"], "FG", b, cl_src)
for ds in ("toniot", "edgeiiot"):
    cloud[f"{ds}|FG_vs_B5"] = test([ds], "FG", "B5", cl_src)
cloud_summary = {}
for (ds, m), c in ccell.items():
    v = [r["MCC"] for r in c.values()]
    cloud_summary[f"{ds}|{m}"] = dict(n=len(v), mean=float(np.mean(v)), disc=int(sum(x > 0 for x in v)),
                                      neg=int(sum(x < 0 for x in v)))
agree = {}
for (ds, m), c in ccell.items():
    loc = cell.get((ds, m), {}); ks = [k for k in c if k in loc]
    same = sum(all(c[k][x] == loc[k][x] for x in ("TP", "FP", "TN", "FN")) for k in ks)
    dm = [abs(c[k]["MCC"] - loc[k]["MCC"]) for k in ks]
    agree[f"{ds}|{m}"] = dict(n=len(ks), identical=int(same), mean_abs_dMCC=float(np.mean(dm)) if dm else None)
fg_pairs = [k for k in agree if k.endswith("|FG")]
agree["FG_all"] = dict(n=sum(agree[k]["n"] for k in fg_pairs), identical=sum(agree[k]["identical"] for k in fg_pairs),
                       mean_abs_dMCC=float(np.average([agree[k]["mean_abs_dMCC"] for k in fg_pairs],
                                                      weights=[agree[k]["n"] for k in fg_pairs])))

# dev cross-platform agreement (cloud dev_results.jsonl vs local)
dl = {(r["dataset"], r["p_label"], r["seed"], r["config"]): r for r in load([f"{V3}/dev_results_local.jsonl"])}
dc = load([f"{V3}/dev_results.jsonl"]); ks = [(r["dataset"], r["p_label"], r["seed"], r["config"]) for r in dc]
same = sum(all(r[x] == dl[k][x] for x in ("TP", "FP", "TN", "FN")) for r, k in zip(dc, ks) if k in dl)
agree["dev"] = dict(n=sum(k in dl for k in ks), identical=int(same))

# ---------------- post hoc matched-steps check (B1-B4 with E=30; Linux)
mrows = []
for p in glob.glob(f"{V3}/posthoc/matched_E30*.jsonl") + glob.glob(f"{V3}/matched_E30*.jsonl"):
    mrows += load([p])
mcell = defaultdict(dict)
for r in mrows:
    mcell[(r["dataset"], r["config"])][(r["partition_seed"], r["seed"])] = r
msrc = {k: v for k, v in mcell.items()}
for ds in ("wustl", "toniot", "edgeiiot"):
    msrc[(ds, "FG")] = cell[(ds, "FG")]           # FedGTCL: pre-registered runs (Windows), unchanged
matched = {"n_runs": len(mrows), "summary": {}, "tests": {}}
for (ds, m), c in mcell.items():
    v = [r["MCC"] for r in c.values()]
    matched["summary"][f"{ds}|{m}"] = dict(n=len(v), mean=float(np.mean(v)), disc=int(sum(x > 0 for x in v)),
        benign_only=int(sum(r["degenerate"] == "always-benign" for r in c.values())),
        attack_only=int(sum(r["degenerate"] == "always-attack" for r in c.values())),
        mean_E3=summary[f"{ds}|{m}"]["mean"])
for b in ("B1", "B2", "B3", "B4"):
    for dss, key in ((["toniot", "edgeiiot"], "toniot+edgeiiot"), (["toniot"], "toniot"), (["edgeiiot"], "edgeiiot"),
                     (["wustl"], "wustl")):
        if all(len(mcell.get((d, b), {})) == 20 for d in dss):
            matched["tests"][f"{key}|FG_vs_{b}"] = test(dss, "FG", b, msrc)

# ---------------- post hoc STEP-MATCHED check (E=12 WUSTL, E=18 TON/Edge; Linux). Primary comparison uses the
# Linux FedGTCL runs (same platform); the Windows (pre-registered) FedGTCL runs are reported alongside.
srows = []
for p in glob.glob(f"{V3}/posthoc/stepmatched_*.jsonl"):
    srows += load([p])
scell = defaultdict(dict)
for r in srows:
    scell[(r["dataset"], r["config"])][(r["partition_seed"], r["seed"])] = r
stepm = {"n_runs": len(srows), "E": {ds: sorted({r["E"] for r in srows if r["dataset"] == ds}) for ds in ("wustl", "toniot", "edgeiiot")},
         "summary": {}, "tests_linuxFG": {}, "tests_windowsFG": {}, "tests_E30_linuxFG": {}}
for (ds, m), c in scell.items():
    v = [r["MCC"] for r in c.values()]
    stepm["summary"][f"{ds}|{m}"] = dict(n=len(v), mean=float(np.mean(v)), disc=int(sum(x > 0 for x in v)),
        benign_only=int(sum(r["degenerate"] == "always-benign" for r in c.values())),
        attack_only=int(sum(r["degenerate"] == "always-attack" for r in c.values())),
        mean_E3=summary[f"{ds}|{m}"]["mean"], mean_E30=matched["summary"].get(f"{ds}|{m}", {}).get("mean"))
for ds in ("wustl", "toniot", "edgeiiot"):
    v = [r["MCC"] for r in ccell[(ds, "FG")].values()]
    stepm["summary"][f"{ds}|FG_linux"] = dict(n=len(v), mean=float(np.mean(v)))
srcL = dict(scell); srcW = dict(scell); srcL30 = dict(mcell)
for ds in ("wustl", "toniot", "edgeiiot"):
    srcL[(ds, "FG")] = ccell[(ds, "FG")]; srcW[(ds, "FG")] = cell[(ds, "FG")]; srcL30[(ds, "FG")] = ccell[(ds, "FG")]
for b in ("B1", "B2", "B3", "B4"):
    for dss, key in ((["toniot", "edgeiiot"], "toniot+edgeiiot"), (["toniot"], "toniot"), (["edgeiiot"], "edgeiiot"),
                     (["wustl"], "wustl")):
        if all(len(scell.get((d, b), {})) == 20 for d in dss):
            stepm["tests_linuxFG"][f"{key}|FG_vs_{b}"] = test(dss, "FG", b, srcL)
            stepm["tests_windowsFG"][f"{key}|FG_vs_{b}"] = test(dss, "FG", b, srcW)
        if all(len(mcell.get((d, b), {})) == 20 for d in dss):
            stepm["tests_E30_linuxFG"][f"{key}|FG_vs_{b}"] = test(dss, "FG", b, srcL30)

fg_neg_phase2 = sum(r["MCC"] < 0 for r in L if r["config"] == "FG" and r["dataset"] in ("wustl", "toniot", "edgeiiot", "cicapt"))
out = dict(development=D, summary=summary, family1=fam1, family2=fam2, exploratory=expl, regime=regime,
           cloud_tests=cloud, matched=matched, stepmatched=stepm, cloud_summary=cloud_summary, agreement=agree,
           fg_negative_runs_phase2=dict(neg=int(fg_neg_phase2), of=80),
           runs=dict(local_confirmatory=len(L), dev=len(dev), cloud=len(C)))
json.dump(out, open(OUT, "w"), indent=1)
for h, r in {**fam1, **fam2}.items():
    print(h, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("cloud", {h: round(r["p"], 4) for h, r in cloud.items()})
print("agreement", agree)
print("runs", out["runs"], "FG neg", out["fg_negative_runs_phase2"])
for f_ in M:
    print(f_, D[f_]["n_configs"], D[f_]["selected"], round(D[f_]["selected_mean"], 3))
