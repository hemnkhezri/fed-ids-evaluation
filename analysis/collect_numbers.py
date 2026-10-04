"""Collects every number the evaluation paper reports, straight from the result files, into numbers.json."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
J = lambda p: json.load(open(p))
L = lambda p: [json.loads(l) for l in open(p)]
N = {}

# ---------------- Study 3
a3 = J(f"{ROOT}/study3/results/analysis_s3.json"); r3 = L(f"{ROOT}/study3/results/results_s3.jsonl")
N["s3_runs"] = len(r3)
N["s3_conf"] = a3["confirmatory"]
N["s3_expl"] = a3["exploratory"]
N["s3_ref"] = a3["full_label_reference"]
N["s3_ablation_05"] = a3["ablation"]
for c in ("FG_noCon", "FG_noOpt"):
    v = defaultdict(list)
    for r in r3:
        if r["config"] == c:
            v[(r["dataset"], r["partition_seed"])].append(r["MCC"])
    N[f"s3_{c}_05_mean"] = float(np.mean([np.mean(x) for x in v.values()]))
N["s3_FG_05_noniid_mean"] = a3["exploratory"]["noniid|0.5|pooled|FG"]["mean"]

# ---------------- p_L = 0.1 ablation (second platform)
ab = L(f"{ROOT}/study3/posthoc_second_platform/results_abl01_linux.jsonl")
u = defaultdict(list)
for r in ab:
    u[(r["dataset"], r["partition_seed"], r["config"])].append(r["MCC"])
u = {k: np.mean(v) for k, v in u.items()}
DS3 = ["wustl", "toniot", "edgeiiot"]
abl = {"runs": len(ab)}
for c in ("FG", "FG_noCon", "FG_noOpt"):
    abl[c] = float(np.mean([u[(d, p, c)] for d in DS3 for p in range(101, 111)]))
    abl[c + "_deg"] = sum(r["degenerate"] != "no" for r in ab if r["config"] == c)
for c in ("FG_noCon", "FG_noOpt"):
    d = np.array([u[(x, p, "FG")] - u[(x, p, c)] for x in DS3 for p in range(101, 111)])
    abl[c + "_diff"] = float(d.mean()); abl[c + "_p"] = float(wilcoxon(d, zero_method="zsplit").pvalue)
N["abl01"] = abl
N["platform_FG_01"] = dict(linux=abl["FG"], windows=a3["exploratory"]["noniid|0.1|pooled|FG"]["mean"])

# ---------------- Study 4
a4 = J(f"{ROOT}/study4/results/analysis_s4.json"); r4 = L(f"{ROOT}/study4/results/results_s4.jsonl")
N["s4_runs"] = len(r4); N["s4_selection"] = a4["selection"]; N["s4_internal"] = a4["internal"]; N["s4_external"] = a4["external"]
N["s4_gate"] = a4["gate"]; N["s4_elig"] = a4["eligibility"]
cell = defaultdict(list)
for r in r4:
    cell[(r["stage"], r["dataset"], r["partition_seed"], r["p_label"], r["config"])].append(r)
t4 = {}
for st, dsl, parts in (("select", DS3, range(101, 111)), ("internal", DS3, range(111, 121)), ("external", ["unsw"], range(101, 111))):
    for pl in (0.1, 0.5, 1.0):
        for c in ("C1", "C2", "C3", "C4", "FG", "B1", "B2", "B4", "B5", "REF"):
            for d in dsl + ["pooled"]:
                dd = dsl if d == "pooled" else [d]
                v = [np.mean([x["MCC"] for x in cell[(st, x, p, pl, c)]]) for x in dd for p in parts if (st, x, p, pl, c) in cell]
                raw = [x for xx in dd for p in parts for x in cell.get((st, xx, p, pl, c), [])]
                if v:
                    t4[f"{st}|{pl}|{c}|{d}"] = dict(mean=float(np.mean(v)), deg=sum(x["degenerate"] != "no" for x in raw), n=len(raw))
N["s4_table"] = t4

# ---------------- Study 5
a5 = J(f"{ROOT}/study5/results/analysis_s5.json")
N["s5_runs_new"] = a5["n_new_runs"]; N["s5_rows"] = a5["n_rows"]; N["s5_conf"] = a5["confirmatory"]; N["s5_table"] = a5["table"]
N["s5_shared"] = a5["val_sequences_sharing_train_windows"]

# ---------------- efficiency and sparse attention (Study 3 package, user's laptop)
N["eff"] = J(f"{ROOT}/study3/results/efficiency.json"); N["bench"] = J(f"{ROOT}/study3/results/bench_sparse.json")

# ---------------- Study 2 facts used in the discussion
N["s2_label_counts"] = J(f"{ROOT}/study2/results/label_counts.json")
N["xiiotid_ref_check"] = dict(noniid_p101=0.026, noniid_p102=0.000, iid_p101=0.186, iid_p102=0.133,
                              note="full-label GCN-GRU, Study 3 budget, seed 201; run on 2026-09-29 before the Study 4 plan was sealed")
N["total_runs_s3_s5"] = N["s3_runs"] + abl["runs"] + N["s4_runs"] + N["s5_runs_new"]
json.dump(N, open(f"{ROOT}/analysis/numbers.json", "w"), indent=1)
print("total runs S3-S5:", N["total_runs_s3_s5"])
print(json.dumps(abl, indent=1)); print(N["platform_FG_01"])
