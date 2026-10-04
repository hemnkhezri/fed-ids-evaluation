"""Study 4 runner (PLAN_STUDY4.md). Runs the stages in order, in parallel worker processes (one CPU thread each).
Resumable: finished runs are kept in results/results_s4.jsonl and skipped on a re-run.

  gate      full-label reference on UNSW-NB15 partitions 101-103 (seed 201); the external test runs only if the
            mean MCC is >= 0.30
  select    candidates C1-C4 on the development datasets (partitions 101-110, seeds 201-202, non-IID,
            p_L 0.1 and 0.5); the candidate with the highest mean MCC is selected automatically (results/selected.json)
  internal  selected candidate, FedProx (B4) and pseudo-labelling (B5) on NEW partitions 111-120 with NEW seeds 203-204
  external  UNSW-NB15 partitions 101-110, seeds 201-202: selected candidate, B4, B5 (confirmatory) and FG, B1, B2,
            REF (reported)

Usage (from the Study 4 folder):  python code/run_s4.py --workers 5
          python code/run_s4.py --list      (job counts, no run)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, pickle, platform, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DEV_DS = ["wustl", "toniot", "edgeiiot"]
CANDIDATES = ["C1", "C2", "C3", "C4"]
R, STEPS = int(os.environ.get("S4_R", 50)), 20            # S4_R only for smoke tests
GATE_MIN = 0.30
OUT = os.path.join(ROOT, "results", "results_s4.jsonl")


def jobs_gate():
    return [dict(stage="gate", dataset="unsw", partition_seed=ps, seed=201, regime="noniid", p_label=1.0, config="REF")
            for ps in (101, 102, 103)]


def jobs_select():
    return [dict(stage="select", dataset=ds, partition_seed=ps, seed=s, regime="noniid", p_label=pl, config=c)
            for ps in range(101, 111) for ds in DEV_DS for s in (201, 202) for pl in (0.1, 0.5) for c in CANDIDATES]


def jobs_internal(sel):
    return [dict(stage="internal", dataset=ds, partition_seed=ps, seed=s, regime="noniid", p_label=pl, config=c)
            for ps in range(111, 121) for ds in DEV_DS for s in (203, 204) for pl in (0.1, 0.5) for c in (sel, "B4", "B5")]


def jobs_external(sel):
    j = [dict(stage="external", dataset="unsw", partition_seed=ps, seed=s, regime="noniid", p_label=pl, config=c)
         for ps in range(101, 111) for s in (201, 202) for pl in (0.1, 0.5) for c in (sel, "B4", "B5", "FG", "B1", "B2")]
    j += [dict(stage="external", dataset="unsw", partition_seed=ps, seed=s, regime="noniid", p_label=1.0, config="REF")
          for ps in range(101, 111) for s in (201, 202)]
    return j


def key(j):
    return (j["stage"], j["dataset"], j["partition_seed"], j["seed"], j["regime"], j["p_label"], j["config"])


_P, _BASE = {}, {}


def load_partition(ds, ps):
    if ds == "unsw":
        pf = os.path.join(ROOT, "partitions", "unsw", f"partition_unsw_p{ps}.pkl")
    else:
        pf = os.path.join(ROOT, "partitions", "dev", f"partition_{ds}_p{ps}.pkl")
    if pf in _P:
        return _P[pf]
    _P.clear()
    Q = pickle.load(open(pf, "rb"))
    if "base" in Q:                                    # UNSW: shared window file + per-partition assignment
        bf = os.path.join(ROOT, Q["base"])
        if bf not in _BASE:
            _BASE.clear(); _BASE[bf] = pickle.load(open(bf, "rb"))
        B = _BASE[bf]; by_a = {s["anchor_idx"]: s for s in B["sequences"]}
        P = dict(B); P["noniid_train"] = [[by_a[a] for a in cl] for cl in Q["train_anchor"]]
        P["noniid_val"] = [[by_a[a] for a in cl] for cl in Q["val_anchor"]]
        P["partition_seed"] = Q["partition_seed"]; P["regime"] = Q["regime"]
    else:
        P = Q
    _P[pf] = P
    return P


def work(j):
    import torch
    torch.set_num_threads(1)
    import train_s4 as T
    cfg = json.load(open(os.path.join(HERE, "cfg_s4.json")))[j["config"]]
    P = load_partition(j["dataset"], j["partition_seed"])
    t = time.time()
    r = T.run(P, j["regime"], "val", j["seed"], j["p_label"], cfg, R=R, steps=STEPS, full_labels=cfg.get("full_labels", False))
    return dict(j, cfg=cfg, R=R, steps=STEPS, secs=round(time.time() - t, 1), platform=platform.platform(),
                torch=torch.__version__, **r)


def done_rows():
    rows = []
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def run_stage(name, jobs, workers):
    done = {key(r) for r in done_rows()}
    todo = [j for j in jobs if key(j) not in done]
    print(f"\n== stage {name}: {len(jobs)} runs, {len(todo)} to do", flush=True)
    if not todo:
        return
    t0 = time.time()
    with Pool(workers) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(work, todo), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            el = time.time() - t0; eta = el / i * (len(todo) - i)
            print(f"[{name} {i}/{len(todo)}] {r['dataset']} p{r['partition_seed']} s{r['seed']} pL={r['p_label']} "
                  f"{r['config']}: MCC {r['MCC']:.3f}  ({r['secs']:.0f}s)  ETA {eta / 3600:.1f} h", flush=True)


def select():
    import numpy as np
    from collections import defaultdict
    cell = defaultdict(list)
    for r in done_rows():
        if r["stage"] == "select":
            cell[(r["config"], r["dataset"], r["partition_seed"], r["p_label"])].append(r["MCC"])
    score, deg = {}, {}
    for c in CANDIDATES:
        units = [np.mean(v) for (cc, *_), v in cell.items() if cc == c]
        assert len(units) == 60, f"selection incomplete for {c}: {len(units)} of 60 units"
        score[c] = float(np.mean(units))
    for r in done_rows():
        if r["stage"] == "select":
            deg[r["config"]] = deg.get(r["config"], 0) + (r["degenerate"] != "no")
    best = max(CANDIDATES, key=lambda c: (round(score[c], 3), -CANDIDATES.index(c)))
    sel = dict(selected=best, scores=score, degenerate=deg, rule="highest mean MCC over 60 units; ties at 3 decimals "
               "go to the earlier candidate in the order C1, C2, C3, C4")
    json.dump(sel, open(os.path.join(ROOT, "results", "selected.json"), "w"), indent=1)
    print("\nSELECTION:", json.dumps(sel, indent=1), flush=True)
    return best


def gate():
    import numpy as np
    v = [r["MCC"] for r in done_rows() if r["stage"] == "gate"]
    ok = len(v) == 3 and float(np.mean(v)) >= GATE_MIN
    g = dict(passed=ok, mean_MCC=float(np.mean(v)) if v else None, runs=v, threshold=GATE_MIN)
    json.dump(g, open(os.path.join(ROOT, "results", "gate.json"), "w"), indent=1)
    print("\nGATE:", json.dumps(g), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, j in (("gate", jobs_gate()), ("select", jobs_select()), ("internal", jobs_internal("C1")),
                     ("external", jobs_external("C1"))):
            print(n, len(j))
        return
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    elig = json.load(open(os.path.join(ROOT, "results", "unsw_eligibility.json")))
    ext_ok = elig["eligible"]
    if ext_ok:
        run_stage("gate", jobs_gate(), a.workers)
        ext_ok = gate()
    else:
        print("UNSW-NB15 is not eligible (unsw_eligibility.json): the external test is skipped.")
    run_stage("select", jobs_select(), a.workers)
    sel = select()
    run_stage("internal", jobs_internal(sel), a.workers)
    if ext_ok:
        run_stage("external", jobs_external(sel), a.workers)
    else:
        print("External test skipped (eligibility or learnability gate not met).")
    print("\nAll stages finished.")


if __name__ == "__main__":
    main()
