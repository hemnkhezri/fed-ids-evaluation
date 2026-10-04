"""Study 3 runner: all jobs of the design in parallel worker processes (one CPU thread each), resumable.

Usage (from the study3 folder):
    python code/run_s3.py --workers 5                 # full design (PLAN_STUDY3.md)
    python code/run_s3.py --workers 5 --stage main    # only the confirmatory part
    python code/run_s3.py --list                      # print the job count per stage and exit
Results are appended to results/results_s3.jsonl, one line per finished run. Re-running skips finished runs.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, pickle, sys, time, platform
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATASETS = ["wustl", "toniot", "edgeiiot"]
PARTS = list(range(101, 111))
SEEDS = [201, 202]
R, STEPS = int(os.environ.get("S3_R", 50)), 20   # S3_R only for smoke tests


def design():
    jobs = []
    for ps in PARTS:                      # partitions outermost, so partial results stay balanced
        for ds in DATASETS:
            for seed in SEEDS:
                for regime in ("noniid", "iid"):
                    for pl in (0.1, 0.5):
                        for m in ("FG", "B1", "B2", "B4", "B5"):
                            jobs.append(dict(stage="main", dataset=ds, partition_seed=ps, seed=seed, regime=regime, p_label=pl, config=m))
                    jobs.append(dict(stage="main", dataset=ds, partition_seed=ps, seed=seed, regime=regime, p_label=1.0, config="REF"))
                for m in ("FG_noCon", "FG_noOpt"):
                    jobs.append(dict(stage="ablation", dataset=ds, partition_seed=ps, seed=seed, regime="noniid", p_label=0.5, config=m))
    return jobs


def key(j):
    return (j["dataset"], j["partition_seed"], j["seed"], j["regime"], j["p_label"], j["config"])


_P = {}


def work(j):
    import torch
    torch.set_num_threads(1)
    import train_s3 as T
    cfgs = json.load(open(os.path.join(HERE, "cfg_s3.json")))
    pf = os.path.join(ROOT, "partitions", f"partition_{j['dataset']}_p{j['partition_seed']}.pkl")
    if pf not in _P:
        _P.clear(); _P[pf] = pickle.load(open(pf, "rb"))
    cfg = cfgs[j["config"]]
    t = time.time()
    r = T.run(_P[pf], j["regime"], "val", j["seed"], j["p_label"], cfg, R=R, steps=STEPS, full_labels=cfg.get("full_labels", False))
    return dict(j, cfg=cfg, R=R, steps=STEPS, secs=round(time.time() - t, 1), platform=platform.platform(),
                torch=torch.__version__, **r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--stage", choices=["main", "ablation", "all"], default="all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results_s3.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N pending jobs (testing)")
    a = ap.parse_args()
    jobs = [j for j in design() if a.stage == "all" or j["stage"] == a.stage]
    if a.list:
        from collections import Counter
        print(Counter((j["stage"], j["config"]) for j in jobs)); print("total", len(jobs)); return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(key(json.loads(line)))
            except Exception:
                pass
    todo = [j for j in jobs if key(j) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(jobs)} jobs in design, {len(done)} already done, {len(todo)} to run with {a.workers} workers", flush=True)
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(work, todo):
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            n += 1
            el = time.time() - t0; eta = el / n * (len(todo) - n)
            print(f"[{n}/{len(todo)}] {row['dataset']} p{row['partition_seed']} s{row['seed']} {row['regime']} pL={row['p_label']} "
                  f"{row['config']}: MCC={row['MCC']:.3f} ({row['secs']}s)  elapsed {el/3600:.2f} h, remaining ~{eta/3600:.2f} h", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
