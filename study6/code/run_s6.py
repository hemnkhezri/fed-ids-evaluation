"""Study 6 runner (PLAN_STUDY6.md). Phases, run in this order (each resumable):
  select : Part B grid on the inner split (lr in {3e-4, 1e-3, 3e-3} for FG, B4, B1; seed 201; 1,000 steps)
  curves : Part A, 150 rounds x 20 steps with checkpoints, five methods, seeds 201-202, validation regions
  confirm: Part B confirmation with the selected learning rates (only those different from 1e-3)
Usage: python code/run_s6.py --phase select|curves|confirm --workers 2"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, copy, json, pickle, platform, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DS = ["wustl", "toniot", "edgeiiot"]; PS = range(101, 111)
LRS = [3e-4, 1e-3, 3e-3]; TUNED = ["FG", "B4", "B1"]; CURVE = ["FG", "B1", "B2", "B4", "B5"]
CKPT = (1, 2, 5, 10, 20, 50, 100, 150)
CFG = json.load(open(os.path.join(HERE, "cfg_s6.json")))
OUT = {p: os.path.join(ROOT, "results", f"results_s6_{p}.jsonl") for p in ("select", "curves", "confirm")}


def load(ds, ps):
    return pickle.load(open(os.path.join(ROOT, "partitions", "dev", f"partition_{ds}_p{ps}.pkl"), "rb"))


def selected():
    rows = [json.loads(l) for l in open(OUT["select"])]
    best = {}
    for m in TUNED:
        sc = {}
        for lr in LRS:
            v = [r["MCC"] for r in rows if r["config"] == m and r["lr"] == lr]
            assert len(v) == 30, (m, lr, len(v))
            sc[lr] = sum(v) / len(v)
        top = max(sc.values()); cands = [lr for lr in LRS if sc[lr] == top]
        best[m] = dict(lr=1e-3 if 1e-3 in cands else cands[0], scores=sc)
    return best


def jobs(phase):
    if phase == "select":
        return [dict(phase=phase, dataset=d, partition_seed=p, seed=201, config=m, lr=lr, split="inner", R=50)
                for p in PS for d in DS for m in TUNED for lr in LRS]
    if phase == "curves":
        return [dict(phase=phase, dataset=d, partition_seed=p, seed=s, config=m, lr=CFG[m].get("lr", 1e-3), split="val", R=150)
                for p in PS for d in DS for s in (201, 202) for m in CURVE]
    best = selected()
    return [dict(phase=phase, dataset=d, partition_seed=p, seed=s, config=m, lr=best[m]["lr"], split="val", R=50)
            for p in PS for d in DS for s in (201, 202) for m in TUNED if best[m]["lr"] != 1e-3]


def key(j):
    return (j["dataset"], j["partition_seed"], j["seed"], j["config"], j["lr"], j["split"], j["R"])


def work(j):
    import torch; torch.set_num_threads(1)
    import train_s6 as T
    cfg = dict(CFG[j["config"]]); cfg["lr"] = j["lr"]
    P = load(j["dataset"], j["partition_seed"]); t = time.time()
    r = T.run(P, "noniid", j["split"], j["seed"], 0.5, cfg, R=j["R"], steps=20,
              eval_rounds=CKPT if j["phase"] == "curves" else ())
    return dict(j, cfg=cfg, steps=20, secs=round(time.time() - t, 1), platform=platform.platform(), torch=torch.__version__, **r)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import subprocess
    if subprocess.run([sys.executable, os.path.join(HERE, "verify_seal.py")]).returncode:
        sys.exit("seal check failed")
    out = OUT[a.phase]; os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(key(json.loads(l)))
            except Exception: pass
    todo = [j for j in jobs(a.phase) if key(j) not in done]
    print(f"{a.phase}: {len(todo)} runs to do", flush=True); t0 = time.time()
    with Pool(a.workers) as pool, open(out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(work, todo), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{i}/{len(todo)}] {r['dataset']} p{r['partition_seed']} s{r['seed']} {r['config']} lr={r['lr']} R={r['R']}: "
                  f"MCC {r['MCC']:.3f} ({r['secs']:.0f}s) ETA {(time.time() - t0) / i * (len(todo) - i) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
