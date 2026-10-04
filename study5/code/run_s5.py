"""Study 5 runner (PLAN_STUDY5.md): split x budget x configuration on four datasets, resumable.

  split   random : same clients as the partition file; inside each client, a random 80/20 split of its overlapping
                   T=5 sequences (the pilot protocol; training and validation sequences share windows)
          chrono : the chronological regions of Studies 3-4 (no window shared)
  budget  pilot  : R = 10 rounds x 3 local steps (30 steps)
          full   : R = 50 rounds x 20 local steps (1,000 steps)
  config  FG, B1, B2, B4 with p_L = 0.5; B1F, B2F, B4F = the same baselines with every label

Cells (chrono, full) for FG, B1, B2, B4 and B1F are reused from Studies 3-4 (reused/*.jsonl) and not re-run.
Usage (from the Study 5 folder): python code/run_s5.py --workers 5      |  python code/run_s5.py --list
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, pickle, platform, random, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATASETS = ["wustl", "toniot", "edgeiiot", "unsw"]
CONFIGS = ["FG", "B1", "B2", "B4", "B1F", "B2F", "B4F"]
BUDGET = {"pilot": (10, 3), "full": (50, 20)}
REUSED = {("chrono", "full", c) for c in ("FG", "B1", "B2", "B4", "B1F")}
OUT = os.path.join(ROOT, "results", "results_s5.jsonl")


def design():
    return [dict(split=sp, budget=b, dataset=ds, partition_seed=ps, seed=s, p_label=0.5, config=c)
            for ps in range(101, 111) for ds in DATASETS for s in (201, 202)
            for sp in ("random", "chrono") for b in ("pilot", "full") for c in CONFIGS
            if (sp, b, c) not in REUSED]


def key(j):
    return (j["split"], j["budget"], j["dataset"], j["partition_seed"], j["seed"], j["config"])


_P, _BASE = {}, {}


def load(ds, ps):
    pf = os.path.join(ROOT, "partitions", "unsw" if ds == "unsw" else "dev", f"partition_{ds}_p{ps}.pkl")
    if pf in _P:
        return _P[pf]
    _P.clear()
    Q = pickle.load(open(pf, "rb"))
    if "base" in Q:
        bf = os.path.join(ROOT, Q["base"])
        if bf not in _BASE:
            _BASE.clear(); _BASE[bf] = pickle.load(open(bf, "rb"))
        B = _BASE[bf]; by_a = {s["anchor_idx"]: s for s in B["sequences"]}
        P = dict(B); P["noniid_train"] = [[by_a[a] for a in cl] for cl in Q["train_anchor"]]
        P["noniid_val"] = [[by_a[a] for a in cl] for cl in Q["val_anchor"]]
        P["partition_seed"] = Q["partition_seed"]
    else:
        P = Q
    _P[pf] = P
    return P


def random_split(P):
    """Pilot protocol: each client's sequences (both regions) split 80/20 at random; augmentation may use any window."""
    ps = int(P["partition_seed"]); tr, va = [], []
    for k, (a, b) in enumerate(zip(P["noniid_train"], P["noniid_val"])):
        seqs = sorted(a + b, key=lambda e: e["anchor_idx"])
        random.Random(1_000_003 * ps + k).shuffle(seqs)
        cut = max(1, int(0.8 * len(seqs)))
        tr.append(seqs[:cut]); va.append(seqs[cut:])
    Q = dict(P); Q["noniid_train"] = tr; Q["noniid_val"] = va
    Q["train_region_window_ids"] = [i for i, w in enumerate(P["windows"]) if w is not None]
    trw = {w for cl in tr for e in cl for w in e["win_ids"]}
    vals = [e for cl in va for e in cl]
    shared = sum(any(w in trw for w in e["win_ids"]) for e in vals) / max(1, len(vals))
    return Q, shared


def work(j):
    import torch
    torch.set_num_threads(1)
    import train_s4 as T
    cfg = json.load(open(os.path.join(HERE, "cfg_s5.json")))[j["config"]]
    P = load(j["dataset"], j["partition_seed"]); shared = 0.0
    if j["split"] == "random":
        P, shared = random_split(P)
    R, steps = BUDGET[j["budget"]]
    t = time.time()
    r = T.run(P, "noniid", "val", j["seed"], j["p_label"], cfg, R=R, steps=steps, full_labels=cfg.get("full_labels", False))
    return dict(j, cfg=cfg, R=R, steps=steps, val_share_with_train_windows=round(shared, 4), secs=round(time.time() - t, 1),
                platform=platform.platform(), torch=torch.__version__, **r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5); ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    jobs = design()
    if a.list:
        from collections import Counter
        print(Counter((j["split"], j["budget"]) for j in jobs)); print("total", len(jobs)); return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(key(json.loads(line)))
            except Exception:
                pass
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} runs in the design, {len(todo)} to do", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(work, todo), 1):
            r.pop("loss_curve", None)
            f.write(json.dumps(r) + "\n"); f.flush()
            eta = (time.time() - t0) / i * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {r['split']}/{r['budget']} {r['dataset']} p{r['partition_seed']} s{r['seed']} "
                  f"{r['config']}: MCC {r['MCC']:.3f} ({r['secs']:.0f}s) ETA {eta / 3600:.1f} h", flush=True)
    print("All runs finished.")


if __name__ == "__main__":
    main()
