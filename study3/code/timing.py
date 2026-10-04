"""Measures this computer's speed on a few short runs (about 2-4 minutes) and estimates the time of the full
Study 3 design for a given number of parallel workers. Usage: python code/timing.py --workers 5"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import torch
torch.set_num_threads(1)
import train_s3 as T
import run_s3 as RS

ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=5); a = ap.parse_args()
cfgs = json.load(open(os.path.join(HERE, "cfg_s3.json")))
per_round = {}
for ds in RS.DATASETS:
    P = pickle.load(open(os.path.join(ROOT, "partitions", f"partition_{ds}_p101.pkl"), "rb"))
    for nm in ("FG", "B1", "B2", "B5"):
        t = time.time()
        T.run(P, "noniid", "val", 201, 0.5, cfgs[nm], R=2, steps=RS.STEPS)
        per_round[(ds, nm)] = (time.time() - t) / 2
        print(f"{ds:9s} {nm:3s}: {per_round[(ds, nm)]:.2f} s per round", flush=True)


def est(j):
    nm = j["config"]; fam = "FG" if nm.startswith("FG") else ("B2" if nm == "B2" else ("B5" if nm == "B5" else "B1"))
    return per_round[(j["dataset"], fam)] * RS.R


jobs = RS.design()
tot = sum(est(j) for j in jobs)
main = sum(est(j) for j in jobs if j["stage"] == "main")
print(f"\nCPU time: full design {tot/3600:.1f} h ({len(jobs)} runs), confirmatory part only {main/3600:.1f} h")
print(f"With {a.workers} parallel workers: about {tot/3600/a.workers:.1f} h (full), {main/3600/a.workers:.1f} h (main stage)")
print("Expect 10-20% more if the laptop throttles under sustained load.")
