"""Exploratory (post hoc, not in the sealed plan): ablation of FedGTCL at p_L = 0.1, non-IID, same partitions and seeds
as Study 3. FG is re-run on this platform so that the paired differences come from one platform."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/claude/s3/code")
import run_s3
from multiprocessing import Pool

OUT = "/home/claude/s3linux/results_abl01_linux.jsonl"
jobs = [dict(stage="ablation01", dataset=ds, partition_seed=ps, seed=s, regime="noniid", p_label=0.1, config=m)
        for ps in run_s3.PARTS for ds in run_s3.DATASETS for s in run_s3.SEEDS for m in ("FG_noCon", "FG_noOpt", "FG")]
done = set()
if os.path.exists(OUT):
    done = {run_s3.key(json.loads(l)) for l in open(OUT)}
todo = [j for j in jobs if run_s3.key(j) not in done]
print(len(todo), "todo", flush=True)
if __name__ == "__main__":
    with Pool(2) as p:
        for r in p.imap_unordered(run_s3.work, todo):
            r.pop("loss_curve", None)
            open(OUT, "a").write(json.dumps(r) + "\n")
