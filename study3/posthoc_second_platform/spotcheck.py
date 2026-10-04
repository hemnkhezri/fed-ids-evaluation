"""Records the cross-platform spot check cited in the paper: Study 3 cells re-run on Linux, compared with the Windows runs."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/claude/s3/code")
import run_s3
W = {run_s3.key(r): r for r in (json.loads(l) for l in open("/home/claude/s3res/results/results_s3.jsonl"))}
picks = [("wustl", 101, 201, "noniid", 0.1, "B4"), ("toniot", 103, 202, "noniid", 0.5, "B5"),
         ("edgeiiot", 105, 201, "noniid", 0.1, "FG"), ("toniot", 101, 201, "noniid", 0.1, "FG")]
done = {tuple(json.loads(l)["cell"]) for l in open("platform_spotcheck.jsonl")} if os.path.exists("platform_spotcheck.jsonl") else set()
for p in picks:
    if p in done: continue
    j = dict(stage="main", dataset=p[0], partition_seed=p[1], seed=p[2], regime=p[3], p_label=p[4], config=p[5])
    r = run_s3.work(j); w = W[p]
    rec = dict(cell=list(p), linux_MCC=r["MCC"], windows_MCC=w["MCC"], linux_platform=r["platform"], windows_platform=w["platform"])
    open("platform_spotcheck.jsonl", "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
