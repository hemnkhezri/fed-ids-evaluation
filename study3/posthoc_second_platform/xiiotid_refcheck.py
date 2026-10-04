"""Records the X-IIoTID learnability check cited in the paper: full-label FedAvg-GCN-GRU (REF), Study 3 budget, seed 201."""
import os, sys, json, pickle
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/claude/s3/code")
import torch; torch.set_num_threads(1)
import train_s3 as T
cfg = json.load(open("/home/claude/s3/code/cfg_s3.json"))["REF"]
for ps in (101, 102):
    P = pickle.load(open(f"/mnt/attach/outputs/rerun_v3/heldout2/partition_xiiotid_p{ps}.pkl", "rb"))
    for regime in ("noniid", "iid"):
        r = T.run(P, regime, "val", 201, 1.0, cfg, R=50, steps=20, full_labels=True)
        rec = dict(dataset="xiiotid", partition_seed=ps, regime=regime, seed=201, config="REF", R=50, steps=20,
                   MCC=r["MCC"], BalancedAcc=r["BalancedAcc"], TP=r["TP"], FP=r["FP"], TN=r["TN"], FN=r["FN"], degenerate=r["degenerate"])
        open("xiiotid_ref_check.jsonl", "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
