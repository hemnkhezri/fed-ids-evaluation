"""Rebuild the 10 X-IIoTID partitions from xiiotid_windows.pkl and verify them against assignment_hashes.json."""
import subprocess, sys, pickle, json, hashlib, os
py = sys.executable
os.makedirs("data_x", exist_ok=True)
subprocess.check_call([py, "build_xiiotid_partition.py", "xiiotid_windows.pkl", "data_x/partition_xiiotid_seed42.pkl"])
subprocess.check_call([py, "make_phase2_partitions.py", "data_x/partition_xiiotid_seed42.pkl", "data_x", "xiiotid"])
exp = json.load(open("assignment_hashes.json"))
for p in range(101, 111):
    P = pickle.load(open(f"data_x/partition_xiiotid_p{p}.pkl", "rb"))
    a = json.dumps([[s["anchor_idx"] for s in c] for c in P["noniid_train"]] + [[s["anchor_idx"] for s in c] for c in P["noniid_val"]])
    h = hashlib.sha256(a.encode()).hexdigest()[:16]
    assert h == exp[str(p)], f"partition {p} mismatch: {h} != {exp[str(p)]}"
print("all 10 X-IIoTID partitions verified")
