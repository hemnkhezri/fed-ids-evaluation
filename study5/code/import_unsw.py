"""Copies the UNSW-NB15 window file and partitions built in Study 4 into this folder and checks them against the
SHA-256 values Study 4 recorded (reused/unsw_eligibility_study4.json). Nothing is rebuilt.
Usage: python code/import_unsw.py "C:/path/to/FedGTCL_study4" """
import hashlib, json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S4 = sys.argv[1]
el = json.load(open(os.path.join(ROOT, "reused", "unsw_eligibility_study4.json")))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True); os.makedirs(os.path.join(ROOT, "partitions", "unsw"), exist_ok=True)
pairs = [(os.path.join(S4, "data", "unsw_base.pkl"), os.path.join(ROOT, "data", "unsw_base.pkl"), el["base_sha256"])]
for p in el["partitions"]:
    n = f"partition_unsw_p{p['partition_seed']}.pkl"
    pairs.append((os.path.join(S4, "partitions", "unsw", n), os.path.join(ROOT, "partitions", "unsw", n), p["sha256"]))
for src, dst, h in pairs:
    if not os.path.exists(dst) or sha(dst) != h:
        if not os.path.exists(src):
            sys.exit(f"missing {src}: give the path of the FedGTCL_study4 folder")
        shutil.copyfile(src, dst)
    if sha(dst) != h:
        sys.exit(f"hash mismatch for {os.path.basename(dst)}: not the file Study 4 used")
    print("OK", os.path.basename(dst))
print("UNSW-NB15 files imported and verified.")
