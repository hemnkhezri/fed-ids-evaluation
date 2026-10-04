"""Checks every file listed in SEAL_STUDY4.sha256 against its sealed hash. Stops the run if anything differs."""
import hashlib, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seal = os.path.join(ROOT, "SEAL_STUDY4.sha256")
if not os.path.exists(seal):
    sys.exit("SEAL_STUDY4.sha256 is missing: the package is incomplete.")
bad = 0
for line in open(seal):
    if line.startswith("#") or not line.strip():
        continue
    h, p = line.split(maxsplit=1); p = p.strip()
    fp = os.path.join(ROOT, *p.split("/"))
    if not os.path.exists(fp) or hashlib.sha256(open(fp, "rb").read()).hexdigest() != h:
        print("CHANGED OR MISSING:", p); bad += 1
if bad:
    sys.exit(f"{bad} file(s) differ from the sealed package. Do not edit files in code/, partitions/ or the plan.")
print("seal check OK:", open(seal).readline().strip())
