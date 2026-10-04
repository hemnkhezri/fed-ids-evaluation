"""Seals the Study 3 plan: writes SEAL_STUDY3.sha256 with the SHA-256 of the plan, code, configurations and
partition files and the UTC time. Run once, before the first run. Usage: python code/seal.py"""
import datetime, glob, hashlib, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = ["PLAN_STUDY3.md"] + sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "code", "*.py")) + glob.glob(os.path.join(ROOT, "code", "*.json"))) \
        + sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "partitions", "*.pkl")))
out = os.path.join(ROOT, "SEAL_STUDY3.sha256")
if os.path.exists(out):
    raise SystemExit("SEAL_STUDY3.sha256 already exists; the plan is sealed. Add a dated addendum instead of re-sealing.")
with open(out, "w") as f:
    f.write(f"# sealed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")
    for p in files:
        f.write(f"{hashlib.sha256(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()}  {p.replace(os.sep, '/')}\n")
print(open(out).read())
