"""Seals the Study 6 plan: writes SEAL_STUDY6.sha256 with the SHA-256 of the plan, code, configurations,
and development partitions, and the UTC time. Run once, before the first run."""
import datetime, glob, hashlib, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rel = lambda ps: sorted(os.path.relpath(p, ROOT) for p in ps)
files = ["PLAN_STUDY6.md"] \
    + rel(glob.glob(os.path.join(ROOT, "code", "*.py")) + glob.glob(os.path.join(ROOT, "code", "*.json"))) \
    + rel(glob.glob(os.path.join(ROOT, "partitions", "dev", "*.pkl")))
files = [f for f in files if not f.endswith("seal.py") and not f.endswith("verify_seal.py")]
out = os.path.join(ROOT, "SEAL_STUDY6.sha256")
if os.path.exists(out):
    raise SystemExit("SEAL_STUDY6.sha256 already exists; the plan is sealed. Add a dated addendum instead of re-sealing.")
with open(out, "w") as f:
    f.write(f"# sealed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")
    for p in files:
        f.write(f"{hashlib.sha256(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()}  {p.replace(os.sep, '/')}\n")
print(open(out).read())
