"""Apply the PHASE3B_PLAN selection rule to the B5 development results and write cfg_phase3b.json."""
import json, glob, sys
from collections import defaultdict
import numpy as np
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*.jsonl") for l in open(f)]
rows = [r for r in rows if r["config"].startswith("B5_") and r["split"] == "inner"]
order = list(json.load(open("cfg_dev_b5.json")))
by = defaultdict(list)
for r in rows:
    by[r["config"]].append(r["MCC"])
assert all(len(by[c]) == 30 for c in order), {c: len(by[c]) for c in order}
means = {c: float(np.mean(by[c])) for c in order}
best = max(order, key=lambda c: (means[c], -order.index(c)))
cfg = json.load(open("cfg_phase2.json")); cfg["B5"] = json.load(open("cfg_dev_b5.json"))[best]
json.dump(cfg, open("cfg_phase3b.json", "w"), indent=1)
json.dump(dict(selected=best, means=means), open("b5_selection.json", "w"), indent=1)
print("selected", best, round(means[best], 3))
