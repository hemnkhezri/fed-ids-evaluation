"""Checks every partition file before Study 3: training sequences only use training-region windows, validation
sequences only validation-region windows, no sequence crosses a segment, no window is shared between training and
validation. Usage: python code/check_partitions.py"""
import glob, os, pickle, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = 0
for f in sorted(glob.glob(os.path.join(ROOT, "partitions", "partition_*_p*.pkl"))):
    P = pickle.load(open(f, "rb")); reg, seg = P["region"], P["segment"]
    tr = [e for c in P["noniid_train"] for e in c]; va = [e for c in P["noniid_val"] for e in c]
    ok = (all(reg[w] == "train" for e in tr for w in e["win_ids"]) and all(reg[w] == "val" for e in va for w in e["win_ids"])
          and all(len({seg[w] for w in e["win_ids"]}) == 1 for e in tr + va)
          and not ({w for e in tr for w in e["win_ids"]} & {w for e in va for w in e["win_ids"]}))
    bad += not ok
    print(f"{os.path.basename(f)}: {'OK' if ok else 'FAILED'}  train {len(tr)} seq, val {len(va)} seq ({sum(e['label'] for e in va)} attack)")
print("ALL OK" if not bad else f"{bad} FILE(S) FAILED"); sys.exit(1 if bad else 0)
