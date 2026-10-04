"""UNSW-NB15 label-balance check (Study 4, step 0). No model is run and no feature is read.

For several window lengths it counts non-empty windows and the share of attack windows in each of the five
chronological blocks (blocks 2 and 4 = validation region, as for the other datasets). The window length for
Study 4 is then chosen by a rule that uses only these label counts (PLAN_STUDY4.md).

Usage (PowerShell, in the folder that holds UNSW-NB15_1.csv ... UNSW-NB15_4.csv):
    python unsw_check.py
Output: unsw_check.txt (send this file back).
"""
import glob, os, time, hashlib
import numpy as np
import pandas as pd

FILES = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
LENGTHS = [60, 30, 10, 5, 2, 1]
out = open("unsw_check.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()


t0 = time.time()
parts = []
for f in FILES:
    if not os.path.exists(f):
        raise SystemExit(f"missing {f}: run this script in the folder with the four UNSW-NB15 CSV files")
    say(f, os.path.getsize(f), "bytes", "sha256", hashlib.sha256(open(f, "rb").read()).hexdigest())
    d = pd.read_csv(f, header=None, usecols=[0, 2, 28, 48], names=["srcip", "dstip", "stime", "label"],
                    encoding="latin1", low_memory=False)
    parts.append(d)
D = pd.concat(parts, ignore_index=True)
D["stime"] = pd.to_numeric(D["stime"], errors="coerce")
D["label"] = pd.to_numeric(D["label"], errors="coerce")
bad = D["stime"].isna() | D["label"].isna()
say("rows", len(D), "unparseable time/label rows", int(bad.sum()))
D = D[~bad]
say("attack rows", int((D.label == 1).sum()), f"({(D.label == 1).mean():.3%})")
t = D["stime"].to_numpy(np.float64); y = D["label"].to_numpy(np.int64)
say("time span", int(t.min()), int(t.max()), f"{(t.max() - t.min()) / 3600:.1f} h")
ts = np.unique(np.floor(t))
gaps = np.diff(ts); big = np.where(gaps > 3600)[0]
say("gaps longer than 1 h:", [(int(ts[i]), int(gaps[i] // 3600)) for i in big])
say("distinct hosts:", D["srcip"].nunique(), "src,", D["dstip"].nunique(), "dst")

for L in LENGTHS:
    b = np.floor((t - t.min()) / L).astype(np.int64)
    lab = pd.Series(y).groupby(b).max()
    rows = pd.Series(y).groupby(b).size()
    W = len(lab); v = lab.to_numpy()
    bounds = [int(round(i * W / 5)) for i in range(6)]
    say(f"\n== window {L} s: {W} non-empty windows, attack windows {v.mean():.1%}, median rows/window {int(rows.median())}")
    tr_a = tr_n = va_a = va_n = 0
    for bl in range(5):
        seg = v[bounds[bl]:bounds[bl + 1]]
        reg = "val" if bl in (1, 3) else "train"
        say(f"   block {bl + 1} ({reg}): {len(seg)} windows, attack {(seg.mean() if len(seg) else 0):.1%}, benign windows {int((seg == 0).sum())}")
        if reg == "val":
            va_a += int(seg.sum()); va_n += len(seg)
        else:
            tr_a += int(seg.sum()); tr_n += len(seg)
    say(f"   train region attack share {tr_a / max(tr_n, 1):.1%} | val region attack share {va_a / max(va_n, 1):.1%}, "
        f"val benign windows {va_n - va_a}, val attack windows {va_a}")
say(f"\ndone in {time.time() - t0:.0f} s")
