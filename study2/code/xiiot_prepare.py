"""X-IIoTID -> window graphs (same output format and rules as cicapt_prepare.py; XIIOTID_CRITERIA.md)."""
import sys, pickle, hashlib
import numpy as np, pandas as pd

TOP_M, MAX_NODES, WIN = 8, 30, 60
df = pd.read_csv(sys.argv[1], dtype=str, low_memory=False)
n0 = len(df)
df["ts"] = pd.to_numeric(df["Timestamp"], errors="coerce")
ipok = df["Scr_IP"].str.contains(r"^\d+\.\d+\.\d+\.\d+$", na=False) & df["Des_IP"].str.contains(r"^\d+\.\d+\.\d+\.\d+$", na=False)
df = df[df["ts"].notna() & ipok].copy()
print("rows", n0, "->", len(df))
df["label"] = (df["class3"] == "Attack").astype(int)
EXCL = ["Date", "Timestamp", "Scr_IP", "Des_IP", "Scr_port", "Des_port", "class1", "class2", "class3", "ts", "label"]
for c in ("Protocol", "Service"):
    df[c] = df[c].astype("category").cat.codes
FEATS = [c for c in df.columns if c not in EXCL]
for c in FEATS:
    s = df[c].replace({"True": "1", "False": "0", "true": "1", "false": "0"})
    df[c] = pd.to_numeric(s, errors="coerce").fillna(0).astype(np.float64)
print("features", len(FEATS))
tmin = df["ts"].min()
b = ((df["ts"] - tmin) // WIN).astype(np.int64)
buckets = np.sort(b.unique()); W = len(buckets); widx = {k: i for i, k in enumerate(buckets)}
df["w"] = b.map(widx)
bounds = [int(round(i * W / 5)) for i in range(6)]
region = np.empty(W, dtype=object); segment = np.zeros(W, dtype=int)
for bl in range(5):
    region[bounds[bl]:bounds[bl + 1]] = "val" if bl in (1, 3) else "train"; segment[bounds[bl]:bounds[bl + 1]] = bl
tr = region[df["w"].to_numpy()] == "train"
lo = df.loc[tr, FEATS].min(); hi = df.loc[tr, FEATS].max(); rng = (hi - lo).replace(0, np.nan)
X = ((df[FEATS] - lo) / rng).fillna(0).clip(0, 1).to_numpy(np.float64)
src = df["Scr_IP"].to_numpy(); dst = df["Des_IP"].to_numpy(); wv = df["w"].to_numpy(); yv = df["label"].to_numpy()
order = np.argsort(wv, kind="mergesort"); starts = np.searchsorted(wv[order], np.arange(W + 1))
windows = []
for w in range(W):
    r = order[starts[w]:starts[w + 1]]
    label = int(yv[r].max()); s, d = src[r], dst[r]
    vc = pd.Series(np.concatenate([s, d])).value_counts()
    vc = vc.reset_index(); vc.columns = ["h", "n"]; vc = vc.sort_values(["n", "h"], ascending=[False, True])
    keep = set(vc["h"].head(MAX_NODES))
    m = np.array([a in keep and c in keep for a, c in zip(s, d)], bool)
    r, s, d = r[m], s[m], d[m]
    if len(r) == 0:
        windows.append(None); continue
    hosts = sorted(set(s) | set(d)); ix = {h: j for j, h in enumerate(hosts)}; N = len(hosts)
    si = np.array([ix[h] for h in s]); di = np.array([ix[h] for h in d])
    nf = np.zeros((N, len(FEATS))); cnt = np.zeros(N)
    np.add.at(nf, si, X[r]); np.add.at(cnt, si, 1); np.add.at(nf, di, X[r]); np.add.at(cnt, di, 1)
    ew = np.zeros((N, N), np.float32); np.add.at(ew, (si, di), 1); np.add.at(ew, (di, si), 1)
    adj = np.zeros((N, N), bool)
    for v in range(N):
        row = ew[v].copy(); row[v] = -1; me = min(TOP_M, int((row > 0).sum()))
        if me > 0:
            top = np.argpartition(-row, me - 1)[:me]; adj[v, top[row[top] > 0]] = True
    windows.append(dict(win=w, node_ids=hosts, feats=(nf / np.maximum(cnt, 1)[:, None]).astype(np.float32), mask=adj,
                        label=label, n_flows=len(r)))
out = dict(dataset="xiiotid", feature_cols=FEATS, windows=windows, region=region.tolist(), segment=segment.tolist(),
           tmin=float(tmin), W=W)
pickle.dump(out, open(sys.argv[2], "wb"))
print("windows", W, "empty", sum(x is None for x in windows), "sha256", hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest())
