"""UNSW-NB15 preprocessing for Study 4 (rules in PLAN_STUDY4.md, section "External dataset"). Runs locally,
reads the four raw CSV files in chunks, runs no model. Only numpy and pandas are needed.

Usage (from the Study 4 folder):
    python code/unsw_prepare.py "C:/path/to/folder/with/UNSW-NB15_1.csv"
Output: data/unsw_windows.pkl and data/unsw_prepare_log.txt
"""
import hashlib, os, pickle, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else "."
FILES = [os.path.join(SRC, f"UNSW-NB15_{i}.csv") for i in range(1, 5)]
OUT_DIR = os.path.join(ROOT, "data"); os.makedirs(OUT_DIR, exist_ok=True)
CHUNK = 400_000
WIN, N_BLOCKS, GAP = 1, 10, 3600          # 1-s windows, 10 interleaved blocks, capture sessions split at gaps > 1 h
TOP_M, MAX_NODES = 8, 30
NAMES = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
         "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
         "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack",
         "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
         "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
         "attack_cat", "Label"]
IDENT = ["srcip", "sport", "dstip", "dsport"]               # identifiers
ABS_TIME = ["Stime", "Ltime"]                               # absolute time
LABELS = ["attack_cat", "Label"]
TTL = ["sttl", "dttl", "ct_state_ttl"]                      # testbed artefact (see PLAN_STUDY4.md)
CATEG = ["proto", "state", "service"]                       # integer-encoded
FEATS = [c for c in NAMES if c not in IDENT + ABS_TIME + LABELS + TTL]
log = open(os.path.join(OUT_DIR, "unsw_prepare_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


def chunks(cols):
    for f in FILES:
        for ch in pd.read_csv(f, header=None, names=NAMES, usecols=cols, chunksize=CHUNK, encoding="latin1",
                              dtype={c: str for c in ("srcip", "dstip", "proto", "state", "service") if c in cols},
                              low_memory=False):
            ch["Stime"] = pd.to_numeric(ch["Stime"], errors="coerce")
            if "Label" in ch:
                ch["Label"] = pd.to_numeric(ch["Label"], errors="coerce")
            ch = ch.dropna(subset=["Stime"] + (["Label"] if "Label" in ch else []))
            yield ch


for f in FILES:
    if not os.path.exists(f):
        raise SystemExit(f"missing {f}")
    say(os.path.basename(f), "sha256", hashlib.sha256(open(f, "rb").read()).hexdigest())
say("features", len(FEATS), FEATS)
t0 = time.time()

# ---------------- pass 1: time origin, labels and host activity per window, category vocabularies
tmin = min(ch["Stime"].min() for ch in chunks(["Stime"]))
bucket_label, host_cnt, vocab = {}, {}, {c: set() for c in CATEG}
for ch in chunks(["srcip", "dstip", "Stime", "Label"] + CATEG):
    b = ((ch["Stime"] - tmin) // WIN).astype(np.int64)
    for k, v in ch.groupby(b)["Label"].max().items():
        bucket_label[k] = max(bucket_label.get(k, 0), int(v))
    for col in ("srcip", "dstip"):
        for (k, h), v in ch.groupby([b, ch[col]]).size().items():
            d = host_cnt.setdefault(k, {}); d[h] = d.get(h, 0) + int(v)
    for c in CATEG:
        vocab[c].update(ch[c].fillna("-").astype(str).str.strip().unique().tolist())
say(f"pass1 {time.time() - t0:.0f}s")
codes = {c: {v: i for i, v in enumerate(sorted(vocab[c]))} for c in CATEG}
buckets = sorted(bucket_label); W = len(buckets)
bounds = [int(round(i * W / N_BLOCKS)) for i in range(N_BLOCKS + 1)]
block = np.zeros(W, dtype=int)
for bl in range(N_BLOCKS):
    block[bounds[bl]:bounds[bl + 1]] = bl
region = np.where(block % 2 == 1, "val", "train")          # blocks 2, 4, 6, 8, 10 (1-based) are validation
session = np.concatenate([[0], np.cumsum(np.diff(np.array(buckets)) > GAP // WIN)])
segment = (block * 10 + session).tolist()
keep = {k: set(h for h, _ in sorted(d.items(), key=lambda x: (-x[1], x[0]))[:MAX_NODES]) for k, d in host_cnt.items()}
lab = np.array([bucket_label[k] for k in buckets])
say("windows", W, "attack windows", int(lab.sum()), "sessions", int(session.max()) + 1)
for bl in range(N_BLOCKS):
    s = lab[bounds[bl]:bounds[bl + 1]]
    say(f"  block {bl + 1} ({'val' if bl % 2 else 'train'}): {len(s)} windows, attack {s.mean():.1%}")


def numeric(ch):
    X = ch[FEATS].copy()
    for c in CATEG:
        X[c] = X[c].fillna("-").astype(str).str.strip().map(codes[c])
    return X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)


# ---------------- pass 2: min/max over training-region rows
train_b = set(buckets[i] for i in range(W) if region[i] == "train")
lo = pd.Series(np.inf, index=FEATS); hi = pd.Series(-np.inf, index=FEATS)
for ch in chunks(["Stime"] + FEATS):
    b = ((ch["Stime"] - tmin) // WIN).astype(np.int64)
    sub = numeric(ch.loc[b.isin(train_b)])
    if len(sub):
        lo = np.minimum(lo, sub.min()); hi = np.maximum(hi, sub.max())
say(f"pass2 {time.time() - t0:.0f}s")
rng = (hi - lo).replace(0, np.nan)

# ---------------- pass 3: node features (mean of normalised rows) and edge counts between kept hosts
node_sum, node_cnt, edge_w = {}, {}, {}
for ch in chunks(["srcip", "dstip", "Stime"] + FEATS):
    b = ((ch["Stime"] - tmin) // WIN).astype(np.int64).to_numpy()
    s = ch["srcip"].to_numpy(); d = ch["dstip"].to_numpy()
    ok = np.fromiter((si in keep[bi] and di in keep[bi] for bi, si, di in zip(b, s, d)), bool, len(b))
    X = ((numeric(ch) - lo) / rng).fillna(0).clip(0, 1).to_numpy(np.float64)[ok]
    b = b[ok]; s = s[ok]; d = d[ok]
    for ends in (s, d):
        g = pd.DataFrame(X).groupby(pd.MultiIndex.from_arrays([b, ends]))
        for kk, row in zip(*(lambda t: (t.index, t.to_numpy()))(g.sum())):
            if kk in node_sum:
                node_sum[kk] += row
            else:
                node_sum[kk] = row.copy()
        for kk, c in g.size().items():
            node_cnt[kk] = node_cnt.get(kk, 0) + int(c)
    for kk, c in pd.Series(1, index=pd.MultiIndex.from_arrays([b, s, d])).groupby(level=[0, 1, 2]).size().items():
        edge_w[kk] = edge_w.get(kk, 0) + int(c)
say(f"pass3 {time.time() - t0:.0f}s")

# ---------------- window graphs
by_nodes, by_edges = {}, {}
for (bi, h) in node_cnt:
    by_nodes.setdefault(bi, []).append(h)
for (bi, si, di), c in edge_w.items():
    by_edges.setdefault(bi, []).append((si, di, c))
windows = []
for i, bk in enumerate(buckets):
    hosts = sorted(by_nodes.get(bk, []))
    if not hosts:
        windows.append(None); continue
    ix = {h: j for j, h in enumerate(hosts)}; N = len(hosts)
    feats = np.stack([node_sum[(bk, h)] / node_cnt[(bk, h)] for h in hosts]).astype(np.float32)
    ew = np.zeros((N, N), np.float32)
    for si, di, c in by_edges.get(bk, []):
        ew[ix[si], ix[di]] += c; ew[ix[di], ix[si]] += c
    adj = np.zeros((N, N), bool)
    for v in range(N):
        row = ew[v].copy(); row[v] = -1
        me = min(TOP_M, int((row > 0).sum()))
        if me > 0:
            top = np.argpartition(-row, me - 1)[:me]; adj[v, top[row[top] > 0]] = True
    windows.append(dict(win=i, node_ids=hosts, feats=feats, mask=adj, label=int(bucket_label[bk]),
                        n_flows=int(sum(node_cnt[(bk, h)] for h in hosts) // 2)))
out = dict(dataset="unsw", feature_cols=FEATS, windows=windows, region=region.tolist(), segment=segment,
           block=block.tolist(), tmin=float(tmin), W=W, category_codes=codes)
dst = os.path.join(OUT_DIR, "unsw_windows.pkl")
with open(dst, "wb") as f:
    pickle.dump(out, f)
say("empty windows after host capping:", sum(w is None for w in windows))
say("median hosts per window:", int(np.median([len(w["node_ids"]) for w in windows if w is not None])))
say("sha256", hashlib.sha256(open(dst, "rb").read()).hexdigest())
say(f"done in {time.time() - t0:.0f}s")
