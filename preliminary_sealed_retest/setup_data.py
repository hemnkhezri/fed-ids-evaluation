"""
Leakage-free data setup (manuscript Section 4.2), re-implemented from the written
specification after the 2026-09-26 environment reset (see ANALYSIS_PLAN_v2_ADDENDUM.md).

For each dataset:
  1. load raw CSV, drop duplicates / label-missing rows, column-role classification
     (text fields with <=50 distinct values integer-encoded, others dropped; numeric
     fields coerced, NaN -> 0). NO normalisation yet.
  2. assign rows to windows (WUSTL, Edge-IIoTset: 60-s buckets; TON_IoT: 300-row
     chunks inside each single-scenario block).
  3. assign windows to regions: WUSTL / Edge -> five contiguous equal-length blocks of
     the global window timeline, blocks 2 and 4 (1-based) = validation; TON_IoT -> first
     80% of each block's windows = training, rest = validation.
  4. Eq. (1) min/max fitted on training-region rows only, applied to all rows, clipped
     to [0,1].
  5. per-window device graph (host nodes, mean node features, top-m=8 adjacency).
  6. T=5 stride-1 sequences whose five windows lie in one contiguous region segment
     (sequences straddling a boundary are discarded).
  7. IID and per-class Dirichlet(alpha=0.3) non-IID partitions over sequences
     (partition seed 42, min 15 sequences per client, rejection sampling, 200 attempts);
     each client's train / val = its sequences in the train / val region.
  8. programmatic check: no raw window shared between any train and any val sequence.
"""
import argparse, pickle, random, re, json, hashlib
import numpy as np
import pandas as pd
import torch

UP = "/mnt/user-data/uploads/"
CFG = {
    "wustl": dict(path=UP + "wustl_iiot_2021.csv", ids=["SrcAddr", "DstAddr", "Sport", "Dport"],
                  times=["StartTime", "LastTime"], label="Target", typ="Traffic",
                  src="SrcAddr", dst="DstAddr"),
    "toniot": dict(path=UP + "TON_IoT_Network_Dataset_train_test_network.csv",
                   ids=["src_ip", "dst_ip", "src_port", "dst_port"], times=[], label="label",
                   typ="type", src="src_ip", dst="dst_ip"),
    "edgeiiot": dict(path=UP + "EdgeIIoT_sample2_200k.csv",
                     ids=["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port"],
                     times=["frame.time"], label="Attack_label", typ="Attack_type",
                     src="ip.src_host", dst="ip.dst_host"),
}
T, K, TOP_M, MAX_NODES, WINDOW_ROWS = 5, 4, 8, 30, 300
PART_SEED = 42


def load_clean(c):
    df = pd.read_csv(c["path"], low_memory=False)
    n0 = len(df)
    df = df.drop_duplicates().dropna(subset=[c["label"], c["typ"]]).reset_index(drop=True)
    reserved = set(c["ids"]) | set(c["times"]) | {c["label"], c["typ"]}
    cats, drops, nums = [], [], []
    for col in df.columns:
        if col in reserved:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            (cats if df[col].replace("-", np.nan).nunique(dropna=True) <= 50 else drops).append(col)
        else:
            nums.append(col)
    df = df.drop(columns=drops)
    for col in cats:
        df[col] = df[col].replace("-", "unknown").astype(str).astype("category").cat.codes
    for col in nums:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    feats = cats + nums
    df[feats] = df[feats].astype(np.float64)
    df[c["label"]] = df[c["label"]].astype(int)
    for col in (c["src"], c["dst"]):
        df[col] = df[col].astype(str)
    print(f"rows {n0} -> {len(df)}; features {len(feats)} (cat {len(cats)}, num {len(nums)}, dropped {drops})")
    return df, feats


def parse_secs(s):
    m = re.search(r"(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", str(s))
    if not m:
        return np.nan
    h, mi, se = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(se)


def assign_windows(ds, df, c):
    """Returns df with integer column 'win' (0..W-1, chronological/order) and a
    per-window 'segment key' list used for regioning."""
    if ds == "wustl":
        t = pd.to_datetime(df["StartTime"])
        df = df.assign(_t=t).sort_values("_t", kind="mergesort").reset_index(drop=True)
        wid = ((df["_t"] - df["_t"].min()).dt.total_seconds() // 60).astype(int)
        df["win"] = pd.factorize(wid, sort=True)[0]
        df["block"] = 0
    elif ds == "edgeiiot":
        df = df.assign(_t=df["frame.time"].map(parse_secs))
        n0 = len(df)
        df = df.dropna(subset=["_t"]).sort_values("_t", kind="mergesort").reset_index(drop=True)
        print(f"dropped {n0-len(df)} rows with unparseable frame.time")
        wid = ((df["_t"] - df["_t"].min()) // 60).astype(int)
        df["win"] = pd.factorize(wid, sort=True)[0]
        df["block"] = 0
    else:  # toniot: file order, single-scenario blocks, 300-row chunks inside a block
        df["block"] = (df[c["typ"]] != df[c["typ"]].shift()).cumsum() - 1
        pos = df.groupby("block").cumcount()
        size = df.groupby("block")["block"].transform("size")
        nwin = size // WINDOW_ROWS
        chunk = pos // WINDOW_ROWS
        keep = chunk < nwin
        df = df[keep].copy()
        df["chunk"] = chunk[keep]
        key = df["block"].astype(np.int64) * 100000 + df["chunk"].astype(np.int64)
        df["win"] = pd.factorize(key, sort=True)[0]
    return df


def region_of_windows(ds, win_block):
    """win_block: array (W,) block id per window (all 0 for timeline datasets).
    Returns region array ('train'/'val') and segment id per window (sequences must stay
    inside one segment)."""
    W = len(win_block)
    region = np.empty(W, dtype=object)
    segment = np.zeros(W, dtype=int)
    if ds in ("wustl", "edgeiiot"):
        bounds = [int(round(i * W / 5)) for i in range(6)]
        for b in range(5):
            region[bounds[b]:bounds[b + 1]] = "val" if b in (1, 3) else "train"
            segment[bounds[b]:bounds[b + 1]] = b
    else:
        seg = 0
        for blk in np.unique(win_block):
            idx = np.where(win_block == blk)[0]
            cut = int(0.8 * len(idx))
            region[idx[:cut]] = "train"; segment[idx[:cut]] = seg; seg += 1
            region[idx[cut:]] = "val"; segment[idx[cut:]] = seg; seg += 1
    return region, segment


def build_graphs(ds, df, c, feats):
    X = df[feats].to_numpy(np.float32)
    win = df["win"].to_numpy()
    y = df[c["label"]].to_numpy()
    src = df[c["src"]].to_numpy(); dst = df[c["dst"]].to_numpy()
    order = np.argsort(win, kind="mergesort")
    starts = np.searchsorted(win[order], np.arange(win.max() + 2))
    windows = []
    for w in range(win.max() + 1):
        r = order[starts[w]:starts[w + 1]]
        s, d_ = src[r], dst[r]
        if ds == "edgeiiot":  # cap to MAX_NODES most active hosts (by flow count)
            vc = pd.Series(np.concatenate([s, d_])).value_counts()
            keep = set(vc.head(MAX_NODES).index)
            m = np.array([a in keep and b in keep for a, b in zip(s, d_)], dtype=bool)
            if m.sum() == 0:
                windows.append(None); continue
            r, s, d_ = r[m], s[m], d_[m]
        hosts = sorted(set(s.tolist()) | set(d_.tolist()))
        idx = {h: i for i, h in enumerate(hosts)}
        N = len(hosts)
        si = np.fromiter((idx[h] for h in s), int, len(s))
        di = np.fromiter((idx[h] for h in d_), int, len(d_))
        nf = np.zeros((N, X.shape[1]), np.float64); cnt = np.zeros(N)
        np.add.at(nf, si, X[r]); np.add.at(cnt, si, 1)
        np.add.at(nf, di, X[r]); np.add.at(cnt, di, 1)
        nf = (nf / np.maximum(cnt, 1)[:, None]).astype(np.float32)
        ew = np.zeros((N, N), np.float32)
        np.add.at(ew, (si, di), 1.0); np.add.at(ew, (di, si), 1.0)
        adj = np.zeros((N, N), bool)
        for v in range(N):
            row = ew[v].copy(); row[v] = -1
            m_eff = min(TOP_M, int((row > 0).sum()))
            if m_eff > 0:
                top = np.argpartition(-row, m_eff - 1)[:m_eff]
                adj[v, top[row[top] > 0]] = True
        windows.append(dict(win=w, node_ids=hosts, feats=torch.tensor(nf), mask=torch.tensor(adj),
                            label=int(y[r].max()), n_flows=len(r)))
    return windows


def partition_iid(n, k):
    idx = list(range(n)); random.shuffle(idx)
    return [idx[i::k] for i in range(k)]


def partition_noniid(labels, k, alpha=0.3, min_per_client=15):
    labels = np.asarray(labels)
    by_c = {cl: np.where(labels == cl)[0].tolist() for cl in np.unique(labels)}
    for attempt in range(200):
        ci = [[] for _ in range(k)]
        for cl, ids in by_c.items():
            sh = ids.copy(); random.shuffle(sh)
            p = np.random.dirichlet([alpha] * k)
            cuts = (np.cumsum(p) * len(sh)).astype(int)[:-1]
            for j, part in enumerate(np.split(np.array(sh), cuts)):
                ci[j].extend(part.tolist())
        if min(len(c_) for c_ in ci) >= min_per_client:
            return ci, attempt + 1, False
    return ci, 200, True


def main(ds):
    c = CFG[ds]
    random.seed(PART_SEED); np.random.seed(PART_SEED); torch.manual_seed(PART_SEED)
    df, feats = load_clean(c)
    df = assign_windows(ds, df, c)
    W = int(df["win"].max()) + 1
    win_block = df.groupby("win")["block"].first().reindex(range(W)).to_numpy()
    region, segment = region_of_windows(ds, win_block)
    # Eq. (1): min/max fitted on training-region rows only
    tr_rows = region[df["win"].to_numpy()] == "train"
    lo = df.loc[tr_rows, feats].min(); hi = df.loc[tr_rows, feats].max()
    rng_ = (hi - lo).replace(0, np.nan)
    df[feats] = ((df[feats] - lo) / rng_).fillna(0.0).clip(0.0, 1.0)
    windows = build_graphs(ds, df, c, feats)
    # drop empty windows (edge host-cap) while keeping order; they break contiguity
    valid = [w for w in windows if w is not None]
    n_empty = len(windows) - len(valid)
    # sequences: 5 consecutive (in window order) windows, same segment, none empty
    seqs = []
    for i in range(T - 1, W):
        ids = list(range(i - T + 1, i + 1))
        if any(windows[j] is None for j in ids):
            continue
        if len({segment[j] for j in ids}) != 1:
            continue
        seqs.append(dict(win_ids=ids, label=windows[i]["label"], anchor_idx=i, region=region[i]))
    labels = [s["label"] for s in seqs]
    iid = partition_iid(len(seqs), K)
    non, attempts, fallback = partition_noniid(labels, K)

    def split(cl):
        tr = [[seqs[i] for i in sorted(c_) if seqs[i]["region"] == "train"] for c_ in cl]
        va = [[seqs[i] for i in sorted(c_) if seqs[i]["region"] == "val"] for c_ in cl]
        return tr, va
    iid_tr, iid_va = split(iid); non_tr, non_va = split(non)
    # leakage check
    for tr, va in ((iid_tr, iid_va), (non_tr, non_va)):
        trw = {w for cl in tr for s in cl for w in s["win_ids"]}
        vaw = {w for cl in va for s in cl for w in s["win_ids"]}
        assert not (trw & vaw), "LEAK: shared raw window between train and val"
        assert all(region[w] == "train" for w in trw) and all(region[w] == "val" for w in vaw)
    train_region = sorted(int(i) for i in np.where(region == "train")[0] if windows[i] is not None)
    out = dict(dataset=ds, IN_DIM=len(feats), T=T, K=K, feature_cols=feats,
               windows=windows, segment=segment.tolist(), region=region.tolist(),
               train_region_window_ids=train_region, sequences=seqs,
               iid_train=iid_tr, iid_val=iid_va, noniid_train=non_tr, noniid_val=non_va,
               noniid_attempts=attempts, noniid_fallback_used=fallback)
    path = f"/mnt/user-data/outputs/rerun_v2/data/partition_{ds}_seed42.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    summ = dict(dataset=ds, rows=len(df), features=len(feats), windows=W, empty_windows=n_empty,
                window_attack=sum(w["label"] for w in valid), sequences=len(seqs),
                seq_attack=int(sum(labels)), noniid_attempts=attempts, noniid_fallback=fallback)
    for nm, tr, va in (("iid", iid_tr, iid_va), ("noniid", non_tr, non_va)):
        summ[nm] = dict(train=[len(x) for x in tr], train_attack=[sum(s["label"] for s in x) for x in tr],
                        val=[len(x) for x in va], val_attack=[sum(s["label"] for s in x) for x in va])
    summ["sha256_pkl"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(json.dumps(summ, indent=1))
    with open(f"/mnt/user-data/outputs/rerun_v2/partition_summary_{ds}.json", "w") as f:
        json.dump(summ, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); a = ap.parse_args()
    main(a.dataset)
