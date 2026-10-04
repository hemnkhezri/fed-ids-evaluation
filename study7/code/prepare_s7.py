"""Study 7 data preparation (PLAN_STUDY7.md, "Data" and "Split protocols"). Runs no model.
Writes data/<ds>.npz (X unscaled float32, y int8, temporal region 0 = train / 1 = validation) and results/prep_log.txt."""
import hashlib, math, os, re, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = {"wustl": "/home/claude/run/data/raw/wustl_iiot_2021.csv",
       "edgeiiot": "/home/claude/data_edgefull/DNN-EdgeIIoT-dataset.csv",
       "xiiotid": "/home/claude/xprep/data/raw/X-IIoTID_dataset.csv",
       "unsw": [f"/tmp/claude-0/-home-claude/808cc933-921c-5867-8084-85e8c8c44187/scratchpad/ut/UNSW-NB15_{i}.csv" for i in range(1, 5)]}
CAP, MAXCAT = 300_000, 50
log = open(os.path.join(ROOT, "results", "prep_log.txt"), "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
UNSW_NAMES = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack",
    "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"]
SPEC = {
    "wustl": dict(reserved=["SrcAddr", "DstAddr", "Sport", "Dport", "StartTime", "LastTime", "Traffic", "Target"], label="Target", n_blocks=5),
    "edgeiiot": dict(reserved=["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port", "frame.time", "Attack_label", "Attack_type"], label="Attack_label", n_blocks=5),
    "xiiotid": dict(reserved=["Date", "Timestamp", "Scr_IP", "Des_IP", "Scr_port", "Des_port", "class1", "class2", "class3", "y"], label="y", n_blocks=5),
    "unsw": dict(reserved=["srcip", "sport", "dstip", "dsport", "Stime", "Ltime", "attack_cat", "Label", "sttl", "dttl", "ct_state_ttl"], label="Label", n_blocks=10),
}

def read(ds):
    if ds == "unsw":
        d = pd.concat([pd.read_csv(f, header=None, names=UNSW_NAMES, encoding="latin1", low_memory=False) for f in RAW[ds]], ignore_index=True)
        d["_t"] = pd.to_numeric(d["Stime"], errors="coerce")
    elif ds == "xiiotid":
        # rules of the main study's X-IIoTID preparation (xiiotid_prepare.py)
        d = pd.read_csv(RAW[ds], low_memory=False, dtype=str, keep_default_na=False)
        d["_t"] = pd.to_numeric(d["Timestamp"], errors="coerce")
        ip = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
        d = d[d["Scr_IP"].str.match(ip) & d["Des_IP"].str.match(ip)].copy()
        d["y"] = (d["class3"].str.strip() == "Attack").astype(int)
        for c in d.columns:
            if c in SPEC["xiiotid"]["reserved"] or c in ("_t", "Protocol", "Service"):
                continue
            v = d[c].str.strip().str.lower().replace({"true": "1", "false": "0"})
            d[c] = pd.to_numeric(v, errors="coerce").fillna(0.0)
    elif ds == "wustl":
        d = pd.read_csv(RAW[ds], low_memory=False)
        d["_t"] = pd.to_datetime(d["StartTime"], errors="coerce").astype("int64") / 1e9
        d.loc[pd.to_datetime(d["StartTime"], errors="coerce").isna(), "_t"] = np.nan
    else:
        d = pd.read_csv(RAW[ds], low_memory=False)
        m = d["frame.time"].astype(str).str.extract(r"(\d+):(\d+):(\d+(?:\.\d+)?)")
        d["_t"] = m[0].astype(float) * 3600 + m[1].astype(float) * 60 + m[2].astype(float)
    return d

def read_edge_chunked():
    """Same rules as read()/prep() for Edge-IIoTset, applied in two chunked passes so that the 2.2 M-row file fits
    in memory (addendum 2). Pass 1 finds the text columns and their distinct values; pass 2 codes and converts."""
    sp = SPEC["edgeiiot"]; CH = 200_000; text, uniq, over = set(), {}, set()
    for ch in pd.read_csv(RAW["edgeiiot"], chunksize=CH, low_memory=False):
        for c in ch.columns:
            if c in sp["reserved"]:
                continue
            if pd.api.types.is_object_dtype(ch[c]) or pd.api.types.is_string_dtype(ch[c]):
                text.add(c)
                if c not in over:
                    u = uniq.setdefault(c, set()); u.update(ch[c].replace("-", np.nan).dropna().unique().tolist())
                    if len(u) > MAXCAT: over.add(c)
    cat = sorted(c for c in text if c not in over); drop = sorted(over)
    vocab = {c: set() for c in cat}
    for ch in pd.read_csv(RAW["edgeiiot"], chunksize=CH, low_memory=False, usecols=lambda c: c not in drop):
        for c in cat:
            vocab[c].update(ch[c].fillna("-").astype(str).str.strip().replace("-", "unknown").unique().tolist())
    codes = {c: {x: i for i, x in enumerate(sorted(vocab[c]))} for c in cat}
    parts = []
    for ch in pd.read_csv(RAW["edgeiiot"], chunksize=CH, low_memory=False, usecols=lambda c: c not in drop):
        m = ch["frame.time"].astype(str).str.extract(r"(\d+):(\d+):(\d+(?:\.\d+)?)")
        o = pd.DataFrame({"_t": m[0].astype(float) * 3600 + m[1].astype(float) * 60 + m[2].astype(float),
                          sp["label"]: pd.to_numeric(ch[sp["label"]], errors="coerce")})
        for c in ch.columns:
            if c in sp["reserved"]:
                continue
            if c in cat:
                o[c] = ch[c].fillna("-").astype(str).str.strip().replace("-", "unknown").map(codes[c]).astype(np.float32)
            else:
                o[c] = pd.to_numeric(ch[c], errors="coerce").astype(np.float32)
        parts.append(o)
    d = pd.concat(parts, ignore_index=True); del parts
    return d, cat, drop

def prep(ds):
    sp = SPEC[ds]
    if ds == "edgeiiot":
        d, cat0, drop0 = read_edge_chunked()
    else:
        d = read(ds); cat0 = None
    n0 = len(d)
    d[sp["label"]] = pd.to_numeric(d[sp["label"]], errors="coerce")
    d = d.dropna(subset=["_t", sp["label"]]); say(ds, "rows", n0, "with time and label", len(d))
    feats, cat, drop = [], [], []
    if cat0 is not None:
        cat, drop = cat0, drop0
        feats = [c for c in d.columns if c not in sp["reserved"] and c != "_t" and c not in cat]
    for c in ([] if cat0 is not None else d.columns):
        if c in sp["reserved"] or c == "_t":
            continue
        if pd.api.types.is_object_dtype(d[c]) or pd.api.types.is_string_dtype(d[c]):
            nu = d[c].replace("-", np.nan).nunique(dropna=True)
            (cat if nu <= MAXCAT else drop).append(c)
        else:
            feats.append(c)
    for c in ([] if cat0 is not None else cat):
        v = d[c].fillna("-").astype(str).str.strip().replace("-", "unknown")
        voc = {x: i for i, x in enumerate(sorted(v.unique()))}; d[c] = v.map(voc).astype(np.float64)
    for c in feats:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    cols = feats + cat
    X = d[cols].to_numpy(dtype=np.float64); X[~np.isfinite(X)] = 0.0
    y = d[sp["label"]].to_numpy().astype(np.int8); t = d["_t"].to_numpy()
    say(ds, "features", len(cols), "categorical", cat, "dropped", drop)
    # exact duplicates (features + label)
    key = pd.DataFrame(np.column_stack([X, y])).duplicated().to_numpy()
    X, y, t = X[~key], y[~key], t[~key]; say(ds, "after duplicate removal", len(y))
    o = np.argsort(t, kind="stable"); X, y, t = X[o], y[o], t[o]
    if len(y) > CAP:
        k = math.ceil(len(y) / CAP); X, y, t = X[::k], y[::k], t[::k]; say(ds, f"systematic sample every {k}th record ->", len(y))
    def regions(nb):
        b = np.minimum((np.arange(len(y)) * nb) // len(y), nb - 1)
        return (b % 2 == 1).astype(np.int8)          # 1-based blocks 2, 4, ... are validation
    reg = regions(sp["n_blocks"])
    def ok(r):
        return all(min((y[r == g] == 0).mean(), (y[r == g] == 1).mean()) >= 0.01 for g in (0, 1))
    if not ok(reg):
        say(ds, f"label-count rule: {sp['n_blocks']} blocks leave a class below 1% in a region -> 10 interleaved blocks")
        reg = regions(10)
    for g, nm in ((0, "train"), (1, "validation")):
        say(ds, nm, int((reg == g).sum()), "records, attack share", round(float(y[reg == g].mean()), 4))
    say(ds, "region rule satisfied:", ok(reg))
    out = os.path.join(ROOT, "data", f"{ds}.npz")
    np.savez_compressed(out, X=X.astype(np.float32), y=y, region=reg, cols=np.array(cols))
    say(ds, "saved", out, "sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())

if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["wustl", "xiiotid", "edgeiiot"]):
        prep(ds)
