"""Turn cicapt_windows.pkl (built locally by cicapt_prepare.py) into a partition file with the same format
and the same rules as rerun_v2/setup_data.py (T=5 sequences inside one region segment, partition seed 42,
IID round-robin and per-class Dirichlet(0.3) non-IID with >=15 sequences per client), then apply the
feasibility rule of CICAPT_PROTOCOL.md (item 10). No model is run."""
import sys, pickle, random, hashlib, json
import numpy as np
import torch

src, dst = sys.argv[1], sys.argv[2]
T, K, SEED = 5, 4, 42
C = pickle.load(open(src, "rb"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
W = C["windows"]; region = C["region"]; segment = C["segment"]
windows = [None if w is None else dict(win=w["win"], node_ids=w["node_ids"], feats=torch.tensor(w["feats"]),
                                        mask=torch.tensor(w["mask"]), label=w["label"], n_flows=w["n_flows"]) for w in W]
seqs = []
for i in range(T - 1, len(windows)):
    ids = list(range(i - T + 1, i + 1))
    if any(windows[j] is None for j in ids) or len({segment[j] for j in ids}) != 1:
        continue
    seqs.append(dict(win_ids=ids, label=windows[i]["label"], anchor_idx=i, region=region[i]))


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


iid = partition_iid(len(seqs), K)
non, attempts, fallback = partition_noniid([s["label"] for s in seqs], K)


def split(cl):
    return ([[seqs[i] for i in sorted(c_) if seqs[i]["region"] == "train"] for c_ in cl],
            [[seqs[i] for i in sorted(c_) if seqs[i]["region"] == "val"] for c_ in cl])


iid_tr, iid_va = split(iid); non_tr, non_va = split(non)
for tr, va in ((iid_tr, iid_va), (non_tr, non_va)):
    trw = {w for cl in tr for s in cl for w in s["win_ids"]}; vaw = {w for cl in va for s in cl for w in s["win_ids"]}
    assert not (trw & vaw)
train_region = sorted(i for i, r in enumerate(region) if r == "train" and windows[i] is not None)
out = dict(dataset="xiiotid", IN_DIM=len(C["feature_cols"]), T=T, K=K, feature_cols=C["feature_cols"], windows=windows,
           segment=segment, region=region, train_region_window_ids=train_region, sequences=seqs,
           iid_train=iid_tr, iid_val=iid_va, noniid_train=non_tr, noniid_val=non_va,
           noniid_attempts=attempts, noniid_fallback_used=fallback)
pickle.dump(out, open(dst, "wb"))
val_attack = sum(s["label"] for cl in non_va for s in cl)
summ = dict(windows=len(windows), sequences=len(seqs), noniid_attempts=attempts, noniid_fallback=fallback,
            noniid_train=[len(x) for x in non_tr], noniid_val=[len(x) for x in non_va],
            noniid_val_attack_total=val_attack,
            feasibility=("confirmatory tests allowed" if val_attack >= 10 else "DESCRIPTIVE ONLY (protocol item 10)"),
            sha256=hashlib.sha256(open(dst, "rb").read()).hexdigest())
print(json.dumps(summ, indent=1))
json.dump(summ, open(dst.replace(".pkl", "_summary.json"), "w"), indent=1)
