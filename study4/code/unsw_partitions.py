"""UNSW-NB15: window graphs -> T=5 sequences -> ten non-IID partitions (seeds 101-110), then the eligibility check
of PLAN_STUDY4.md. Same sequence and partition rules as the other datasets. No model is run.
Usage: python code/unsw_partitions.py
Input: data/unsw_windows.pkl. Output: data/unsw_base.pkl, partitions/unsw/partition_unsw_p101..p110.pkl,
results/unsw_eligibility.json (the run stops later if the dataset is not eligible)."""
import hashlib, json, os, pickle, random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T, K = 5, 4
C = pickle.load(open(os.path.join(ROOT, "data", "unsw_windows.pkl"), "rb"))
region, segment = C["region"], C["segment"]
windows = [None if w is None else dict(win=w["win"], node_ids=w["node_ids"], feats=torch.tensor(w["feats"]),
                                        mask=torch.tensor(w["mask"]), label=w["label"], n_flows=w["n_flows"])
           for w in C["windows"]]
seqs = []
for i in range(T - 1, len(windows)):
    ids = list(range(i - T + 1, i + 1))
    if any(windows[j] is None for j in ids) or len({segment[j] for j in ids}) != 1:
        continue
    seqs.append(dict(win_ids=ids, label=windows[i]["label"], anchor_idx=i, region=region[i]))
base = dict(dataset="unsw", IN_DIM=len(C["feature_cols"]), T=T, K=K, feature_cols=C["feature_cols"], windows=windows,
            segment=segment, region=region,
            train_region_window_ids=sorted(i for i, r in enumerate(region) if r == "train" and windows[i] is not None),
            sequences=seqs)
bp = os.path.join(ROOT, "data", "unsw_base.pkl")
pickle.dump(base, open(bp, "wb"))
os.makedirs(os.path.join(ROOT, "partitions", "unsw"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

labels = np.array([s["label"] for s in seqs])
summ = []
for pseed in range(101, 111):
    random.seed(pseed); np.random.seed(pseed)
    by_c = {c: np.where(labels == c)[0].tolist() for c in np.unique(labels)}
    for attempt in range(200):
        ci = [[] for _ in range(K)]
        for c, ids in by_c.items():
            sh = ids.copy(); random.shuffle(sh)
            p = np.random.dirichlet([0.3] * K)
            cuts = (np.cumsum(p) * len(sh)).astype(int)[:-1]
            for j, part in enumerate(np.split(np.array(sh), cuts)):
                ci[j].extend(part.tolist())
        if min(len(x) for x in ci) >= 15:
            break
    tr = [[seqs[i]["anchor_idx"] for i in sorted(c) if seqs[i]["region"] == "train"] for c in ci]
    va = [[seqs[i]["anchor_idx"] for i in sorted(c) if seqs[i]["region"] == "val"] for c in ci]
    by_anchor = {s["anchor_idx"]: s for s in seqs}
    trw = {w for cl in tr for a in cl for w in by_anchor[a]["win_ids"]}
    vaw = {w for cl in va for a in cl for w in by_anchor[a]["win_ids"]}
    assert not (trw & vaw), "shared window between training and validation"
    tr_att = [sum(by_anchor[a]["label"] for a in cl) for cl in tr]
    Q = dict(base="data/unsw_base.pkl", dataset="unsw", partition_seed=pseed, noniid_attempts=attempt + 1,
             train_anchor=tr, val_anchor=va, regime="extreme" if sum(a >= 10 for a in tr_att) <= 1 else "moderate")
    pf = os.path.join(ROOT, "partitions", "unsw", f"partition_unsw_p{pseed}.pkl")
    pickle.dump(Q, open(pf, "wb"))
    va_att = int(sum(by_anchor[a]["label"] for cl in va for a in cl)); va_n = sum(len(cl) for cl in va)
    summ.append(dict(partition_seed=pseed, attempts=attempt + 1, train_sizes=[len(x) for x in tr], train_attack=tr_att,
                     val_total=va_n, val_attack=va_att, sha256=hashlib.sha256(open(pf, "rb").read()).hexdigest()))
    print(json.dumps(summ[-1]))

tr_att_total = min(sum(s["train_attack"]) for s in summ)
tr_ben_total = min(sum(s["train_sizes"]) - sum(s["train_attack"]) for s in summ)
va_att = summ[0]["val_attack"]; va_ben = summ[0]["val_total"] - va_att
crit = {"C1 training attack sequences >= 500 in every partition": tr_att_total >= 500,
        "C2 training benign sequences >= 500 in every partition": tr_ben_total >= 500,
        "C3 validation attack sequences >= 30": va_att >= 30,
        "C4 validation benign sequences >= 30": va_ben >= 30}
elig = dict(eligible=all(crit.values()), criteria=crit, windows=len(windows), sequences=len(seqs),
            train_attack_min=tr_att_total, train_benign_min=tr_ben_total, val_attack=va_att, val_benign=va_ben,
            base_sha256=hashlib.sha256(open(bp, "rb").read()).hexdigest(), partitions=summ)
json.dump(elig, open(os.path.join(ROOT, "results", "unsw_eligibility.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in elig.items() if k != "partitions"}, indent=1))
print("ELIGIBLE" if elig["eligible"] else "NOT ELIGIBLE: the external test is not run (PLAN_STUDY4.md)")
