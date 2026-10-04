"""Phase-2 partitions: new Dirichlet(0.3) non-IID partitions (partition seeds 101-110) of the existing
window/sequence files, same rules as study 1 (per-class Dirichlet, >=15 sequences per client, 200 attempts),
and the pre-specified regime classification (PHASE2_PLAN.md): a partition is EXTREME-SKEW if at most one of
the K=4 clients holds >= 10 attack sequences in its training split. No model is run."""
import sys, pickle, random, json
import numpy as np

src, out_dir, ds = sys.argv[1], sys.argv[2], sys.argv[3]
P = pickle.load(open(src, "rb"))
seqs = P["sequences"]; K = 4
summary = []
for pseed in range(101, 111):
    random.seed(pseed); np.random.seed(pseed)
    labels = np.array([s["label"] for s in seqs])
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
    fallback = attempt == 199 and min(len(x) for x in ci) < 15
    tr = [[seqs[i] for i in sorted(c) if seqs[i]["region"] == "train"] for c in ci]
    va = [[seqs[i] for i in sorted(c) if seqs[i]["region"] == "val"] for c in ci]
    tr_att = [sum(s["label"] for s in x) for x in tr]
    extreme = sum(a >= 10 for a in tr_att) <= 1
    Q = dict(P); Q["noniid_train"] = tr; Q["noniid_val"] = va
    Q["partition_seed"] = pseed; Q["regime"] = "extreme" if extreme else "moderate"
    for k in ("iid_train", "iid_val"):
        Q.pop(k, None)
    pickle.dump(Q, open(f"{out_dir}/partition_{ds}_p{pseed}.pkl", "wb"))
    summary.append(dict(dataset=ds, partition_seed=pseed, attempts=attempt + 1, fallback=bool(fallback),
                        regime=Q["regime"], train_sizes=[len(x) for x in tr], train_attack=tr_att,
                        val_sizes=[len(x) for x in va], val_attack_total=int(sum(s["label"] for x in va for s in x))))
    print(json.dumps(summary[-1]))
json.dump(summary, open(f"{out_dir}/phase2_partitions_{ds}.json", "w"), indent=1)
