"""Study 4, internal re-test: new non-IID partitions (partition seeds 111-120) of the three development datasets.
Same rules as Study 3's partitions 101-110 (per-class Dirichlet(0.3) over all sequences, >=15 sequences per client,
at most 200 attempts, then the training/validation region of each sequence decides where it goes). The window
graphs, regions and sequences are taken unchanged from the Study 3 partition files. No model is run.
Usage: python code/make_dev_partitions.py   (writes partitions/dev/partition_<ds>_p111..p120.pkl)"""
import json, os, pickle, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "partitions", "dev")
K = 4
summary = []
for ds in ("wustl", "toniot", "edgeiiot"):
    P = pickle.load(open(os.path.join(DEV, f"partition_{ds}_p101.pkl"), "rb"))
    seqs = P["sequences"]
    for pseed in range(111, 121):
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
        tr = [[seqs[i] for i in sorted(c) if seqs[i]["region"] == "train"] for c in ci]
        va = [[seqs[i] for i in sorted(c) if seqs[i]["region"] == "val"] for c in ci]
        trw = {w for cl in tr for s in cl for w in s["win_ids"]}; vaw = {w for cl in va for s in cl for w in s["win_ids"]}
        assert not (trw & vaw), "shared window between training and validation"
        tr_att = [sum(s["label"] for s in x) for x in tr]
        Q = dict(P); Q["noniid_train"] = tr; Q["noniid_val"] = va; Q["partition_seed"] = pseed
        Q["noniid_attempts"] = attempt + 1
        Q["regime"] = "extreme" if sum(a >= 10 for a in tr_att) <= 1 else "moderate"
        pickle.dump(Q, open(os.path.join(DEV, f"partition_{ds}_p{pseed}.pkl"), "wb"))
        summary.append(dict(dataset=ds, partition_seed=pseed, attempts=attempt + 1, train_sizes=[len(x) for x in tr],
                            train_attack=tr_att, val_total=sum(len(x) for x in va),
                            val_attack=int(sum(s["label"] for x in va for s in x))))
        print(json.dumps(summary[-1]))
json.dump(summary, open(os.path.join(DEV, "dev_partitions_p111_p120.json"), "w"), indent=1)
