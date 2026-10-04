"""Study 7 runner (PLAN_STUDY7.md). Usage: python code/run_s7.py --workers 2"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, platform, subprocess, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DS = ["wustl", "edgeiiot", "xiiotid"]; METHODS = ["LR", "MLP", "MLPprox"]; BUDGET = {"short": (10, 3), "full": (50, 20)}
K, ALPHA, MINC, BATCH, LR, MU, NVAL = 4, 0.3, 100, 64, 1e-3, 0.1, 40_000
OUT = os.path.join(ROOT, "results", "results_s7.jsonl")
_D = {}

def data(ds):
    if ds not in _D:
        z = np.load(os.path.join(ROOT, "data", f"{ds}.npz")); _D.clear(); _D[ds] = (z["X"], z["y"], z["region"])
    return _D[ds]

_S = {}

def split(ds, ps, sp):
    if (ds, ps, sp) in _S:
        return _S[(ds, ps, sp)]
    _S.clear(); _S[(ds, ps, sp)] = _split(ds, ps, sp); return _S[(ds, ps, sp)]

def _split(ds, ps, sp):
    X, y, reg = data(ds)
    if sp == "temporal":
        val = reg == 1
    else:
        val = np.random.default_rng(1_000_003 * ps).random(len(y)) < reg.mean()
    tr = np.where(~val)[0]; va = np.where(val)[0]
    va = np.sort(np.random.default_rng(ps).choice(va, min(NVAL, len(va)), replace=False))
    lo, hi = X[tr].min(0), X[tr].max(0); rng_ = np.where(hi > lo, hi - lo, 1.0)
    sc = lambda A: np.clip((A - lo) / rng_, -10, 10).astype(np.float32) * (hi > lo)
    Xtr, Xva = sc(X[tr]), sc(X[va])
    # clients: per-class Dirichlet(alpha), redrawn until each client has >= MINC records
    for sub in range(1000):
        r = np.random.default_rng([ps, sub]); parts = [[] for _ in range(K)]
        for c in (0, 1):
            idx = np.where(y[tr] == c)[0]; r.shuffle(idx)
            cut = (np.cumsum(r.dirichlet([ALPHA] * K)) * len(idx)).astype(int)[:-1]
            for k, ch in enumerate(np.split(idx, cut)): parts[k] += ch.tolist()
        if min(map(len, parts)) >= MINC: break
    trk = set(map(bytes, np.round(Xtr, 4)))
    dup = float(np.mean([bytes(v) in trk for v in np.round(Xva, 4)]))
    return Xtr, y[tr], [np.array(p) for p in parts], Xva, y[va], dup, sub

def work(j):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(1)
    Xtr, ytr, parts, Xva, yva, dup, sub = split(j["dataset"], j["partition_seed"], j["split"])
    torch.manual_seed(j["seed"]); rs = np.random.default_rng(j["seed"] * 7 + j["partition_seed"])
    d = Xtr.shape[1]
    net = (nn.Linear(d, 2) if j["method"] == "LR" else nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)))
    mu = MU if j["method"] == "MLPprox" else 0.0
    vec = lambda m: torch.cat([p.data.reshape(-1) for p in m.parameters()]).clone()
    def setv(m, v):
        i = 0
        for p in m.parameters(): n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n
    g = vec(net); R, S = BUDGET[j["budget"]]; t0 = time.time()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    ptr = [0] * K; order = [rs.permutation(p) for p in parts]
    for rnd in range(R):
        upd, w = [], []
        for k in range(K):
            idx = parts[k]; yk = ytr[idx]; n1 = yk.sum(); n0 = len(yk) - n1
            cw = torch.tensor([len(yk) / max(1, 2 * n0), len(yk) / max(1, 2 * n1)], dtype=torch.float32)
            setv(net, g); opt = torch.optim.Adam(net.parameters(), lr=LR)
            for _ in range(S):
                if ptr[k] + BATCH > len(order[k]): order[k] = rs.permutation(parts[k]); ptr[k] = 0
                b = order[k][ptr[k]:ptr[k] + BATCH]; ptr[k] += BATCH
                loss = F.cross_entropy(net(Xt[b]), yt[b], weight=cw)
                if mu:
                    loss = loss + 0.5 * mu * ((torch.cat([p.reshape(-1) for p in net.parameters()]) - g) ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            upd.append(vec(net) - g); w.append(len(idx))
        g = g + sum((wi / sum(w)) * u for wi, u in zip(w, upd))
    setv(net, g)
    with torch.no_grad():
        pr = net(torch.from_numpy(Xva)).argmax(1).numpy()
    TP = int(((pr == 1) & (yva == 1)).sum()); FP = int(((pr == 1) & (yva == 0)).sum())
    TN = int(((pr == 0) & (yva == 0)).sum()); FN = int(((pr == 0) & (yva == 1)).sum())
    den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    mcc = 0.0 if den == 0 else (TP * TN - FP * FN) / den
    f1 = 2 * TP / (2 * TP + FP + FN) if TP else 0.0
    return dict(j, MCC=mcc, Accuracy=(TP + TN) / len(yva), F1=100 * f1, TP=TP, FP=FP, TN=TN, FN=FN,
                degenerate="always-benign" if TP + FP == 0 else "always-attack" if TN + FN == 0 else "no",
                n_train=int(len(ytr)), n_val=int(len(yva)), val_attack=int(yva.sum()), client_sizes=[int(len(p)) for p in parts],
                partition_subseed=sub, val_near_duplicate_share=round(dup, 4), secs=round(time.time() - t0, 1),
                platform=platform.platform(), torch=torch.__version__)

def jobs():
    return [dict(dataset=d, partition_seed=p, seed=s, method=m, split=sp, budget=b) for d in DS for p in range(101, 111)
            for sp in ("random", "temporal") for s in (201, 202) for m in METHODS for b in BUDGET]
key = lambda j: tuple(j[k] for k in ("dataset", "partition_seed", "seed", "method", "split", "budget"))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    if subprocess.run([sys.executable, os.path.join(HERE, "verify_seal.py")]).returncode: sys.exit("seal check failed")
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(key(json.loads(l)))
            except Exception: pass
    todo = [j for j in jobs() if key(j) not in done]; print(len(todo), "runs to do", flush=True); t0 = time.time()
    with Pool(a.workers) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(work, todo, chunksize=4), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{i}/{len(todo)}] {r['dataset']} p{r['partition_seed']} s{r['seed']} {r['method']} {r['split']} {r['budget']}: "
                  f"MCC {r['MCC']:.3f} acc {r['Accuracy']:.3f} ({r['secs']}s) ETA {(time.time() - t0) / i * (len(todo) - i) / 3600:.1f} h", flush=True)

if __name__ == "__main__":
    main()
