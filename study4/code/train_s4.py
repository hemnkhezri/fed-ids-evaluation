"""Study 4 runner: train_s3.py plus one option, cfg["labelled_only"] (encoder family draws its mini-batches from
labelled sequences only, as the supervised baselines do). Everything else is train_s3.py unchanged.

FedGTCL study 3 runner (fork of the study 2 runner train_v3.py).

Changes from train_v3.py, and only these:
  1. Equal optimizer budget: every client of every method takes exactly `steps` Adam steps per round, each on a
     mini-batch of up to 16 sequences drawn by cycling through its (shuffled) usable data. Supervised baselines
     cycle over labelled sequences only, FedGTCL and the pseudo-labelling baseline over all sequences.
  2. Equal labels: every method sees the same labelled subset (p_label). `full_labels=True` gives a supervised
     reference that sees every label.
  3. IID partitions: part="iid" splits all training-region sequences uniformly over the K clients (partition seed).
  4. Optional true-sparse attention for FedGTCL (FEDGTCL_SPARSE=1, sparse_model.py). It is numerically
     equivalent to the dense-masked version; it only changes speed.
Everything else (models, losses, server rules, evaluation region, metrics) is unchanged.

Original train_v3 docstring:

Families
  fedgtcl : FedGTCL encoder. options: lam (contrastive weight), mode ('joint' | 'twophase'),
            pre (pretraining rounds for twophase), gamma (bool), sparsify s (0 = dense),
            server ('adaptopt' | 'fedavg'), eta_s, bias_corr, lr, contrastive (bool)
  gcngru  : GCN-GRU on top-m adjacency, server 'fedavg' (server_lr) or 'adam' (eta_s, bias_corr), lr
  lstm    : graph-free LSTM, server as gcngru
Splits
  'val'   : confirmatory validation region of the partition file (study-1 protocol)
  'inner' : development split carved from the TRAINING region only: in every training segment the
            first 70% of windows train, the last 30% validate (sequences straddling the cut dropped);
            the confirmatory validation region is never touched.
"""
import argparse, copy, hashlib, json, math, os, pickle, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("model_mod", os.path.join(HERE, "04_model.py"))
model_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(model_mod)
import batched_model as bm
if os.environ.get("FEDGTCL_SPARSE", "0") == "1":
    import sparse_model as bm  # same interface, O(N*m) attention

TAU, RHO, BATCH = 0.5, 0.2, 16
DEV = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("FEDGTCL_GPU", "0") == "1" else "cpu")
torch.set_num_threads(int(os.environ.get("FEDGTCL_THREADS", "1")))


def reseed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_label_masks(client_train, p_label, seed):
    rng = random.Random(100_003 * seed + int(round(p_label * 1000)))
    out = []
    for tr in client_train:
        ids = sorted(e["anchor_idx"] for e in tr)
        n = min(len(ids), max(2, int(len(ids) * p_label)))
        out.append(frozenset(rng.sample(ids, n)))
    return out


def class_weights(labels):
    n = len(labels); n1 = sum(labels); n0 = n - n1
    return torch.tensor([n / max(1, 2 * n0), n / max(1, 2 * n1)], dtype=torch.float32)


# ------------------------------------------------------------------ splits
def inner_split(P, part):
    seg = P["segment"]; W = P["windows"]
    by_seg = {}
    for i, s in enumerate(seg):
        if P["region"][i] == "train":
            by_seg.setdefault(s, []).append(i)
    cut_of = {}
    for s, ids in by_seg.items():
        ids = sorted(ids); k = int(0.7 * len(ids))
        for j, w in enumerate(ids):
            cut_of[w] = "itrain" if j < k else "ival"
    tr, va = [], []
    for cl in P[f"{part}_train"]:
        a = [e for e in cl if all(cut_of.get(w) == "itrain" for w in e["win_ids"])]
        b = [e for e in cl if all(cut_of.get(w) == "ival" for w in e["win_ids"])]
        tr.append(a); va.append(b)
    train_ok = {w for w, v in cut_of.items() if v == "itrain"}
    return tr, va, train_ok


def iid_split(P):
    allseq = sorted((e for cl in P["noniid_train"] for e in cl), key=lambda e: e["anchor_idx"])
    rng = random.Random(7919 * int(P.get("partition_seed", 42)))
    rng.shuffle(allseq)
    K = len(P["noniid_train"])
    return [allseq[k::K] for k in range(K)]


def get_split(P, part, split):
    if part == "iid":
        assert split == "val"
        return iid_split(P), P["noniid_val"], set(P["train_region_window_ids"])
    if split == "val":
        return P[f"{part}_train"], P[f"{part}_val"], set(P["train_region_window_ids"])
    return inner_split(P, part)


# ------------------------------------------------------------------ views
class Views:
    def __init__(self, P, train_ok):
        self.W = P["windows"]; self.seg = P["segment"]; self.ok = train_ok

    def ids(self, entry, augment):
        ids = entry["win_ids"]
        if not augment:
            return ids
        opts = [ids]
        for d in (-1, 1):
            new = [i + d for i in ids]
            if all(j in self.ok and 0 <= j < len(self.W) and self.W[j] is not None for j in new) and \
                    len({self.seg[j] for j in new}) == 1:
                opts.append(new)
        return random.choice(opts)

    def view(self, entry, augment):
        ws = [self.W[i] for i in self.ids(entry, augment)]
        feats = [w["feats"] for w in ws]
        if augment:
            masks = []
            for w in ws:
                m = w["mask"] & ~(torch.rand(w["mask"].shape) < RHO)
                iso = ~m.any(dim=1)
                if iso.any():
                    m = m.clone(); m[iso, iso] = True
                masks.append(m)
        else:
            masks = [w["mask"] for w in ws]
        return feats, masks, frozenset(ws[-1]["node_ids"])


def nt_xent(proj, hosts, use_gamma):
    n = proj.size(0)
    sim = proj @ proj.t() / TAU
    idx = torch.arange(n); pos = idx ^ 1
    g = torch.ones(n, n)
    if use_gamma:
        for a in range(n):
            for j in range(a + 1, n):
                u = len(hosts[a] | hosts[j])
                g[a, j] = g[j, a] = 1.0 - (len(hosts[a] & hosts[j]) / u if u else 1.0)
    neg = torch.ones(n, n, dtype=torch.bool); neg[idx, idx] = False; neg[idx, pos] = False
    s_pos = sim[idx, pos]
    mx = sim.max(dim=1, keepdim=True).values.detach()
    den = torch.exp(s_pos - mx[:, 0]) + (g * torch.exp(sim - mx) * neg).sum(dim=1)
    return (-(s_pos - mx[:, 0] - torch.log(den + 1e-12))).mean()


# ------------------------------------------------------------------ baselines (batched)
class GCNGRU(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.g1 = nn.Linear(d, h); self.g2 = nn.Linear(h, h)
        self.gru = nn.GRU(h, h, batch_first=True); self.cls = nn.Linear(h, 2)

    def forward_batch(self, sf, sm):
        B, T = len(sf), len(sf[0])
        X, M, V = bm.pack([(f, m) for fs, ms in zip(sf, sm) for f, m in zip(fs, ms)])
        A = M.float() + torch.diag_embed(V.float())
        A = A / A.sum(dim=2, keepdim=True).clamp(min=1)
        h = torch.relu(A @ self.g1(X)); h = torch.relu(A @ self.g2(h))
        emb = (h * V.unsqueeze(-1)).sum(1) / V.sum(1, keepdim=True)
        out, _ = self.gru(emb.view(B, T, -1))
        return self.cls(out[:, -1])


class LSTMBase(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True); self.cls = nn.Linear(h, 2)

    def forward_batch(self, sf, sm):
        x = torch.stack([torch.stack([f.mean(0) for f in fs]) for fs in sf])
        out, _ = self.lstm(x)
        return self.cls(out[:, -1])


def flat(m):
    return torch.cat([p.data.reshape(-1) for p in m.parameters()]).clone()


def load_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


def topk(delta, s):
    k = max(1, int(s * delta.numel()))
    _, idx = torch.topk(delta.abs(), k)
    out = torch.zeros_like(delta); out[idx] = delta[idx]
    return out


def metrics(TP, FP, TN, FN):
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = 0.0 if den == 0 else (TP * TN - FP * FN) / den
    tpr = TP / (TP + FN) if TP + FN else float("nan"); tnr = TN / (TN + FP) if TN + FP else float("nan")
    prec = TP / (TP + FP) if TP + FP else 0.0; rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(MCC=mcc, BalancedAcc=(tpr + tnr) / 2, F1=100 * f1, TP=TP, FP=FP, TN=TN, FN=FN,
                degenerate=("always-benign" if TP + FP == 0 else "always-attack" if TN + FN == 0 else "no"))


def evaluate(model, vec, seqs, V, family):
    load_flat(model, vec); model.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for b0 in range(0, len(seqs), 64):
            batch = seqs[b0:b0 + 64]
            views = [V.view(e, False) for e in batch]
            sf = [v[0] for v in views]; sm = [v[1] for v in views]
            logits = bm.forward_batch(model, sf, sm)[2] if family == "fedgtcl" else model.forward_batch(sf, sm)
            pred = logits.argmax(1).tolist()
            for p, e in zip(pred, batch):
                y = e["label"]
                TP += p == 1 and y == 1; FP += p == 1 and y == 0; TN += p == 0 and y == 0; FN += p == 0 and y == 1
    return metrics(int(TP), int(FP), int(TN), int(FN))


def run(P, part, split, seed, p_label, cfg, R=10, E=3, steps=None, full_labels=False):
    ctr, cva, ok = get_split(P, part, split)
    V = Views(P, ok)
    masks = make_label_masks(ctr, 1.0 if full_labels else p_label, seed)
    reseed(seed)
    fam = cfg["family"]
    if fam == "fedgtcl":
        model = model_mod.FedGTCLEncoder(in_dim=P["IN_DIM"], hidden_dim=64, embed_dim=64, low_rank_dim=32,
                                         num_classes=2, proj_dim=32, temporal_kernel=3)
    elif fam == "gcngru":
        model = GCNGRU(P["IN_DIM"])
    else:
        model = LSTMBase(P["IN_DIM"])
    g = flat(model); m_t = torch.zeros_like(g); v_t = torch.zeros_like(g)
    lr = cfg.get("lr", 1e-3); server = cfg.get("server", "fedavg"); eta = cfg.get("eta_s", 0.02)
    bc = cfg.get("bias_corr", False); slr = cfg.get("server_lr", 1.0); s = cfg.get("s", 0.0)
    lam = cfg.get("lam", 1.0); mode = cfg.get("mode", "joint"); pre = cfg.get("pre", 0)
    use_con = cfg.get("contrastive", True); use_gamma = cfg.get("gamma", True); mu = cfg.get("mu", 0.0)
    ssl = cfg.get("ssl", False); tau_c = cfg.get("tau_conf", 0.9); lam_u = cfg.get("lam_u", 1.0)
    lab_only = cfg.get("labelled_only", False)  # study 4: sample mini-batches from labelled sequences only
    cw = [class_weights([e["label"] for e in tr if e["anchor_idx"] in masks[k]]) for k, tr in enumerate(ctr)]
    n_tr = [len(tr) for tr in ctr]
    curve = []
    for rnd in range(R):
        deltas, ws, rl = [], [], []
        pretrain = fam == "fedgtcl" and mode == "twophase" and rnd < pre
        for k, tr in enumerate(ctr):
            data = list(tr) if ((fam == "fedgtcl" and not lab_only) or ssl) else [e for e in tr if e["anchor_idx"] in masks[k]]
            if len(data) < (2 if fam == "fedgtcl" else 1):
                continue
            local = copy.deepcopy(model); load_flat(local, g); local.train()
            opt = torch.optim.Adam(local.parameters(), lr=lr)
            if steps is None:
                sched = []
                for ep in range(E):
                    random.shuffle(data)
                    sched += [((ep + 1) / E, data[b0:b0 + BATCH]) for b0 in range(0, len(data), BATCH)]
            else:
                sched, ptr, order = [], 0, []
                for st in range(steps):
                    if len(data) <= BATCH:
                        sched.append(((st + 1) / steps, random.sample(data, len(data)))); continue
                    if ptr + BATCH > len(order):
                        order = order[ptr:] + random.sample(data, len(data)); ptr = 0
                    sched.append(((st + 1) / steps, order[ptr:ptr + BATCH])); ptr += BATCH
            for beta_raw, batch in [(x[0], x[1]) for x in sched]:
                    beta = beta_raw if mode == "joint" else 1.0
                    if fam == "fedgtcl":
                        if len(batch) < 2:
                            continue
                        v1 = [V.view(e, True) for e in batch]; v2 = [V.view(e, True) for e in batch]
                        sf, sm, hs = [], [], []
                        for a_, b_ in zip(v1, v2):
                            sf += [a_[0], b_[0]]; sm += [a_[1], b_[1]]; hs += [a_[2], b_[2]]
                        _, proj, logits = bm.forward_batch(local, sf, sm)
                        lcon = nt_xent(proj, hs, use_gamma) if use_con else proj.sum() * 0.0
                        lab_i = [i for i, e in enumerate(batch) if e["anchor_idx"] in masks[k]]
                        if pretrain:
                            if not use_con:
                                continue
                            loss = lcon
                        else:
                            if not lab_i and (not use_con or lam == 0):
                                continue
                            lsup = (F.cross_entropy(logits[[2 * i for i in lab_i]],
                                                    torch.tensor([batch[i]["label"] for i in lab_i]), weight=cw[k])
                                    if lab_i else torch.tensor(0.0))
                            loss = lam * lcon + beta * lsup
                    elif ssl:
                        # FixMatch-style pseudo-labelling: supervised CE on labelled items; for unlabelled items,
                        # a pseudo-label from the clean view is kept if its confidence >= tau_conf and the model
                        # is trained on it through an augmented view (edge dropout + temporal jitter).
                        lab = [e for e in batch if e["anchor_idx"] in masks[k]]
                        unl = [e for e in batch if e["anchor_idx"] not in masks[k]]
                        loss = torch.tensor(0.0)
                        if lab:
                            vl = [V.view(e, False) for e in lab]
                            lg = local.forward_batch([v[0] for v in vl], [v[1] for v in vl])
                            loss = loss + F.cross_entropy(lg, torch.tensor([e["label"] for e in lab]), weight=cw[k])
                        if unl:
                            vc = [V.view(e, False) for e in unl]
                            with torch.no_grad():
                                pr = torch.softmax(local.forward_batch([v[0] for v in vc], [v[1] for v in vc]), 1)
                            conf, pl = pr.max(1); keep = conf >= tau_c
                            if keep.any():
                                va = [V.view(e, True) for e, kk in zip(unl, keep.tolist()) if kk]
                                lu = local.forward_batch([v[0] for v in va], [v[1] for v in va])
                                loss = loss + lam_u * F.cross_entropy(lu, pl[keep], weight=cw[k])
                        if not loss.requires_grad:
                            continue
                    else:
                        views = [V.view(e, False) for e in batch]
                        logits = local.forward_batch([v[0] for v in views], [v[1] for v in views])
                        loss = F.cross_entropy(logits, torch.tensor([e["label"] for e in batch]), weight=cw[k])
                    if mu > 0:
                        gp = 0; i0 = 0
                        prox = 0.0
                        for p_ in local.parameters():
                            n_ = p_.numel(); prox = prox + ((p_ - g[i0:i0 + n_].view_as(p_)) ** 2).sum(); i0 += n_
                        loss = loss + 0.5 * mu * prox
                    opt.zero_grad(); loss.backward(); opt.step(); rl.append(loss.item())
            d = flat(local) - g
            deltas.append(topk(d, s) if s > 0 else d); ws.append(n_tr[k])
        tot = sum(ws); agg = sum((w / tot) * d for w, d in zip(ws, deltas))
        if server in ("adaptopt", "adam"):
            m_t = 0.9 * m_t + 0.1 * agg; v_t = 0.99 * v_t + 0.01 * agg ** 2
            mh, vh = (m_t / (1 - 0.9 ** (rnd + 1)), v_t / (1 - 0.99 ** (rnd + 1))) if bc else (m_t, v_t)
            g = g + eta * mh / (torch.sqrt(vh) + 1e-8)
        else:
            g = g + slr * agg
        curve.append(float(np.mean(rl)) if rl else None)
    val = [e for cv in cva for e in cv]
    res = evaluate(model, g, val, V, fam)
    res.update(loss_curve=curve, n_val=len(val), val_attack=sum(e["label"] for e in val),
               param_sha=hashlib.sha256(g.numpy().tobytes()).hexdigest()[:16])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--partition", default="noniid")
    ap.add_argument("--split", default="inner"); ap.add_argument("--p_label", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--configs", required=True, help="json file: {name: cfg}")
    ap.add_argument("--names", nargs="*"); ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="dev")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "rerun_v2", "data"))
    ap.add_argument("--partition_file", default=None, help="explicit partition file (phase 2 / 3)")
    a = ap.parse_args()
    pf = a.partition_file or os.path.join(a.data_dir, f"partition_{a.dataset}_seed42.pkl")
    P = pickle.load(open(pf, "rb"))
    pseed = P.get("partition_seed", 42)
    cfgs = json.load(open(a.configs)); names = a.names or list(cfgs)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["dataset"], r.get("partition_seed", 42), r["partition"], r["split"], r["p_label"], r["R"], r["seed"], r["config"]))
    for seed in a.seeds:
        for nm in names:
            key = (a.dataset, pseed, a.partition, a.split, a.p_label, a.R, seed, nm)
            if key in done:
                continue
            t = time.time()
            r = run(P, a.partition, a.split, seed, a.p_label, cfgs[nm], R=a.R)
            row = dict(dataset=a.dataset, partition_seed=pseed, regime=P.get("regime"), partition=a.partition, split=a.split, p_label=a.p_label, R=a.R, seed=seed,
                       config=nm, cfg=cfgs[nm], tag=a.tag, secs=round(time.time() - t, 1), **r)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{a.dataset} {nm} seed={seed}: done ({row['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
