"""
Confirmatory runner for ANALYSIS_PLAN_v2 (+ addendum), re-implemented from the written
specification after the 2026-09-26 environment reset.

Arms (plan v2):
  A1 FedGTCL: structure-aware NT-Xent (gamma), FedAdaptOpt server, top-s (s=0.3)
  A2 A1 with the contrastive term zeroed (identical code path)
  A3 A1 with plain NT-Xent (gamma == 1)
  A4 A1 with a plain FedAvg server update (sparsification kept)
  B1 FedAvg, graph-based GCN-GRU on the true top-m adjacency
  B2 FedAvg, LSTM (no graph)
  B3 FedAdam server (eta_s, beta1, beta2, eps as FedAdaptOpt), graph GCN-GRU
Every arm: K=4, R=10, E=3, client Adam lr 1e-3, batch 16, every local mini-batch,
per-client class weights from that client's labelled training sequences, and the
labelled mask fixed once per (dataset, partition, p_label, seed) and shared by all arms.
Baselines train on labelled sequences only.

Usage: python3 train.py --dataset wustl --seeds 1 2 --arms A1 B1 --budget 250
Results are appended (one JSON line per run) to results_v2r.jsonl; completed cells skipped.
"""
import argparse, copy, hashlib, json, math, os, pickle, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("FEDGTCL_RESULTS", os.path.join(HERE, "results_v2r.jsonl"))
PLAN_TAG = "b5d26257+addendum"
torch.set_num_threads(1)

spec = importlib.util.spec_from_file_location("model_mod", os.path.join(HERE, "04_model.py"))
model_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(model_mod)

R_DEF, E_DEF, LR, BATCH = 10, 3, 1e-3, 16
ETA_S, B1_, B2_, EPS, S_TOP = 0.02, 0.9, 0.99, 1e-8, 0.3
TAU, RHO = 0.5, 0.2


def reseed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_label_masks(client_train, p_label, seed):
    rng = random.Random(100_003 * seed + int(round(p_label * 1000)))
    masks = []
    for tr in client_train:
        ids = sorted(e["anchor_idx"] for e in tr)
        n = min(len(ids), max(2, int(len(ids) * p_label)))
        masks.append(frozenset(rng.sample(ids, n)))
    return masks


def client_class_weights(labels):
    n = len(labels); n1 = sum(labels); n0 = n - n1
    return torch.tensor([n / max(1, 2 * n0), n / max(1, 2 * n1)], dtype=torch.float32)


# ------------------------------------------------------------------ data views
class Views:
    def __init__(self, P):
        self.W = P["windows"]; self.seg = P["segment"]
        self.train_ok = set(P["train_region_window_ids"])

    def jitter_ids(self, ids):
        """shift start by delta in {-1,0,+1}, only to positions whose windows are all
        training-region windows of the same segment (Section 4.2 safeguard)."""
        ok = []
        for d in (-1, 0, 1):
            new = [i + d for i in ids]
            if d == 0 or (all(j in self.train_ok and self.W[j] is not None for j in new)
                          and len({self.seg[j] for j in new}) == 1):
                ok.append(new)
        return random.choice(ok)

    def view(self, entry, augment):
        ids = self.jitter_ids(entry["win_ids"]) if augment else entry["win_ids"]
        ws = [self.W[i] for i in ids]
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


# ------------------------------------------------------------------ losses
def nt_xent(proj, hosts, use_gamma):
    """proj: [2B, d] normalised, interleaved (a0,b0,a1,b1,...). Denominator:
    exp(s_pos) + sum_{j not in {a,pos}} gamma(a,j) exp(s_aj)  (Eq. 5 / 5a)."""
    n = proj.size(0)
    sim = proj @ proj.t() / TAU
    pos = torch.arange(n) ^ 1
    if use_gamma:
        g = torch.ones(n, n)
        for a in range(n):
            for j in range(a + 1, n):
                inter = len(hosts[a] & hosts[j]); uni = len(hosts[a] | hosts[j])
                jac = 1.0 if uni == 0 else inter / uni
                g[a, j] = g[j, a] = 1.0 - jac
    else:
        g = torch.ones(n, n)
    neg = torch.ones(n, n, dtype=torch.bool)
    neg[torch.arange(n), torch.arange(n)] = False
    neg[torch.arange(n), pos] = False
    s_pos = sim[torch.arange(n), pos]
    # log-sum-exp with weights, numerically stable
    mx = sim.max(dim=1, keepdim=True).values.detach()
    denom = torch.exp(s_pos - mx.squeeze(1)) + (g * torch.exp(sim - mx) * neg).sum(dim=1)
    return (-(s_pos - mx.squeeze(1) - torch.log(denom + 1e-12))).mean()


# ------------------------------------------------------------------ baselines
class GCNGRU(nn.Module):
    """two GCN layers propagating over A_hat = row-normalised (top-m adjacency + I),
    mean readout, GRU over the T windows, linear classifier."""
    def __init__(self, d, h=64):
        super().__init__()
        self.g1 = nn.Linear(d, h); self.g2 = nn.Linear(h, h)
        self.gru = nn.GRU(h, h, batch_first=True); self.cls = nn.Linear(h, 2)

    def forward(self, feats, masks):
        emb = []
        for x, m in zip(feats, masks):
            A = m.float() + torch.eye(m.size(0))
            A = A / A.sum(dim=1, keepdim=True)
            h = torch.relu(A @ self.g1(x)); h = torch.relu(A @ self.g2(h))
            emb.append(h.mean(dim=0))
        out, _ = self.gru(torch.stack(emb).unsqueeze(0))
        return self.cls(out[0, -1])


class LSTMBase(nn.Module):
    """graph-free: per-window mean of node features -> LSTM -> linear."""
    def __init__(self, d, h=64):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True); self.cls = nn.Linear(h, 2)

    def forward(self, feats, masks):
        x = torch.stack([f.mean(dim=0) for f in feats]).unsqueeze(0)
        out, _ = self.lstm(x)
        return self.cls(out[0, -1])


# ------------------------------------------------------------------ fed utils
def flat(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()]).clone()


def load_flat(model, v):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


def topk(delta, s):
    k = max(1, int(s * delta.numel()))
    _, idx = torch.topk(delta.abs(), k)
    out = torch.zeros_like(delta); out[idx] = delta[idx]
    return out


def evaluate(model, vec, seqs, V, fedgtcl):
    load_flat(model, vec); model.eval()
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for e in seqs:
            f, m, _ = V.view(e, augment=False)
            logits = model(f, m)[2] if fedgtcl else model(f, m)
            p = int(logits.argmax()); y = e["label"]
            TP += p == 1 and y == 1; FP += p == 1 and y == 0
            TN += p == 0 and y == 0; FN += p == 0 and y == 1
    return metrics(TP, FP, TN, FN)


def metrics(TP, FP, TN, FN):
    n = TP + FP + TN + FN
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = 0.0 if den == 0 else (TP * TN - FP * FN) / den
    tpr = TP / (TP + FN) if TP + FN else float("nan")
    tnr = TN / (TN + FP) if TN + FP else float("nan")
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(MCC=mcc, BalancedAcc=(tpr + tnr) / 2, F1=100 * f1, Accuracy=100 * (TP + TN) / max(1, n),
                TP=int(TP), FP=int(FP), TN=int(TN), FN=int(FN),
                degenerate=("always-benign" if TP + FP == 0 else "always-attack" if TN + FN == 0 else "no"))


# ------------------------------------------------------------------ training
def run(P, part, seed, p_label, arm, R=R_DEF, E=E_DEF):
    ctrain = P[f"{part}_train"]; cval = P[f"{part}_val"]
    V = Views(P)
    masks = make_label_masks(ctrain, p_label, seed)
    reseed(seed)
    fedgtcl = arm.startswith("A")
    if fedgtcl:
        model = model_mod.FedGTCLEncoder(in_dim=P["IN_DIM"], hidden_dim=64, embed_dim=64, low_rank_dim=32,
                                         num_classes=2, proj_dim=32, temporal_kernel=3)
    elif arm == "B2":
        model = LSTMBase(P["IN_DIM"])
    else:
        model = GCNGRU(P["IN_DIM"])
    g = flat(model); m_t = torch.zeros_like(g); v_t = torch.zeros_like(g)
    server = {"A4": "fedavg", "B1": "fedavg", "B2": "fedavg"}.get(arm, "adam")
    sparsify = fedgtcl
    use_con = arm != "A2"; use_gamma = arm != "A3"
    cw = []
    for k, tr in enumerate(ctrain):
        cw.append(client_class_weights([e["label"] for e in tr if e["anchor_idx"] in masks[k]]))
    n_tr = [len(tr) for tr in ctrain]
    loss_log = []
    for rnd in range(R):
        deltas, ws, rl = [], [], []
        for k, tr in enumerate(ctrain):
            data = list(tr) if fedgtcl else [e for e in tr if e["anchor_idx"] in masks[k]]
            if len(data) < (2 if fedgtcl else 1):
                continue
            local = copy.deepcopy(model); load_flat(local, g); local.train()
            opt = torch.optim.Adam(local.parameters(), lr=LR)
            for ep in range(E):
                beta = (ep + 1) / E
                random.shuffle(data)
                for b0 in range(0, len(data), BATCH):
                    batch = data[b0:b0 + BATCH]
                    if fedgtcl:
                        if len(batch) < 2:
                            continue
                        projs, hosts, lg, lab = [], [], [], []
                        for e in batch:
                            for vi in range(2):
                                f, m, h = V.view(e, augment=True)
                                _, pr, lo = local(f, m)
                                projs.append(pr); hosts.append(h)
                                if vi == 0 and e["anchor_idx"] in masks[k]:
                                    lg.append(lo); lab.append(e["label"])
                        lcon = nt_xent(torch.stack(projs), hosts, use_gamma)
                        if not use_con:
                            lcon = lcon * 0.0
                        lsup = (F.cross_entropy(torch.stack(lg), torch.tensor(lab), weight=cw[k])
                                if lg else torch.tensor(0.0))
                        loss = lcon + beta * lsup
                        if not use_con and not lg:
                            continue  # A2: no labelled item in batch -> zero gradient; skip the
                                      # Adam step so stale momentum cannot move the weights
                    else:
                        lg = []
                        for e in batch:
                            f, m, _ = V.view(e, augment=False)
                            lg.append(local(f, m))
                        loss = F.cross_entropy(torch.stack(lg), torch.tensor([e["label"] for e in batch]),
                                               weight=cw[k])
                    if loss.requires_grad:
                        opt.zero_grad(); loss.backward(); opt.step()
                    rl.append(loss.item())
            d = flat(local) - g
            deltas.append(topk(d, S_TOP) if sparsify else d); ws.append(n_tr[k])
        tot = sum(ws)
        agg = sum((w / tot) * d for w, d in zip(ws, deltas))
        if server == "adam":
            m_t = B1_ * m_t + (1 - B1_) * agg
            v_t = B2_ * v_t + (1 - B2_) * agg ** 2
            g = g + ETA_S * m_t / (torch.sqrt(v_t) + EPS)
        else:
            g = g + agg
        loss_log.append(float(np.mean(rl)) if rl else None)
    val = [e for cv in cval for e in cv]
    res = evaluate(model, g, val, V, fedgtcl)
    res.update(loss_curve=loss_log, label_mask_sizes=[len(m) for m in masks],
               param_sha=hashlib.sha256(g.numpy().tobytes()).hexdigest()[:16])
    return res


def done_keys():
    keys = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            r = json.loads(line)
            keys.add((r["dataset"], r["partition"], r["p_label"], r["seed"], r["arm"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--partition", default="noniid")
    ap.add_argument("--p_label", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A3", "A4", "B1", "B2", "B3"])
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--tag", default="confirmatory")
    a = ap.parse_args()
    P = pickle.load(open(os.path.join(HERE, "data", f"partition_{a.dataset}_seed42.pkl"), "rb"))
    t0 = time.time(); done = done_keys()
    for seed in a.seeds:
        for arm in a.arms:
            key = (a.dataset, a.partition, a.p_label, seed, arm)
            if key in done:
                continue
            if time.time() - t0 > a.budget:
                print("budget reached"); return
            t1 = time.time()
            r = run(P, a.partition, seed, a.p_label, arm)
            row = dict(dataset=a.dataset, partition=a.partition, p_label=a.p_label, seed=seed, arm=arm,
                       tag=a.tag, plan=PLAN_TAG, secs=round(time.time() - t1, 1), **r)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{a.dataset} {a.partition} p={a.p_label} seed={seed} {arm}: MCC={r['MCC']:+.3f} "
                  f"TP={r['TP']} FP={r['FP']} TN={r['TN']} FN={r['FN']} ({row['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
