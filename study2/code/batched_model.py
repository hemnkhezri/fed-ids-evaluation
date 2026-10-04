"""Batched, numerically equivalent forward pass for FedGTCLEncoder (04_model.py).

The original encoder processes one window graph at a time. This module pads the graphs of a
mini-batch to a common node count and evaluates the same parameters in one pass; padded nodes
are excluded from attention and readout. equivalence_check() verifies agreement with the
reference implementation to floating-point tolerance.
"""
import torch
import torch.nn.functional as F


def pack(graphs):
    """graphs: list of (feats [N,d], mask [N,N] bool). Returns X [G,Nmax,d], M [G,Nmax,Nmax], V [G,Nmax]."""
    G = len(graphs); nmax = max(f.size(0) for f, _ in graphs); d = graphs[0][0].size(1)
    X = torch.zeros(G, nmax, d); M = torch.zeros(G, nmax, nmax, dtype=torch.bool); V = torch.zeros(G, nmax, dtype=torch.bool)
    for g, (f, m) in enumerate(graphs):
        n = f.size(0); X[g, :n] = f; M[g, :n, :n] = m; V[g, :n] = True
    return X, M, V


def encode_windows(model, X, M, V):
    gat = model.gat
    Wh = gat.W2(gat.W1(X))                                     # [G,N,H]
    H = Wh.size(-1)
    a = gat.a.weight.view(-1)                                  # [2H]
    s_i = Wh @ a[:H]; s_j = Wh @ a[H:]                          # [G,N]
    e = F.leaky_relu(s_i.unsqueeze(2) + s_j.unsqueeze(1), 0.2)  # [G,N,N]
    M = M.clone()
    iso = (~M).all(dim=2) & V                                   # isolated real nodes -> self-loop
    idx = torch.arange(M.size(1))
    M[:, idx, idx] |= iso
    M[:, idx, idx] |= ~V                                        # padded rows: self-loop (discarded later)
    e = e.masked_fill(~M, float("-inf"))
    alpha = torch.softmax(e, dim=2)
    h = F.elu(alpha @ Wh)                                       # [G,N,H]
    ro = model.readout
    scores = (ro.key(h) @ ro.query) / (h.size(-1) ** 0.5)      # [G,N]
    scores = scores.masked_fill(~V, float("-inf"))
    w = torch.softmax(scores, dim=1)
    return (w.unsqueeze(-1) * h).sum(dim=1)                    # [G,H]


def forward_batch(model, seqs_feats, seqs_masks):
    """seqs_*: list (B) of lists (T) of window tensors. Returns z [B,H], proj [B,P], logits [B,C]."""
    B = len(seqs_feats); T = len(seqs_feats[0])
    graphs = [(f, m) for fs, ms in zip(seqs_feats, seqs_masks) for f, m in zip(fs, ms)]
    X, M, V = pack(graphs)
    g = encode_windows(model, X, M, V).view(B, T, -1)          # [B,T,H]
    zs = model.temporal(g.transpose(1, 2))                     # [B,H,T]
    z = zs[:, :, -1]
    return z, model.proj_head(z), model.cls_head(z)


def equivalence_check(model, seqs_feats, seqs_masks, tol=1e-5):
    with torch.no_grad():
        zb, pb, lb = forward_batch(model, seqs_feats, seqs_masks)
        for i, (fs, ms) in enumerate(zip(seqs_feats, seqs_masks)):
            z, p, l = model(fs, ms)
            assert torch.allclose(z, zb[i], atol=tol), (i, (z - zb[i]).abs().max())
            assert torch.allclose(l, lb[i], atol=tol)
    return True
