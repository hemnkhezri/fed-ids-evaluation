"""True-sparse forward pass for FedGTCLEncoder (same interface as batched_model.py).

Each node attends only to its retained neighbours (top-m, plus a self-loop for isolated or padded nodes), so the
attention cost is O(N*k) instead of the O(N^2) of the dense-masked version, where k <= m+1. The adjacency is
converted once into a neighbour-index tensor; in deployment this list is produced directly by graph construction.
equivalence_check() verifies agreement with the dense-masked reference (batched_model.py) to float tolerance.
"""
import torch
import torch.nn.functional as F
from batched_model import pack  # identical packing


def neighbour_index(M, V):
    """M [G,N,N] bool, V [G,N] bool -> idx [G,N,k] long, ok [G,N,k] bool (self-loop for isolated/padded rows)."""
    G, N, _ = M.shape
    M = M.clone()
    iso = (~M).all(dim=2) | ~V
    ar = torch.arange(N)
    M[:, ar, ar] |= iso
    k = int(M.sum(dim=2).max().item())
    vals, idx = torch.topk(M.float(), k, dim=2)          # retained neighbours first
    return idx, vals.bool()


def encode_windows_sparse(model, X, idx, ok, V):
    gat = model.gat
    Wh = gat.W2(gat.W1(X))                                # [G,N,H]
    G, N, H = Wh.shape
    a = gat.a.weight.view(-1)
    s_i = Wh @ a[:H]; s_j = Wh @ a[H:]                     # [G,N]
    s_nb = torch.gather(s_j, 1, idx.reshape(G, -1)).view(G, N, -1)   # scores of each node's neighbours
    e = F.leaky_relu(s_i.unsqueeze(2) + s_nb, 0.2).masked_fill(~ok, float("-inf"))   # [G,N,k]
    alpha = torch.softmax(e, dim=2)
    Wh_nb = torch.gather(Wh, 1, idx.reshape(G, -1, 1).expand(G, idx.size(1) * idx.size(2), H)).view(G, N, -1, H)
    h = F.elu((alpha.unsqueeze(-1) * Wh_nb).sum(dim=2))   # [G,N,H]
    ro = model.readout
    scores = (ro.key(h) @ ro.query) / (H ** 0.5)
    scores = scores.masked_fill(~V, float("-inf"))
    w = torch.softmax(scores, dim=1)
    return (w.unsqueeze(-1) * h).sum(dim=1)


def forward_batch(model, seqs_feats, seqs_masks):
    B = len(seqs_feats); T = len(seqs_feats[0])
    graphs = [(f, m) for fs, ms in zip(seqs_feats, seqs_masks) for f, m in zip(fs, ms)]
    X, M, V = pack(graphs)
    idx, ok = neighbour_index(M, V)
    g = encode_windows_sparse(model, X, idx, ok, V).view(B, T, -1)
    zs = model.temporal(g.transpose(1, 2))
    z = zs[:, :, -1]
    return z, model.proj_head(z), model.cls_head(z)


def equivalence_check(model, seqs_feats, seqs_masks, tol=1e-5):
    import batched_model as dense
    with torch.no_grad():
        zd, pd, ld = dense.forward_batch(model, seqs_feats, seqs_masks)
        zs, ps, ls = forward_batch(model, seqs_feats, seqs_masks)
    assert torch.allclose(zd, zs, atol=tol), (zd - zs).abs().max()
    assert torch.allclose(ld, ls, atol=tol), (ld - ls).abs().max()
    return float((zd - zs).abs().max())
