"""Part 2 benchmark: forward-pass time of the graph-attention layer, dense-masked vs true-sparse, as the host count
|V| grows (random graphs, top-m=8 neighbours, CPU, one thread). Also times one full T=5 sequence at the host counts
of the real datasets. Usage: python bench_sparse.py [out.json]"""
import sys, time, json, importlib.util, os
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "04_model.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import batched_model as dense, sparse_model as sparse
torch.manual_seed(42)
IN, M = 41, 8
enc = m.FedGTCLEncoder(in_dim=IN, hidden_dim=64, embed_dim=64, low_rank_dim=32, num_classes=2, proj_dim=32, temporal_kernel=3).eval()


def rand_graph(n):
    x = torch.rand(n, IN)
    w = torch.rand(n, n); w.fill_diagonal_(-1)
    top = torch.topk(w, min(M, n - 1), dim=1).indices
    mask = torch.zeros(n, n, dtype=torch.bool); mask.scatter_(1, top, True)
    return x, mask


def timeit(fn, reps):
    for _ in range(3):
        fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000


rows = []
with torch.no_grad():
    for n in (10, 20, 50, 100, 200, 400, 800, 1600):
        x, mask = rand_graph(n)
        X, Mk, V = dense.pack([(x, mask)])
        idx, ok = sparse.neighbour_index(Mk, V)            # built once, like the graph itself
        reps = 50 if n <= 200 else 10
        t_ref = timeit(lambda: enc.gat(x, mask), reps)                       # paper's per-window reference
        t_den = timeit(lambda: dense.encode_windows(enc, X, Mk, V), reps)     # batched dense-masked
        t_sp = timeit(lambda: sparse.encode_windows_sparse(enc, X, idx, ok, V), reps)
        diff = float((dense.encode_windows(enc, X, Mk, V) - sparse.encode_windows_sparse(enc, X, idx, ok, V)).abs().max())
        rows.append(dict(V=n, ref_dense_ms=round(t_ref, 4), batched_dense_ms=round(t_den, 4), sparse_ms=round(t_sp, 4), max_abs_diff=diff))
        print(rows[-1], flush=True)
base = rows[0]
print(f"growth |V| 10->800: reference dense {rows[6]['ref_dense_ms']/base['ref_dense_ms']:.0f}x, sparse {rows[6]['sparse_ms']/base['sparse_ms']:.0f}x")
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "bench_sparse.json", "w"), indent=1)
