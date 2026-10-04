"""Efficiency numbers for Study 3: parameters, uplink payload per client per round, and per-sequence inference time
at the real host counts (dense-masked reference, batched dense, true sparse). Usage: python code/efficiency.py [out.json]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import torch
torch.set_num_threads(1)
import train_s3 as T, batched_model as dense, sparse_model as sparse

out = {}
for ds in ("wustl", "toniot", "edgeiiot"):
    P = pickle.load(open(os.path.join(ROOT, "partitions", f"partition_{ds}_p101.pkl"), "rb"))
    d = P["IN_DIM"]
    fg = T.model_mod.FedGTCLEncoder(in_dim=d, hidden_dim=64, embed_dim=64, low_rank_dim=32, num_classes=2, proj_dim=32, temporal_kernel=3).eval()
    gg = T.GCNGRU(d).eval(); ls = T.LSTMBase(d).eval()
    n = lambda m: sum(p.numel() for p in m.parameters())
    s = 0.3; k = int(s * n(fg))
    idx_bits = 16 if n(fg) < 65536 else 32
    row = dict(IN_DIM=d, params_FedGTCL=n(fg), params_GCNGRU=n(gg), params_LSTM=n(ls),
               uplink_KiB_FedGTCL_topS=round((k * 4 + k * idx_bits / 8) / 1024, 1), uplink_KiB_FedGTCL_dense=round(n(fg) * 4 / 1024, 1),
               uplink_KiB_GCNGRU=round(n(gg) * 4 / 1024, 1))
    seqs = [e for cl in P["noniid_val"] for e in cl][:32]; W = P["windows"]
    sf = [[W[i]["feats"] for i in e["win_ids"]] for e in seqs]; sm = [[W[i]["mask"] for i in e["win_ids"]] for e in seqs]
    with torch.no_grad():
        def t(fn, reps=20):
            fn(); t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps / len(seqs) * 1000
        row["ms_per_seq_reference"] = round(t(lambda: [fg(f, m) for f, m in zip(sf, sm)]), 3)
        row["ms_per_seq_batched_dense"] = round(t(lambda: dense.forward_batch(fg, sf, sm)), 3)
        row["ms_per_seq_sparse"] = round(t(lambda: sparse.forward_batch(fg, sf, sm)), 3)
        row["ms_per_seq_GCNGRU_batched"] = round(t(lambda: gg.forward_batch(sf, sm)), 3)
        row["median_hosts"] = int(torch.tensor([W[i]["feats"].shape[0] for e in seqs for i in e["win_ids"]]).float().median())
    out[ds] = row; print(ds, row, flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "efficiency.json"), "w"), indent=1)
