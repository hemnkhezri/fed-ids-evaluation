"""POST HOC (not in any sealed plan), added after a separate review of the code against the plans.
(a) repeat B4 on the ten WUSTL-IIoT-2021 partitions on the Linux instance (E=3, as pre-registered), so that the
    second-platform test of H_A pairs runs from one platform;
(b) step-matched check: B1-B4 iterate only over labelled sequences, so at p_label=0.1 they take 0.25x (WUSTL) and
    0.16-0.17x (TON_IoT, Edge-IIoTset) of FedGTCL's local optimiser steps per epoch. Re-run with E=12 (WUSTL) and
    E=18 (TON_IoT, Edge-IIoTset), which equalises local steps on average; E=30 (1.7-2.5x FedGTCL's steps) was also run
    as a sensitivity check. Everything else as pre-registered (R=30, p_label=0.1, seeds 201-202, partitions 101-110,
    configurations selected at E=3). FedGTCL itself is not re-run here.
Uses train_v3.run unchanged. Usage: python matched_steps.py <dataset> <E> <out.jsonl> <cfg names...>
(out path is relative to the rerun_v3 folder)"""
import sys, os, json, time, pickle
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.chdir(os.path.join(HERE, ".."))
import train_v3 as T
ds, E, out, names = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
cfgs = json.load(open("cfg_phase3b.json"))
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l); done.add((r["partition_seed"], r["seed"], r["config"]))
for ps in range(101, 111):
    P = pickle.load(open(f"phase2_data/partition_{ds}_p{ps}.pkl", "rb"))
    for seed in (201, 202):
        for nm in names:
            if (ps, seed, nm) in done:
                continue
            t = time.time()
            r = T.run(P, "noniid", "val", seed, 0.1, cfgs[nm], R=30, E=E)
            row = dict(dataset=ds, partition_seed=ps, regime=P.get("regime"), partition="noniid", split="val",
                       p_label=0.1, R=30, E=E, seed=seed, config=nm, cfg=cfgs[nm], tag=f"posthoc_E{E}",
                       platform="linux", secs=round(time.time() - t, 1), **r)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(ds, ps, seed, nm, E, row["secs"], round(r["MCC"], 3), flush=True)
