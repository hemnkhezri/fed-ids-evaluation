# Study 2, phase 1: development protocol (not claim-bearing)

Purpose: repair the design weaknesses found after study 1 (contrastive loss dominating the supervised term;
unstable sign-like server step), and give every method an equal, documented tuning budget, before a new
pre-registered confirmatory study (phase 2) and a held-out external test (phase 3, CICAPT-IIoT2024, not yet seen).

Data: the three study-1 datasets and partitions (partition seed 42, non-IID). Only the TRAINING region is used:
each training segment is split chronologically 70/30 into inner-train / inner-validation (sequences straddling the
cut are dropped). The study-1 validation region is not used in this phase.

Budget: 12 configurations per method family (cfg_dev.json):
- FG  : FedGTCL variants (contrastive weight lam in {1, 0.3, 0.1}; server eta_s / bias correction in
        {(0.02, off), (0.005, on), (0.002, on)}; plus a two-phase schedule: 5 contrastive-only rounds, then lam=0.1)
- B1  : FedAvg GCN-GRU   (client lr {1e-3, 3e-3, 1e-2} x server lr {1, 2, 4, 8})
- B2  : FedAvg LSTM      (same grid)
- B3  : FedAdam GCN-GRU  (client lr x {(0.02,off),(0.01,on),(0.005,on),(0.002,on)})
- B4  : FedProx GCN-GRU  (client lr x mu {0.001, 0.01, 0.1, 1})
Every configuration: K=4, R=30, E=3, batch 16, seeds 101-105, p_label in {0.1, 0.5}, three datasets
(30 runs per configuration).

Selection rule (fixed now): for each family, the configuration with the highest mean inner-validation MCC over
its 30 runs; ties broken by listing order. The five selected configurations are frozen for phases 2 and 3.
Phase-1 results are reported as development results only.
