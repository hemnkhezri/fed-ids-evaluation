# Pre-registered analysis plan (v2) -- FedGTCL re-evaluation

Written and hashed BEFORE any run of the fixed pipeline (22_fixed.py) was used to produce outcome data.
The only runs made with 22_fixed.py before this plan were the R=2 determinism checks, whose outcomes
are not used in any analysis below.

## Why this plan exists
The previous analysis chose its metric (F1 on the attack class) and its collapse definition (F1 > 0)
after seeing the data. F1 on the minority class rewards a degenerate "always-attack" predictor, and the
reported "significant benefit" of contrastive pretraining disappeared under MCC. A shared-list mutation
bug also gave compared arms different labeled subsets. Both are fixed in 22_fixed.py (guarantees G1-G3,
verified: bit-identical final weights under reversed run order).

## Fixed design (not to be changed after results are seen)
- Datasets: WUSTL-IIoT-2021, TON_IoT-Network, Edge-IIoTset (10% stratified); the corrected,
  leakage-free partitions already saved (one Dirichlet alpha=0.3 non-IID partition and one IID partition
  per dataset, partition seed 42). Partition variance is therefore NOT sampled; this is a stated limitation.
- Federation: K=4, R=10 rounds, E=3 local epochs, client lr 1e-3, batch 16, T=5.
- Seeds: 1..10 for every confirmatory cell. A seed fixes model initialisation, the labeled mask, and all
  training randomness. The labeled mask for a given (dataset, partition, p_label, seed) is identical for
  every arm, including baselines, which train on labeled sequences only.
- Arms:
  - A1 FedGTCL (contrastive with gamma, FedAdaptOpt, top-s sparsification s=0.3)
  - A2 FedGTCL without contrastive (identical code path, contrastive term zeroed)
  - A3 FedGTCL without gamma (plain NT-Xent)
  - A4 FedGTCL with plain FedAvg server update (no FedAdaptOpt; sparsification kept)
  - B1 FedAvg, graph-based GCN-GRU on true top-m adjacency
  - B2 FedAvg, LSTM (no graph)
  - B3 FedAdam server optimiser (same eta_s, beta1, beta2 as FedAdaptOpt), graph-based GCN-GRU

## Metric
Primary: Matthews correlation coefficient (MCC) on the pooled confusion matrix of all clients'
validation windows. A run is "discriminative" iff MCC > 0 (equivalently, informedness > 0).
Secondary, reported for completeness: balanced accuracy, F1 (attack class), accuracy.

## Confirmatory hypotheses (the only tests that may support a claim in the paper)
Setting: non-IID partition, p_label = 0.5 (the value in Table 4). For each dataset d:
- H1_d: MCC(A1) differs from MCC(B1)   -- does FedGTCL beat a properly trained FedAvg on the same labels?
- H2_d: MCC(A1) differs from MCC(A2)   -- does contrastive pretraining contribute?
- H3_d: MCC(A1) differs from MCC(A4)   -- does FedAdaptOpt contribute?
Test: two-sided Wilcoxon signed-rank test on the 10 per-seed paired MCC differences
(scipy.stats.wilcoxon, zero_method="zsplit"). Multiplicity: Holm correction over all 9 tests, alpha 0.05.

## Decision rules for the manuscript
- "FedGTCL outperforms FedAvg on dataset d" may be written only if H1_d is significant after Holm and
  the median paired difference is positive. Likewise for contrastive (H2) and FedAdaptOpt (H3).
- A result that is not significant is reported as "no detectable difference at n=10", never as a trend
  supporting the method.
- If no confirmatory test is significant in FedGTCL's favour, the paper's performance claim is withdrawn
  and the contribution is restated around what remains verified (the lightweight encoder, the
  communication accounting, and this leakage-free comparative evaluation itself).
- Every run is reported, including those unfavourable to FedGTCL.

## Exploratory analyses (reported, labelled exploratory, never claim-bearing)
- A1 vs A3 (gamma), A1 vs B2, A1 vs B3 in the same setting.
- Discriminative-rate comparisons with exact McNemar tests.
- p_label = 0.1, non-IID: A1 vs A2, seeds 1..10.
- IID partition, p_label = 0.5: A1, B1, B2, seeds 1..5.
