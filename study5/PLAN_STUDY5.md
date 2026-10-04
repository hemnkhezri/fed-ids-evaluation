# Study 5: how much the split protocol, the training budget and label asymmetry move the results

Written before any Study 5 run and sealed with `code/seal.py` (`SEAL_STUDY5.sha256`). Nothing below changes after
the first run; later changes are dated addenda.

## Why
The submitted pilot showed FedGTCL well ahead of its baselines. Studies 3-4 (sealed) showed that under leakage-free
splits and equal budgets it was not ahead of FedProx, and that a revised variant lost to FedProx on UNSW-NB15.
The pilot differed from Studies 3-4 in three ways at once: random splits of overlapping sequences, 30 optimiser steps,
and baselines that saw every label. Study 5 varies each of them on its own, with everything else fixed.

## Design
- Datasets: WUSTL-IIoT-2021, TON_IoT-Network, Edge-IIoTset (Study 3 partitions 101-110) and UNSW-NB15 (Study 4
  partitions 101-110, imported with their Study 4 hashes). Non-IID; model seeds 201 and 202.
- Split: `random` = the same client assignment as the partition file; inside each client, a random 80/20 split of
  its T = 5 stride-1 sequences from both regions (seed 1,000,003 x partition seed + client index); temporal
  augmentation may use any window. `chrono` = the chronological regions of Studies 3-4. Features keep the scaling
  fitted on the training region in both, so Study 5 isolates sequence-overlap leakage, not scaling leakage.
- Budget: `pilot` = R = 10 rounds x 3 local Adam steps; `full` = R = 50 rounds x 20 local steps. Mini-batches of
  up to 16 sequences as in Study 3.
- Configurations (`code/cfg_s5.json`, unchanged from Study 3): FG (FedGTCL), B1 (FedAvg-GCN-GRU), B2 (FedAvg-LSTM),
  B4 (FedProx, mu = 0.1), each with p_L = 0.5; B1F, B2F, B4F = the same baselines with every label.
- Reuse: the (chrono, full) cells of FG, B1, B2, B4 and B1F (= the Study 3/4 full-label reference) were run in
  Studies 3-4 on the same laptop, partitions, seeds and code, and are taken from `reused/`. All other cells are new:
  1,840 runs.

## Unit and tests
Unit = mean MCC over the two seeds of one (dataset, partition) cell; n = 40. Two-sided Wilcoxon signed-rank
(zsplit); Holm over the six tests; alpha = 0.05. Reported with mean, median, count of positive differences and a
bootstrap 95% interval.
- H1a-H1d: MCC(random, full) - MCC(chrono, full) for FG, B1, B2, B4 (50% labels).
- H2: [FG - B4](chrono, pilot) - [FG - B4](chrono, full).
- H3: [FG - B1F](random, pilot) - [FG - B1](chrono, full), i.e. the FedGTCL-FedAvg gap under the full pilot protocol
  minus the gap under the corrected protocol.

## Reported, not claim-bearing
All four split x budget conditions for all seven configurations, pooled and per dataset; degenerate-predictor
counts; method rankings per condition; the share of random-split validation sequences that share a window with a
training sequence.

## Known limitations fixed in advance
- The pilot's exact scripts (FedGTCL with R = 10 and E = 3 one-step epochs) are approximated by the Study 3 code at
  R = 10 and 3 steps per round; configurations are the Study 3 ones.
- Scaling leakage is not tested. Four datasets, ten partitions each, two seeds.
