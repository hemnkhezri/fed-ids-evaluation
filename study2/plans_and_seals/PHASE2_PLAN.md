# Study 2, phases 2-3: pre-registered confirmatory plan

Written after phase 1 (development) and before any phase-2 or phase-3 run. The plan is sealed with SHA-256
(PHASE2_SEAL.sha256).

## Background from phase 1 (development only, not evidence)
The phase-1 selection rule picked one configuration per method (cfg_phase2.json). Mean inner-validation MCC
over 30 runs was:
- FedProx GCN-GRU (B4): 0.599
- FedAvg GCN-GRU (B1): 0.580
- FedAdam GCN-GRU (B3): 0.576
- FedGTCL (FG): 0.569
- FedAvg LSTM (B2): 0.535

FedGTCL was ahead only on WUSTL-IIoT-2021, where every tuned baseline had MCC = 0 at both label fractions.
FedGTCL was behind on TON_IoT-Network and Edge-IIoTset.

The hypothesis carried forward is therefore narrow: **when labeled attack examples are extremely scarce and
concentrated on few clients, FedGTCL keeps discriminating where tuned supervised federated baselines do not.**

## Frozen elements
- **Configurations:** cfg_phase2.json (FG, B1, B2, B3, B4), unchanged from the phase-1 selection.
- **Training:** K = 4, R = 30, E = 3, batch 16.
- **Code:** train_v3.py. The only change from the version sealed for phase 1 is an option that loads an
  explicit partition file and records its seed.
- **Partitions:** new Dirichlet(alpha = 0.3) non-IID partitions, partition seeds 101-110, built by
  make_phase2_partitions.py.
  - WUSTL-IIoT-2021, TON_IoT-Network and Edge-IIoTset: built from the existing window files.
  - CICAPT-IIoT2024 (held-out): built from heldout/partition_cicapt_seed42.pkl. No model has ever been run on
    this dataset.
- **Model seeds:** 201 and 202.
- **Label fraction (confirmatory):** p_label = 0.1.
- **Evaluation:** the validation region of each dataset (study-1 protocol). Configurations were never
  selected on it. It was used for evaluation in study 1, which is disclosed as a limitation. CICAPT's
  validation region has never been evaluated.
- **Metric:** MCC on the pooled validation windows after the final round.

## Unit of analysis
Mean MCC over the two model seeds for one (dataset, partition seed, method) cell.

The pooled validation set is identical across partitions of a dataset. Partitions vary only the training
distribution across clients.

## Confirmatory hypotheses (comparator: B4, the best baseline in phase 1)
| Hypothesis | Comparison | Data | n |
|---|---|---|---|
| H_A | FG vs B4 | WUSTL-IIoT-2021 | 10 partitions |
| H_B | FG vs B4 | TON_IoT-Network and Edge-IIoTset pooled | 20 partitions |
| H_C | FG vs B4 | CICAPT-IIoT2024 (independent held-out test) | 10 partitions |

- **H_A** is the replication of the phase-1 advantage on new partitions and seeds.
- **H_B** is the setting where phase 1 favoured the baselines. It is tested two-sided so that a significant
  disadvantage is also reported.

**Test:** two-sided Wilcoxon signed-rank test on paired partition-level values (zero_method = "zsplit").
Holm correction over the three tests, alpha = 0.05.

## Decision rules
- The paper may state that FedGTCL outperforms tuned federated baselines in a setting only if the
  corresponding test is significant after Holm correction and the median difference is positive.
- H_C is the claim that generalises beyond the development datasets.
- A significant H_A with a non-significant H_C must be reported as a replication on the development dataset
  only.
- Negative or non-significant outcomes are reported in full, with the same prominence.

## Exploratory (reported, not claim-bearing)
- FG vs B1, B2 and B3 on every dataset.
- The same comparisons at p_label = 0.5, if run.
- Regime moderator: partitions classified by make_phase2_partitions.py. "Extreme" means at most one client
  holds 10 or more attack training sequences.
- Discriminative rates (MCC > 0) with Wilson intervals.
- Cross-platform agreement when runs exist on both machines.
