# Study 6: budget-response curves and re-tuning at the corrected budget

Written before any Study 6 run and sealed with `code/seal.py` (`SEAL_STUDY6.sha256`). Nothing below changes after the
first run; later changes are dated addenda.

## Why
Studies 3-5 compared methods at two budgets only (30 and 1,000 local steps per client), with configurations chosen in
Study 2 under an epoch-based budget and not re-tuned. Two questions remain open and were listed as limitations:
(i) is 1,000 steps enough, or would the ranking of FedGTCL and FedProx change with longer training; (ii) would
re-tuning at the 1,000-step budget change it. Study 6 answers both on the three development datasets.

## Platform
All Study 6 runs are new and run on one Linux machine (cloud container, PyTorch 2.14.0 CPU, one thread per process).
No Study 6 result is compared run-by-run with a laptop result; every comparison is within Study 6.

## Part A: budget-response curves
- Datasets WUSTL-IIoT-2021, TON_IoT-Network, Edge-IIoTset; the Study 3 non-IID partitions 101-110 (files identical
  to the Study 3 seal); chronological validation regions; model seeds 201 and 202; p_L = 0.5.
- Methods and configurations exactly as Study 3 (`code/cfg_s6.json`): FG (FedGTCL), B1 (FedAvg-GCN-GRU),
  B2 (FedAvg-LSTM), B4 (FedProx, mu = 0.1), B5 (pseudo-labelling).
- Each run trains R = 150 rounds x 20 local Adam steps (3,000 steps per client). The global model is evaluated on the
  validation sequences after rounds 1, 2, 5, 10, 20, 50, 100 and 150 (20 ... 3,000 steps). Evaluation consumes no
  randomness, so the round-50 checkpoint is the Study 3 protocol (checked in a smoke test: identical parameters).
  FedGTCL's first five rounds are contrastive only, so its checkpoints before round 6 have an untrained classifier.

## Part B: re-tuning at 1,000 steps
- Grid: local learning rate in {3e-4, 1e-3, 3e-3} for FG, B4 and B1, everything else as Study 3.
- Selection: inner split of the training region (`train_s6.inner_split`, unchanged since Study 2: in each training
  segment the first 70% of windows train and the last 30% validate), partitions 101-110, seed 201, R = 50 x 20
  steps, p_L = 0.5. For each method the learning rate with the highest mean inner MCC over the 30 (dataset,
  partition) units is selected (ties: the Study 3 value 1e-3). The validation regions are not used for selection.
- Confirmation: each method with its selected learning rate on the validation regions, seeds 201-202, R = 50 x 20.
  When the selected value is 1e-3 the run is the round-50 checkpoint of Part A and is not repeated.

## Unit and confirmatory tests
Unit = mean MCC over the two seeds of one (dataset, partition) cell; n = 30. Two-sided Wilcoxon signed-rank
(zsplit); Holm over the three tests; alpha = 0.05; mean, median, positive count and bootstrap 95% interval reported.
- H6.1 Stability: D = [FG - B4] at 3,000 steps minus [FG - B4] at 1,000 steps. Pre-specified reading: the
  1,000-step comparison is called stable if the bootstrap 95% interval of D lies inside (-0.10, +0.10).
- H6.2 Long budget: [FG - B4] at 3,000 steps.
- H6.3 Re-tuned: [FG - B4] at 1,000 steps with the learning rates selected in Part B.

## Reported, not claim-bearing
Mean MCC and collapse rate of every method at every checkpoint, pooled and per dataset; the first checkpoint at
which mean [FG - B4] is negative; each baseline's change from 1,000 to 3,000 steps; the inner-split grid; the three
tests re-estimated with a linear mixed model with a random intercept per dataset (as a robustness check).

## Known limitations fixed in advance
Three development datasets only (UNSW-NB15 is not included: its Study 4 files are on the laptop that ran Studies
3-5); p_L = 0.5 only; one hyperparameter (learning rate) re-tuned, for three methods; validation curves are
computed on the confirmatory validation regions, so they describe the protocol, not a deployable stopping rule.
