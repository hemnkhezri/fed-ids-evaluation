# Study 3: leakage-free, equal-budget re-evaluation of FedGTCL (analysis plan)

Written before any Study 3 run. `seal.py` records the SHA-256 of this file, the code, the configurations and the
partition files in `SEAL_STUDY3.sha256`, with the time of sealing. Nothing below is changed after the first run;
any later change is added as a dated addendum.

## Why this study
The pilot in the submitted manuscript has three weaknesses a reviewer can raise:
1. T=5 sequences overlap by four windows and were split at random into training and validation, so validation
   sequences share windows with training sequences.
2. Each client took 3 optimizer steps per round (30 in total), which leaves every method under-trained.
3. The FedAvg baselines saw all labels while FedGTCL saw half.

Study 3 removes all three and asks whether FedGTCL's advantage survives.

## Data and splits (fixed)
- Datasets: WUSTL-IIoT-2021, TON_IoT-Network, Edge-IIoTset (10% subsample), as window graphs of the main paper
  (host-level nodes, top-8 neighbours, T=5 stride-1 sequences; 60-s windows for WUSTL-IIoT-2021 and Edge-IIoTset,
  300-row chunks inside each single-scenario block for TON_IoT-Network).
- Chronological regions: WUSTL-IIoT-2021 and Edge-IIoTset windows are cut into five equal contiguous blocks of the
  window timeline; blocks 2 and 4 form the validation region, blocks 1, 3 and 5 the training region. TON_IoT-Network:
  the first 80% of each scenario block's windows are training, the rest validation. Sequences never cross a region
  segment, and a programmatic check confirms that no window is shared between any training and any validation
  sequence. Feature normalisation (Eq. (1)) is fitted on training-region rows only.
- Partitions: the ten non-IID Dirichlet (alpha = 0.3) partitions 101-110 of the training region per dataset
  (files in `partitions/`, built before this plan). The IID arm splits the same training sequences uniformly over
  the K=4 clients with the partition seed.
- Validation: the whole validation region of the dataset, pooled over clients; identical for every partition.

## Methods (fixed configurations, `code/cfg_s3.json`)
- FG: FedGTCL (two-phase: 5 contrastive rounds, then contrastive weight 0.1 plus supervised loss; FedAdaptOpt,
  eta_s = 0.02, top-s with s = 0.3). Configuration selected in Study 2 on an inner split of the training region.
- B1: FedAvg GCN-GRU. B2: FedAvg LSTM. B4: FedProx GCN-GRU (mu = 0.1).
- B5: FedAvg GCN-GRU with FixMatch-style pseudo-labelling (threshold 0.9), a semi-supervised baseline that, like
  FedGTCL, uses unlabelled sequences.
- REF: FedAvg GCN-GRU with every label (reference only, not a comparator).
- Ablations: FG_noCon (no contrastive loss) and FG_noOpt (plain FedAvg server, no sparsification).

## Training budget (fixed, equal for all methods)
K = 4 clients, R = 50 rounds, 20 local Adam steps per client per round, mini-batches of up to 16 sequences drawn by
cycling through the client's usable data (labelled sequences for supervised baselines, all sequences for FG and B5).
Label fraction p_L in {0.1, 0.5}; every method sees the same labelled subset. Model seeds 201 and 202. The model
after the last round is evaluated; there is no early stopping and no checkpoint selection.

## Outcome and unit
MCC on the validation region. Unit of analysis: the mean MCC over the two model seeds of one
(dataset, partition, regime, p_L, method) cell. Pooled tests use the 30 partitions of the three datasets.

## Confirmatory family (four comparisons)
FG versus B4 and FG versus B5, non-IID, at p_L = 0.1 and at p_L = 0.5, pooled over datasets (n = 30 each).
- Superiority: two-sided Wilcoxon signed-rank test (zero_method = "zsplit"), Holm correction over the four
  comparisons, alpha = 0.05.
- Non-inferiority: margin Delta = 0.05 MCC, one-sided Wilcoxon signed-rank test of d + Delta > 0 (d = FG - comparator),
  Holm over the four comparisons, alpha = 0.025. Delta is the smallest difference we treat as practically relevant.
- Reported with each test: mean and median difference and a bootstrap 95% interval of the mean difference.

## Decision rules (applied mechanically by `code/analyze_s3.py`)
- Outcome A: at least one superiority test significant with a positive median. The paper may claim superiority
  only in those settings.
- Outcome B: no superiority, but non-inferiority in all four comparisons. The paper claims comparable detection at a
  lower parameter and communication cost.
- Outcome B-partial: non-inferiority in some comparisons only. Claims are limited to those settings.
- Outcome C: neither. The paper reports that FedGTCL is not competitive under this protocol, and the contribution
  must come from elsewhere (efficiency, analysis) or from an improved method.
- A significant disadvantage (superiority test significant with a negative median) is reported with the same
  prominence as an advantage.

## Exploratory (reported, not claim-bearing)
Per-dataset results; the IID arm; FG versus B1 and B2; distance to the full-label reference; ablations; rate of
degenerate (constant) predictors with counts; efficiency (`code/efficiency.py`) and the dense-versus-sparse attention
benchmark (`code/bench_sparse.py`).

## Known limitations fixed in advance
- The validation region was already used for evaluation in Studies 1-2 (never for selecting configurations).
- Configurations come from Study 2's selection; no new tuning is done.
- Two seeds per partition; datasets are pooled in the confirmatory tests.
