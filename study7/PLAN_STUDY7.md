# Study 7: random versus temporal splits for per-record federated classifiers

Written before any Study 7 data were prepared or any model was run, and sealed with `code/seal.py`
(`SEAL_STUDY7.sha256`). Nothing below changes after sealing; later changes are dated addenda.

## Why
Studies 3-6 use one model family, which classifies sequences of window graphs. Most of the 24 audited federated IDS
papers instead classify single flow or packet records with a multilayer perceptron or similar network, and none of
them used a temporally separated split. Study 7 asks whether, in that common setting, a random record-level split
changes MCC (and the accuracy and F1 that the audited papers report) compared with a temporal split, and whether it
changes the ranking of simple federated methods.

## Data
- WUSTL-IIoT-2021 (all records, StartTime), Edge-IIoTset (DNN-EdgeIIoT-dataset.csv, the full release; frame.time
  carries only a time of day, so records are ordered by that time of day, as the window construction of Studies 3-6
  does), UNSW-NB15 (the four CSV files, Stime). TON_IoT-Network is not used: its timestamps do not form one timeline.
- Features: as in Studies 3-5. Address, port, absolute-time, label and attack-type columns are excluded; on UNSW-NB15
  the time-to-live fields (sttl, dttl, ct_state_ttl) are also excluded. Text columns with at most 50 distinct values
  are integer-coded; other text columns are dropped; numeric columns are coerced, non-finite values set to 0.
- Exact duplicate rows (features and label) are removed. Records are sorted by time (stable, file order breaks ties)
  and, if more than 300,000 remain, every k-th record is kept (k = ceil(n / 300,000)).

## Split protocols (same records in both)
- Temporal: the time-ordered records form 5 contiguous blocks of equal size, blocks 2 and 4 validation (WUSTL-IIoT-
  2021, Edge-IIoTset), or 10 blocks with even-numbered blocks validation (UNSW-NB15), as in Studies 3-5. Label-count
  rule, checked before any model is run: if either region has less than 1% of either class, the dataset uses 10
  interleaved blocks instead (logged in `results/prep_log.txt`).
- Random: every record goes to validation with probability equal to the temporal validation share (seed = 1,000,003 x
  partition seed), otherwise to training. The pilot protocol of the main study, at record level.
- In both protocols, min-max scaling is fitted on the training records only, so the comparison isolates the split.
- Validation: at most 40,000 validation records, sampled uniformly (seed = partition seed).

## Federation, methods, budgets
- K = 4 clients; training records are divided by a per-class Dirichlet draw with alpha = 0.3 (partition seeds
  101-110; redrawn with the next sub-seed until every client holds at least 100 records); all labels are used.
- Methods: LR-FedAvg (logistic regression), MLP-FedAvg (two hidden layers of 64 units, ReLU), MLP-FedProx (mu = 0.1).
  Adam, learning rate 1e-3, mini-batches of 64, class-weighted cross-entropy; FedAvg server weighted by client size.
- Budgets: 30 steps (10 rounds x 3 local steps) and 1,000 steps (50 rounds x 20 local steps).
- Model seeds 201 and 202. 3 datasets x 10 partitions x 2 seeds x 3 methods x 2 splits x 2 budgets = 720 runs.

## Unit and confirmatory tests
Unit = mean MCC over the two seeds of one (dataset, partition) cell; n = 30. Two-sided Wilcoxon signed-rank (zsplit),
Holm over three tests, alpha = 0.05; mean, median, positive count, bootstrap 95% interval.
- H7.1 MLP-FedAvg, H7.2 MLP-FedProx, H7.3 LR-FedAvg: MCC(random, 1,000 steps) - MCC(temporal, 1,000 steps).

## Reported, not claim-bearing
The same differences for accuracy, F1 and at 30 steps; per-dataset means; collapse counts; method ranking per
condition; the share of validation records whose scaled feature vector, rounded to four decimals, also occurs among
the training records (a near-duplicate rate) under each protocol; the three tests re-estimated with a linear mixed
model with a random intercept per dataset.

## Known limitations fixed in advance
Three datasets; Edge-IIoTset is ordered by time of day only; one network size and one learning rate; the random and
temporal protocols validate on different records, so their difference combines leakage-like effects (near-duplicate
records, temporal correlation) with a change of test distribution, as in Study 5.

## Addendum 1 (2026-10-01, before any Study 7 model was run)
The data preparation showed that the UNSW-NB15 files available on the Study 7 machine are short test copies
(120,000 rows of four identical files, 323 records after duplicate removal), not the dataset; the full files are on
the laptop that ran Study 4. No model had been run. UNSW-NB15 is therefore replaced by X-IIoTID
(X-IIoTID_dataset.csv, 820,834 records, Unix timestamps), prepared with the rules of the main study's X-IIoTID
preparation: rows whose Timestamp does not parse or whose source or destination is not a dotted IPv4 address are
dropped; the label is class3 = "Attack"; Date, Timestamp, Scr_IP, Des_IP, Scr_port, Des_port and class1-3 are
excluded; Protocol and Service are integer-coded; booleans become 0/1 and other non-numeric tokens 0. Temporal split:
5 blocks with blocks 2 and 4 for validation, with the label-count rule above. Everything else is unchanged. The first
seal is kept as SEAL_STUDY7_v1_superseded.sha256 and the package is sealed again.

## Addendum 2 (2026-10-01, before any Study 7 model was run)
Preparing the full Edge-IIoTset file in one piece exceeded the memory of the Study 7 machine. Its preparation now
reads the file in chunks of 200,000 rows in three passes (text columns and their distinct values; category
vocabularies; conversion). The rules are unchanged. The second seal is kept as SEAL_STUDY7_v2_superseded.sha256 and
the package is sealed again.
