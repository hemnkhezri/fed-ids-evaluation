# Study 4: selection of a revised method and its confirmation on new partitions and on an external dataset

Written before any Study 4 run. `code/seal.py` records the SHA-256 of this file, the code, the configurations, the
development partitions and the UNSW-NB15 label check in `SEAL_STUDY4.sha256`, with the time of sealing. Nothing below
is changed after the first run; any later change is added as a dated addendum.

## Why this study
Study 3 (sealed plan, 1,440 runs, leakage-free splits, equal budgets) gave:
- Confirmatory: FedGTCL was superior to the pseudo-labelling baseline at p_L = 0.1 (mean MCC difference +0.23,
  Holm p = 0.014), and neither superior nor non-inferior to FedProx at either label fraction.
- Ablation (non-IID, p_L = 0.5): removing the contrastive loss raised mean MCC by 0.19 and removing FedAdaptOpt
  by 0.17 (both p = 0.0003).
- Post-hoc ablation at p_L = 0.1 (not in the Study 3 plan; run on a second platform): removing the contrastive loss
  raised mean MCC by 0.105 (p = 0.022), removing FedAdaptOpt by 0.078 (p = 0.012). The platform changed single runs
  but not the aggregate (full FedGTCL: 0.528 on Linux, 0.520 on Windows).

The graph-temporal encoder without the contrastive loss therefore looked better than the full method, but that
conclusion was reached on the same data it would be tested on. Study 4 selects the revised method by a fixed rule
and then tests it on data that played no part in that choice.

X-IIoTID and CICAPT-IIoT2024 cannot serve as that test. On CICAPT the federation holds about five labelled attack
sequences at p_L = 0.1. On X-IIoTID, a full-label GCN-GRU trained with the Study 3 budget reached non-IID MCC 0.026
and 0.000 on partitions 101 and 102 (checked before this plan; no candidate of this study was run on it), so no
method can be told apart from another there.

## Stage 1: selection (development data)
- Candidates (`code/cfg_s4.json`), all using the FedGTCL graph-temporal encoder without the contrastive loss:
  - C1: FedAdaptOpt server, top-s sparsification (s = 0.3), mini-batches drawn from all sequences (the "no
    contrastive" ablation of Study 3, unchanged);
  - C2: as C1, mini-batches drawn from labelled sequences only;
  - C3: labelled-only mini-batches, FedAvg server, dense updates;
  - C4: labelled-only mini-batches, FedProx (mu = 0.1), dense updates.
- Data: the Study 3 partitions 101-110 of WUSTL-IIoT-2021, TON_IoT-Network and Edge-IIoTset, non-IID, validation
  region, p_L in {0.1, 0.5}, model seeds 201 and 202, R = 50 rounds of 20 local steps (Study 3 budget).
- Rule (applied by `code/run_s4.py`): unit = mean MCC over the two seeds of one (dataset, partition, p_L) cell; the
  candidate with the highest mean over its 60 units is selected. Ties at three decimals go to the earlier candidate
  in the order C1, C2, C3, C4. The selected candidate is called S below.

## Stage 2: internal re-test (development datasets, new partitions and seeds)
- New non-IID partitions 111-120 of the same three datasets (`code/make_dev_partitions.py`, same Dirichlet rule
  as partitions 101-110; built before sealing) and new model seeds 203 and 204.
- S, B4 (FedProx GCN-GRU, mu = 0.1) and B5 (pseudo-labelling GCN-GRU), configurations unchanged from Study 3.
- This re-test changes partitions and seeds but not the validation region, which already informed the choice of
  candidates. It is a secondary check; the external test below is primary.

## Stage 3: external test (UNSW-NB15)
UNSW-NB15 has not been used in any earlier study. Only its schema and a label-count check (`docs/unsw_check.txt`,
from `code/unsw_check.py`; no feature was read and no model run) are known. That check showed two capture sessions
separated by 623 h. With the five-block split used for the other datasets, block 2 is all benign and blocks 4-5 are
all attack at every window length from 60 s down to 2 s, so every validation benign window would come from one
session and every validation attack window from the other.

Preprocessing rules (`code/unsw_prepare.py`, `code/unsw_partitions.py`), fixed from those label counts only:
1. Input: UNSW-NB15_1.csv ... UNSW-NB15_4.csv (2,540,047 rows). Row label: `Label` (attack_cat is not used).
2. Window length 1 s. Rule: the longest length in {60, 30, 10, 5, 2, 1} s for which blocks 4 and 5 of the label
   check each hold at least 5% benign windows. Only 1 s qualifies (10.1% and 11.2%; 2 s gives 1.4% and 1.2%).
3. Regions: the non-empty windows in time order are cut into ten equal contiguous blocks; blocks 2, 4, 6, 8 and 10
   are validation, the others training. This interleaving places both capture sessions in both regions.
   Segments are the blocks, further split at the gap between sessions (gap > 1 h); sequences never cross a segment.
4. Features: every column except identifiers (srcip, sport, dstip, dsport), absolute time (Stime, Ltime), labels
   (attack_cat, Label) and the TTL fields sttl, dttl and ct_state_ttl. In this testbed the attack traffic comes from
   dedicated attacker hosts whose IP time-to-live values differ from those of normal traffic, so TTL identifies the
   attacker hosts rather than attack behaviour. proto, state and service are integer-encoded (sorted vocabulary);
   non-numeric tokens become 0. Normalisation: per-feature min/max over training-region rows, clipped to [0, 1].
5. Graph per window, sequences and partitions: as for the other datasets (nodes = IP addresses, at most 30 hosts
   per window, top-8 neighbours, window label = 1 if any row in the window is an attack, T = 5 stride-1 sequences,
   K = 4, Dirichlet(0.3) non-IID partitions with partition seeds 101-110, at least 15 sequences per client).
6. Eligibility (from label counts, `results/unsw_eligibility.json`): every partition holds at least 500 attack and
   500 benign training sequences, and the validation region holds at least 30 attack and 30 benign sequences.
7. Learnability gate: the full-label reference (GCN-GRU, all labels, Study 3 budget) on partitions 101-103 with
   seed 201 must reach a mean MCC of at least 0.30. Otherwise the external test is not run and this is reported.

Runs: S, B4 and B5 (confirmatory) and the original FedGTCL, B1 and B2 (reported), each at p_L in {0.1, 0.5}, plus
the full-label reference, on partitions 101-110 with seeds 201 and 202.

## Confirmatory tests
Two families of four comparisons each: S vs B4 and S vs B5 at p_L = 0.1 and 0.5, non-IID.
- External family: UNSW-NB15, n = 10 partitions per comparison. **Primary.**
- Internal family: partitions 111-120 of the three development datasets, pooled, n = 30. Secondary.
Within each family: superiority by two-sided Wilcoxon signed-rank (zsplit), Holm over the four comparisons,
alpha = 0.05; non-inferiority with margin 0.05 MCC by one-sided Wilcoxon of d + 0.05 > 0, Holm, alpha = 0.025.
Reported with mean and median difference, wins, and a bootstrap 95% interval.
Outcome per family (as in Study 3): A = at least one superiority test significant with a positive median;
B = non-inferior in all four; B-partial = non-inferior in some; C = neither.

## Decision rules for the paper
- The paper's comparative claims follow the external family: superiority only where the external test shows it,
  non-inferiority only where it shows that. A significant disadvantage is reported with the same prominence.
- The internal family is reported as supporting evidence, with the note that its validation region also informed
  the choice of candidates.
- If the external test is not run (eligibility or gate), the paper may state only the internal results, and must
  say that the revised method was chosen with data from the same datasets.
- The original FedGTCL (with contrastive loss and FedAdaptOpt) is reported next to S on UNSW-NB15, whatever the
  result.

## Exploratory (reported, not claim-bearing)
Per-dataset results; the selection table with all four candidates; S vs B1 and B2; the original FedGTCL; the
full-label reference; degenerate-predictor counts.

## Known limitations fixed in advance
- UNSW-NB15 is a general network testbed, not an industrial one.
- Its interleaved split is closer in time than the chronological split of the other datasets; neighbouring blocks
  are separated only at their boundaries.
- n = 10 partitions for the external family limits power; with Holm over four tests, the smallest attainable
  adjusted p is about 0.008.
- Two model seeds per partition. Single runs differ between platforms; the unit of analysis is the partition mean.
