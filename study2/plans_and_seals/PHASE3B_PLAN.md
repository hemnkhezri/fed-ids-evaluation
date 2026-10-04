# Study 2, phase 3b: semi-supervised baseline and second independent test (sealed before any run)

Motivation: every baseline in phases 2-3 was supervised-only, while FedGTCL also uses unlabelled traffic.
The phase-3 test on CICAPT-IIoT2024 did not probe the hypothesis because the federation had about 5 labelled
attack sequences. Phase 3b adds two things:
- **B5**, a semi-supervised federated baseline: FedAvg GCN-GRU with FixMatch-style pseudo-labelling;
- **X-IIoTID**, a second independent dataset, eligible under XIIOTID_CRITERIA.md (see heldout2/ELIGIBILITY.md).

No model has been run on X-IIoTID. B5 has run only in a smoke test (R=5), which is not used.

## Step 1: B5 development (same protocol as phase 1)
- **Grid:** 12 configurations in cfg_dev_b5.json: client lr {1e-3, 3e-3} x confidence threshold
  {0.8, 0.9, 0.95} x unlabelled weight {0.5, 1.0}. This is the same budget as every other family.
- **Setting:** inner split, three development datasets, p_label {0.1, 0.5}, seeds 101-105, R=30.
- **Selection:** the configuration with the highest mean inner-validation MCC over its 30 runs; ties broken
  by listing order. select_b5.py applies this rule automatically and writes cfg_phase3b.json, which holds
  FG, B1-B4 exactly as in cfg_phase2.json plus the selected B5.

## Step 2: confirmatory runs
- p_label = 0.1, model seeds 201 and 202, R = 30.
- **B5:** on all 40 phase-2 partitions (WUSTL, TON_IoT, Edge-IIoTset and CICAPT).
- **FG, B1, B2, B3, B4 and B5:** on the 10 X-IIoTID partitions (seeds 101-110). These are rebuilt locally
  from heldout2/xiiotid_windows.pkl, and must match heldout2/assignment_hashes.json or the run stops.
- The earlier phase-2 runs of FG and B1-B4 are reused unchanged.

## Confirmatory hypotheses
Unit: partition-level mean MCC over the two seeds. Test: two-sided Wilcoxon signed-rank (zsplit), Holm
correction over the three tests, alpha = 0.05.

| Hypothesis | Comparison | Data | n |
|---|---|---|---|
| H_E1 | FG vs B5 | TON_IoT + Edge-IIoTset | 20 partitions |
| H_E2 | FG vs B4 | X-IIoTID | 10 partitions |
| H_E3 | FG vs B5 | X-IIoTID | 10 partitions |

## Decision rules
- The paper may claim that FedGTCL outperforms the semi-supervised baseline (H_E1, H_E3), or tuned federated
  baselines on a second independent dataset (H_E2, H_E3), only if the test is significant after Holm
  correction with a positive median difference.
- Otherwise the result is reported as "no detectable difference" or as a disadvantage, with equal prominence.
- The CICAPT and phase-2 results are reported unchanged.

## Robustness (reported, not claim-bearing)
- Cross-platform re-run of the X-IIoTID FG and B5 cells, if feasible.
- Exploratory comparisons: FG vs B5 on WUSTL and CICAPT; FG vs B1-B3 on X-IIoTID.
