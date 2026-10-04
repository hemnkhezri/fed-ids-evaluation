# Phase 2-3 outcome (analysis as sealed in PHASE2_PLAN.md; cross-checks added afterwards and labelled as such)

Pre-registered runs (local Windows machine, 400 runs, complete):
- H_A WUSTL-IIoT-2021, FG vs B4: mean MCC 0.206 vs 0.068, median diff +0.094, p=0.0137, p_Holm=0.027 -> FG > B4
- H_B TON_IoT + Edge-IIoTset, FG vs B4: 0.557 vs 0.289, median diff +0.238, p=0.0020, p_Holm=0.006 -> FG > B4
- H_C CICAPT-IIoT2024 (held-out), FG vs B4: 0.018 vs -0.001, p=0.094 -> no detectable difference; no method learned
  (FG discriminative in 5/20 runs, B3/B4 in 0/20)

Post-hoc robustness cross-check (not in the plan): the same cells re-run on the Linux cloud machine.
- H_A: FG mean 0.154, median diff +0.043, p=0.32 -> NOT significant. The WUSTL result is platform-fragile.
- H_B: FG mean 0.606 vs B4 0.313, median diff +0.310, p=0.0027 -> holds. It also holds against B1, B2 and B3
  (p <= 0.004 on both platforms).
- Individual FG runs differ strongly between platforms (mean |dMCC| = 0.25; 8/40 identical). Baselines are
  near-identical (B1 40/40, B4 39/40). 9 of 80 FG runs had negative MCC.

Also note: in phase 1 (inner validation inside the training region), the tuned baselines were ahead on
TON_IoT and Edge-IIoTset. On the confirmatory validation region the ordering reversed. This must be reported.
