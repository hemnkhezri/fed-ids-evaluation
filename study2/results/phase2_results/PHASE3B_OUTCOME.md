# Phase 3b outcome (PHASE3B_PLAN.md; cross-check added afterwards and labelled as such)

B5 selection (sealed rule): B5_lr0.003_t0.9_u1.0, mean inner-validation MCC 0.585, the best of 12 configurations.

## Pre-registered tests (local runs; Holm correction over 3 tests)
| Test | Comparison | Data | Mean MCC (FG vs comparator) | Median diff | p | p_Holm | Result |
|---|---|---|---|---|---|---|---|
| H_E1 | FG vs B5 | TON_IoT + Edge-IIoTset | 0.557 vs 0.364 | +0.181 | 0.0073 | 0.022 | FG > B5 |
| H_E2 | FG vs B4 | X-IIoTID | 0.050 vs 0.035 | — | 0.074 | 0.148 | no detectable difference |
| H_E3 | FG vs B5 | X-IIoTID | 0.050 vs 0.021 | — | 0.176 | 0.176 | no detectable difference |

On X-IIoTID no method learned: every mean MCC is at or below 0.06.

## Cross-platform check of H_E1 (cloud re-run of the B5 cells, paired with the cloud FG cells from phase 2)
- H_E1 holds: FG 0.606 vs B5 0.428, median diff +0.156, p = 0.013.
- The effect comes from TON_IoT: p = 0.006 locally, p = 0.004 in the cloud.
- On Edge-IIoTset FG and B5 do not differ: p = 0.375 locally, p = 0.557 in the cloud.
- 25 of 40 B5 cells were identical across the two platforms.
