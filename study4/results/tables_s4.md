# Study 4 results

Selected candidate: **C4** (highest mean MCC over 60 units; ties at 3 decimals go to the earlier candidate in the order C1, C2, C3, C4).

| Candidate | mean MCC (60 units) | degenerate runs |
|---|---|---|
| C1 | 0.678 | 10/120 |
| C2 | 0.649 | 13/120 |
| C3 | 0.625 | 19/120 |
| C4 | 0.702 | 11/120 |

### select, p_L = 0.1 (mean MCC over partitions; degenerate runs in brackets)

| Dataset | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| WUSTL-IIoT-2021 | 0.315 (2/20) | 0.351 (3/20) | 0.215 (6/20) | 0.489 (0/20) |
| TON_IoT-Network | 0.724 (2/20) | 0.720 (1/20) | 0.580 (5/20) | 0.474 (8/20) |
| Edge-IIoTset | 0.854 (1/20) | 0.732 (2/20) | 0.818 (2/20) | 0.920 (0/20) |
| Pooled | 0.631 (5/60) | 0.601 (6/60) | 0.538 (13/60) | 0.627 (8/60) |

### select, p_L = 0.5 (mean MCC over partitions; degenerate runs in brackets)

| Dataset | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| WUSTL-IIoT-2021 | 0.425 (2/20) | 0.385 (3/20) | 0.458 (2/20) | 0.565 (0/20) |
| TON_IoT-Network | 0.924 (0/20) | 0.924 (1/20) | 0.840 (2/20) | 0.876 (1/20) |
| Edge-IIoTset | 0.823 (3/20) | 0.784 (3/20) | 0.841 (2/20) | 0.890 (2/20) |
| Pooled | 0.724 (5/60) | 0.698 (7/60) | 0.713 (6/60) | 0.777 (3/60) |

## External confirmatory family: outcome **C** (neither superior nor non-inferior)

| p_L | vs | n | mean diff | median diff | wins | 95% CI | p sup. (Holm) | p NI, margin 0.05 (Holm) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | FedProx | 10 | -0.138 | -0.145 | 2/10 | [-0.204, -0.067] | 0.0195 | 1.0000 |
| 0.1 | Pseudo-label | 10 | -0.083 | -0.079 | 3/10 | [-0.176, -0.005] | 0.1055 | 1.0000 |
| 0.5 | FedProx | 10 | -0.175 | -0.195 | 1/10 | [-0.242, -0.110] | 0.0156 | 1.0000 |
| 0.5 | Pseudo-label | 10 | -0.291 | -0.305 | 1/10 | [-0.362, -0.208] | 0.0156 | 1.0000 |

## Internal confirmatory family: outcome **A** (superior in: p_L=0.1 vs FedProx, p_L=0.1 vs Pseudo-label, p_L=0.5 vs Pseudo-label)

| p_L | vs | n | mean diff | median diff | wins | 95% CI | p sup. (Holm) | p NI, margin 0.05 (Holm) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | FedProx | 30 | +0.086 | +0.045 | 22/30 | [+0.039, +0.135] | 0.0031 | 0.0001 |
| 0.1 | Pseudo-label | 30 | +0.341 | +0.389 | 26/30 | [+0.252, +0.427] | 0.0000 | 0.0000 |
| 0.5 | FedProx | 30 | +0.060 | +0.000 | 14/30 | [+0.012, +0.116] | 0.2368 | 0.0001 |
| 0.5 | Pseudo-label | 30 | +0.150 | +0.013 | 20/30 | [+0.073, +0.235] | 0.0014 | 0.0000 |

### internal, p_L = 0.1 (mean MCC over partitions; degenerate runs in brackets)

| Dataset | C4 | FedProx | Pseudo-label |
|---|---|---|---|
| WUSTL-IIoT-2021 | 0.483 (1/20) | 0.430 (1/20) | 0.127 (10/20) |
| TON_IoT-Network | 0.720 (2/20) | 0.562 (5/20) | 0.209 (15/20) |
| Edge-IIoTset | 0.961 (0/20) | 0.913 (0/20) | 0.805 (3/20) |
| Pooled | 0.721 (3/60) | 0.635 (6/60) | 0.380 (28/60) |

### internal, p_L = 0.5 (mean MCC over partitions; degenerate runs in brackets)

| Dataset | C4 | FedProx | Pseudo-label |
|---|---|---|---|
| WUSTL-IIoT-2021 | 0.544 (0/20) | 0.536 (0/20) | 0.425 (3/20) |
| TON_IoT-Network | 0.964 (0/20) | 0.793 (1/20) | 0.691 (5/20) |
| Edge-IIoTset | 0.992 (0/20) | 0.991 (0/20) | 0.936 (1/20) |
| Pooled | 0.833 (0/60) | 0.773 (1/60) | 0.684 (9/60) |

### external, p_L = 0.1 (mean MCC over partitions; degenerate runs in brackets)

| Dataset | C4 | FedProx | Pseudo-label | FedGTCL (original) | FedAvg-GCN-GRU | FedAvg-LSTM |
|---|---|---|---|---|---|---|
| UNSW-NB15 | 0.335 (0/20) | 0.473 (2/20) | 0.418 (0/20) | 0.364 (0/20) | 0.589 (2/20) | 0.378 (2/20) |

### external, p_L = 0.5 (mean MCC over partitions; degenerate runs in brackets)

| Dataset | C4 | FedProx | Pseudo-label | FedGTCL (original) | FedAvg-GCN-GRU | FedAvg-LSTM |
|---|---|---|---|---|---|---|
| UNSW-NB15 | 0.354 (0/20) | 0.528 (2/20) | 0.645 (2/20) | 0.399 (1/20) | 0.624 (1/20) | 0.401 (3/20) |

Full-label reference on UNSW-NB15 (non-IID): mean MCC 0.596 over 10 partitions.

Learnability gate: mean MCC 0.704 (threshold 0.3), passed.
