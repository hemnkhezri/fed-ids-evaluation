# Study 5 results

1840 new runs + 400 reused runs (Studies 3-4).

## Confirmatory tests (n = 40 units: 4 datasets x 10 partitions; Wilcoxon, Holm over 6)

| Test | n | mean | median | positive | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| H1 FedGTCL: random minus chronological (1,000 steps, 50% labels) | 40 | +0.169 | +0.065 | 27/40 | [+0.092, +0.255] | 0.0005 * |
| H1 FedAvg-GCN-GRU: random minus chronological (1,000 steps, 50% labels) | 40 | +0.193 | +0.138 | 28/40 | [+0.111, +0.277] | 0.0017 * |
| H1 FedAvg-LSTM: random minus chronological (1,000 steps, 50% labels) | 40 | +0.151 | +0.094 | 27/40 | [+0.073, +0.232] | 0.0037 * |
| H1 FedProx: random minus chronological (1,000 steps, 50% labels) | 40 | +0.132 | +0.022 | 21/40 | [+0.060, +0.215] | 0.0100 * |
| H2 (FedGTCL - FedProx) at 30 steps minus at 1,000 steps (chronological, equal labels) | 40 | +0.535 | +0.531 | 35/40 | [+0.394, +0.675] | 0.0000 * |
| H3 (FedGTCL - FedAvg-GCN-GRU): pilot protocol minus corrected protocol | 40 | +0.320 | +0.488 | 30/40 | [+0.130, +0.498] | 0.0037 * |

Share of validation sequences that share a window with a training sequence (random split): WUSTL-IIoT-2021 100.0%, TON_IoT-Network 100.0%, Edge-IIoTset 100.0%, UNSW-NB15 100.0%

## Mean MCC, pooled over 4 datasets (degenerate runs in brackets)

| Condition | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | FedAvg-GCN-GRU (all labels) | FedAvg-LSTM (all labels) | FedProx (all labels) |
|---|---|---|---|---|---|---|---|
| random split, 30 steps (pilot) | 0.447 (15/80) | 0.162 (55/80) | 0.209 (49/80) | 0.150 (55/80) | 0.146 (60/80) | 0.197 (47/80) | 0.135 (60/80) |
| random split, 1,000 steps | 0.669 (1/80) | 0.712 (4/80) | 0.649 (5/80) | 0.779 (3/80) | 0.781 (3/80) | 0.734 (4/80) | 0.812 (2/80) |
| chronological, 30 steps | 0.430 (17/80) | 0.054 (68/80) | 0.072 (59/80) | 0.042 (70/80) | 0.085 (63/80) | 0.078 (62/80) | 0.068 (63/80) |
| chronological, 1,000 steps (corrected) | 0.500 (5/80) | 0.519 (19/80) | 0.497 (17/80) | 0.647 (7/80) | 0.570 (19/80) | 0.539 (15/80) | 0.695 (4/80) |

## Mean MCC, WUSTL-IIoT-2021 (degenerate runs in brackets)

| Condition | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | FedAvg-GCN-GRU (all labels) | FedAvg-LSTM (all labels) | FedProx (all labels) |
|---|---|---|---|---|---|---|---|
| random split, 30 steps (pilot) | 0.274 (6/20) | 0.080 (17/20) | 0.112 (16/20) | 0.074 (17/20) | 0.033 (19/20) | 0.102 (16/20) | 0.032 (19/20) |
| random split, 1,000 steps | 0.533 (1/20) | 0.625 (1/20) | 0.602 (1/20) | 0.697 (0/20) | 0.742 (0/20) | 0.687 (0/20) | 0.748 (0/20) |
| chronological, 30 steps | 0.256 (4/20) | 0.032 (17/20) | 0.063 (16/20) | 0.038 (17/20) | 0.059 (17/20) | 0.085 (15/20) | 0.059 (17/20) |
| chronological, 1,000 steps (corrected) | 0.371 (1/20) | 0.400 (4/20) | 0.297 (5/20) | 0.531 (0/20) | 0.462 (4/20) | 0.348 (6/20) | 0.531 (0/20) |

## Mean MCC, TON_IoT-Network (degenerate runs in brackets)

| Condition | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | FedAvg-GCN-GRU (all labels) | FedAvg-LSTM (all labels) | FedProx (all labels) |
|---|---|---|---|---|---|---|---|
| random split, 30 steps (pilot) | 0.311 (9/20) | 0.301 (11/20) | 0.390 (9/20) | 0.293 (12/20) | 0.308 (11/20) | 0.376 (9/20) | 0.296 (12/20) |
| random split, 1,000 steps | 0.911 (0/20) | 0.777 (1/20) | 0.804 (1/20) | 0.964 (0/20) | 0.800 (1/20) | 0.834 (2/20) | 0.971 (0/20) |
| chronological, 30 steps | 0.378 (10/20) | 0.000 (20/20) | 0.018 (18/20) | 0.000 (20/20) | 0.000 (20/20) | 0.000 (20/20) | 0.000 (20/20) |
| chronological, 1,000 steps (corrected) | 0.764 (1/20) | 0.315 (9/20) | 0.577 (6/20) | 0.586 (5/20) | 0.525 (8/20) | 0.676 (4/20) | 0.759 (2/20) |

## Mean MCC, Edge-IIoTset (degenerate runs in brackets)

| Condition | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | FedAvg-GCN-GRU (all labels) | FedAvg-LSTM (all labels) | FedProx (all labels) |
|---|---|---|---|---|---|---|---|
| random split, 30 steps (pilot) | 0.814 (0/20) | 0.243 (13/20) | 0.328 (10/20) | 0.225 (13/20) | 0.219 (13/20) | 0.309 (11/20) | 0.196 (13/20) |
| random split, 1,000 steps | 0.813 (0/20) | 0.887 (0/20) | 0.821 (0/20) | 0.985 (0/20) | 0.975 (0/20) | 0.932 (0/20) | 0.990 (0/20) |
| chronological, 30 steps | 0.697 (2/20) | 0.183 (14/20) | 0.199 (13/20) | 0.128 (16/20) | 0.240 (14/20) | 0.222 (13/20) | 0.183 (14/20) |
| chronological, 1,000 steps (corrected) | 0.468 (2/20) | 0.736 (5/20) | 0.715 (3/20) | 0.941 (0/20) | 0.698 (5/20) | 0.692 (3/20) | 0.958 (0/20) |

## Mean MCC, UNSW-NB15 (degenerate runs in brackets)

| Condition | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | FedAvg-GCN-GRU (all labels) | FedAvg-LSTM (all labels) | FedProx (all labels) |
|---|---|---|---|---|---|---|---|
| random split, 30 steps (pilot) | 0.391 (0/20) | 0.026 (14/20) | 0.005 (14/20) | 0.008 (13/20) | 0.024 (17/20) | -0.001 (11/20) | 0.015 (16/20) |
| random split, 1,000 steps | 0.420 (0/20) | 0.556 (2/20) | 0.369 (3/20) | 0.469 (3/20) | 0.608 (2/20) | 0.485 (2/20) | 0.537 (2/20) |
| chronological, 30 steps | 0.387 (1/20) | 0.001 (17/20) | 0.007 (12/20) | 0.001 (17/20) | 0.043 (12/20) | 0.007 (14/20) | 0.029 (12/20) |
| chronological, 1,000 steps (corrected) | 0.399 (1/20) | 0.624 (1/20) | 0.401 (3/20) | 0.528 (2/20) | 0.596 (2/20) | 0.440 (2/20) | 0.534 (2/20) |
