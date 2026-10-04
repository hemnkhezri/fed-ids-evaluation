# Study 3 results

Outcome **A**: FedGTCL is superior in: p_L=0.1 vs B5.

## Confirmatory tests (non-IID, pooled over 3 datasets, unit = partition)

| p_L | FG vs | n | mean diff | median diff | 95% CI (bootstrap) | p superiority (Holm) | p non-inferiority, margin 0.05 (Holm) |
|---|---|---|---|---|---|---|---|
| 0.1 | B4 | 30 | +0.020 | -0.054 | [-0.109, +0.158] | 1.0000 | 1.0000 |
| 0.1 | B5 | 30 | +0.232 | +0.134 | [+0.111, +0.360] | 0.0140 | 0.0005 |
| 0.5 | B4 | 30 | -0.152 | -0.165 | [-0.274, -0.027] | 0.0877 | 1.0000 |
| 0.5 | B5 | 30 | -0.045 | -0.069 | [-0.162, +0.072] | 1.0000 | 1.0000 |

## Mean MCC, noniid, p_L=0.1 (degenerate runs in brackets)

| Dataset | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | Pseudo-label | full-label reference |
|---|---|---|---|---|---|---|
| WUSTL-IIoT-2021 | 0.219 (4/20) | 0.146 (13/20) | 0.106 (13/20) | 0.387 (1/20) | 0.143 (13/20) | 0.462 |
| TON_IoT-Network | 0.590 (4/20) | 0.256 (10/20) | 0.452 (6/20) | 0.223 (15/20) | 0.063 (18/20) | 0.525 |
| Edge-IIoTset | 0.752 (1/20) | 0.740 (3/20) | 0.691 (3/20) | 0.890 (0/20) | 0.658 (5/20) | 0.698 |
| Pooled | 0.520 (9/60) | 0.381 (26/60) | 0.416 (22/60) | 0.500 (16/60) | 0.288 (36/60) | - |

## Mean MCC, noniid, p_L=0.5 (degenerate runs in brackets)

| Dataset | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | Pseudo-label | full-label reference |
|---|---|---|---|---|---|---|
| WUSTL-IIoT-2021 | 0.371 (1/20) | 0.400 (4/20) | 0.297 (5/20) | 0.531 (0/20) | 0.488 (2/20) | 0.462 |
| TON_IoT-Network | 0.764 (1/20) | 0.315 (9/20) | 0.577 (6/20) | 0.586 (5/20) | 0.540 (5/20) | 0.525 |
| Edge-IIoTset | 0.468 (2/20) | 0.736 (5/20) | 0.715 (3/20) | 0.941 (0/20) | 0.710 (3/20) | 0.698 |
| Pooled | 0.534 (4/60) | 0.484 (18/60) | 0.530 (14/60) | 0.686 (5/60) | 0.579 (10/60) | - |

## Mean MCC, iid, p_L=0.1 (degenerate runs in brackets)

| Dataset | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | Pseudo-label | full-label reference |
|---|---|---|---|---|---|---|
| WUSTL-IIoT-2021 | 0.418 (0/20) | 0.380 (3/20) | 0.265 (3/20) | 0.441 (1/20) | 0.350 (9/20) | 0.655 |
| TON_IoT-Network | 0.924 (0/20) | 0.334 (11/20) | 0.694 (0/20) | 0.453 (7/20) | 0.418 (8/20) | 0.970 |
| Edge-IIoTset | 0.906 (0/20) | 0.985 (0/20) | 0.915 (0/20) | 0.973 (0/20) | 0.988 (0/20) | 1.000 |
| Pooled | 0.749 (0/60) | 0.566 (14/60) | 0.625 (3/60) | 0.622 (8/60) | 0.585 (17/60) | - |

## Mean MCC, iid, p_L=0.5 (degenerate runs in brackets)

| Dataset | FedGTCL | FedAvg-GCN-GRU | FedAvg-LSTM | FedProx | Pseudo-label | full-label reference |
|---|---|---|---|---|---|---|
| WUSTL-IIoT-2021 | 0.550 (0/20) | 0.640 (0/20) | 0.576 (0/20) | 0.647 (0/20) | 0.667 (0/20) | 0.655 |
| TON_IoT-Network | 0.919 (1/20) | 0.878 (1/20) | 0.962 (0/20) | 0.949 (0/20) | 0.953 (0/20) | 0.970 |
| Edge-IIoTset | 0.918 (0/20) | 0.998 (0/20) | 0.982 (0/20) | 0.999 (0/20) | 0.996 (0/20) | 1.000 |
| Pooled | 0.795 (1/60) | 0.839 (1/60) | 0.840 (0/60) | 0.865 (0/60) | 0.872 (0/60) | - |

## Ablation (non-IID, p_L=0.5): FedGTCL minus variant

- FG_noCon: mean diff -0.190, n=30, p=0.0003
- FG_noOpt: mean diff -0.174, n=30, p=0.0003
