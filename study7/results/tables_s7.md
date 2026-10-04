# Study 7 results

| Test | n | mean | 95% CI | positive | p (Holm) | per dataset |
|---|---|---|---|---|---|---|
| H7.1 MLP-FedAvg: MCC random minus temporal (1,000 steps) | 30 | +0.148 | [+0.059, +0.237] | 23 | 0.0002 | {'wustl': 0.2764227308464292, 'edgeiiot': 0.00800604524339793, 'xiiotid': 0.1594213768723266} |
| H7.2 MLP-FedProx: MCC random minus temporal (1,000 steps) | 30 | +0.146 | [+0.064, +0.227] | 26 | 0.0002 | {'wustl': 0.2896697073398622, 'edgeiiot': 6.589414658899617e-05, 'xiiotid': 0.14935561469632452} |
| H7.3 LR-FedAvg: MCC random minus temporal (1,000 steps) | 30 | +0.146 | [+0.017, +0.282] | 22 | 0.0093 | {'wustl': 0.14917221222714291, 'edgeiiot': 0.1836037758246874, 'xiiotid': 0.10654319283198746} |

Near-duplicate share: {'random|wustl': 0.007770000000000001, 'random|edgeiiot': 0.02084, 'random|xiiotid': 0.21515, 'temporal|wustl': 0.00297, 'temporal|edgeiiot': 0.01908, 'temporal|xiiotid': 0.00154}
Ranking: {'random|short': ['MLPprox', 'MLP', 'LR'], 'random|full': ['MLP', 'MLPprox', 'LR'], 'temporal|short': ['MLPprox', 'MLP', 'LR'], 'temporal|full': ['MLP', 'MLPprox', 'LR']}

| split | budget | method | dataset | MCC | Acc | F1 | collapsed |
|---|---|---|---|---|---|---|---|
| random | short | LR | wustl | 0.096 | 0.507 | 20.3 | 1/20 |
| random | short | LR | edgeiiot | 0.336 | 0.544 | 50.3 | 0/20 |
| random | short | LR | xiiotid | 0.175 | 0.600 | 43.5 | 0/20 |
| random | short | LR | pooled | 0.202 | 0.550 | 38.0 | 1/60 |
| random | short | MLP | wustl | 0.554 | 0.931 | 55.2 | 5/20 |
| random | short | MLP | edgeiiot | 0.597 | 0.877 | 60.7 | 3/20 |
| random | short | MLP | xiiotid | 0.546 | 0.770 | 63.4 | 1/20 |
| random | short | MLP | pooled | 0.566 | 0.859 | 59.8 | 9/60 |
| random | short | MLPprox | wustl | 0.566 | 0.937 | 55.7 | 3/20 |
| random | short | MLPprox | edgeiiot | 0.606 | 0.877 | 61.3 | 3/20 |
| random | short | MLPprox | xiiotid | 0.546 | 0.769 | 63.5 | 1/20 |
| random | short | MLPprox | pooled | 0.573 | 0.861 | 60.2 | 7/60 |
| random | full | LR | wustl | 0.624 | 0.970 | 62.7 | 4/20 |
| random | full | LR | edgeiiot | 0.949 | 0.982 | 95.1 | 0/20 |
| random | full | LR | xiiotid | 0.748 | 0.876 | 83.7 | 0/20 |
| random | full | LR | pooled | 0.774 | 0.943 | 80.5 | 4/60 |
| random | full | MLP | wustl | 0.803 | 0.984 | 80.6 | 2/20 |
| random | full | MLP | edgeiiot | 1.000 | 1.000 | 100.0 | 0/20 |
| random | full | MLP | xiiotid | 0.857 | 0.928 | 90.6 | 0/20 |
| random | full | MLP | pooled | 0.887 | 0.971 | 90.4 | 2/60 |
| random | full | MLPprox | wustl | 0.792 | 0.982 | 79.6 | 2/20 |
| random | full | MLPprox | edgeiiot | 1.000 | 1.000 | 100.0 | 0/20 |
| random | full | MLPprox | xiiotid | 0.861 | 0.931 | 91.2 | 0/20 |
| random | full | MLPprox | pooled | 0.884 | 0.971 | 90.3 | 2/60 |
| temporal | short | LR | wustl | 0.071 | 0.507 | 15.1 | 0/20 |
| temporal | short | LR | edgeiiot | 0.227 | 0.443 | 31.3 | 0/20 |
| temporal | short | LR | xiiotid | 0.076 | 0.557 | 36.6 | 0/20 |
| temporal | short | LR | pooled | 0.124 | 0.502 | 27.6 | 0/60 |
| temporal | short | MLP | wustl | 0.406 | 0.978 | 40.6 | 6/20 |
| temporal | short | MLP | edgeiiot | 0.471 | 0.878 | 48.1 | 6/20 |
| temporal | short | MLP | xiiotid | 0.487 | 0.737 | 65.0 | 4/20 |
| temporal | short | MLP | pooled | 0.455 | 0.865 | 51.2 | 16/60 |
| temporal | short | MLPprox | wustl | 0.421 | 0.978 | 42.0 | 6/20 |
| temporal | short | MLPprox | edgeiiot | 0.465 | 0.877 | 47.8 | 6/20 |
| temporal | short | MLPprox | xiiotid | 0.488 | 0.738 | 65.3 | 4/20 |
| temporal | short | MLPprox | pooled | 0.458 | 0.864 | 51.7 | 16/60 |
| temporal | full | LR | wustl | 0.475 | 0.979 | 47.6 | 4/20 |
| temporal | full | LR | edgeiiot | 0.765 | 0.967 | 76.6 | 3/20 |
| temporal | full | LR | xiiotid | 0.642 | 0.815 | 74.1 | 0/20 |
| temporal | full | LR | pooled | 0.627 | 0.920 | 66.1 | 7/60 |
| temporal | full | MLP | wustl | 0.526 | 0.983 | 51.3 | 3/20 |
| temporal | full | MLP | edgeiiot | 0.992 | 0.998 | 99.2 | 0/20 |
| temporal | full | MLP | xiiotid | 0.697 | 0.840 | 78.3 | 0/20 |
| temporal | full | MLP | pooled | 0.739 | 0.940 | 76.3 | 3/60 |
| temporal | full | MLPprox | wustl | 0.502 | 0.981 | 49.1 | 2/20 |
| temporal | full | MLPprox | edgeiiot | 1.000 | 1.000 | 100.0 | 0/20 |
| temporal | full | MLPprox | xiiotid | 0.712 | 0.848 | 79.6 | 0/20 |
| temporal | full | MLPprox | pooled | 0.738 | 0.943 | 76.2 | 2/60 |
