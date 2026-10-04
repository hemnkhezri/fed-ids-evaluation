# CICAPT-IIoT2024 held-out test: preprocessing protocol (fixed before any window is built)

Information used to write this protocol: the dataset's Readme, attack_info.csv, the authors' feature-extraction
code, and a schema check of phase2_NetworkData.csv. The schema check gave column names, the row count
(9,536,823), the time span (1701468974.6-1701728424.6, about 72 h) and the label counts (label=1 on 1,004 rows;
26 technique names). No window, graph, split, model or metric has been computed on this data.

1. Input: phase2_NetworkData.csv only.
2. Row label: `label` (0 = benign, 1 = APT activity). `subLabel` and `subLabelCat` are not used.
3. Reserved columns, never features: ts, Source IP, Destination IP, Source Port, Destination Port, Protocol_name,
   label, subLabel, subLabelCat.
4. Excluded as absolute-time carriers: max_duration, min_duration, sum_duration, average_duration,
   flow_idle_time and IAT. The authors' extractor fills these with raw epoch timestamps, and a chronological
   split would let a model use them to learn the attack period instead of attack behaviour.
   All remaining numeric columns (55) are features.
5. Windows: 60-second buckets of ts measured from the first packet. Only non-empty buckets are kept, in time
   order. This is the WUSTL-IIoT-2021 rule.
6. Regions: the non-empty windows are divided into five contiguous blocks of equal length (boundaries at
   round(i*W/5)); blocks 2 and 4 are validation, the rest training. This is the rule used for WUSTL-IIoT-2021
   and Edge-IIoTset.
7. Normalisation: per-feature min/max over the rows of training-region windows, applied to every row and
   clipped to [0,1].
8. Graph per window: nodes are the IP addresses. If a window has more than 30 hosts, the 30 with the most rows
   (as source or destination) are kept, and only rows between kept hosts contribute features and edges (the
   Edge-IIoTset rule). A node's features are the mean of the normalised rows in which it appears as source or
   destination. Edge weight is the symmetric row count. Each node keeps its top-m=8 neighbours.
   Window label = 1 if any row in the window (before host capping) has label 1.
9. Sequences, partitions and training: identical to study 1 and study 2. That means T=5 stride-1 sequences
   inside one region segment, K=4, partition seed 42, IID and Dirichlet alpha=0.3 non-IID partitions with a
   minimum of 15 sequences per client, and labelled masks as in train_v3.py.
10. Feasibility rule, fixed now: if the non-IID validation region holds fewer than 10 attack sequences, the
    CICAPT results are reported descriptively only and no hypothesis test is run on them. The split is not
    redrawn.
11. Nothing on this dataset is used for development or tuning. The configurations selected in phase 1 are
    evaluated on it once, in phase 3, after the phase-2 plan has been sealed.
