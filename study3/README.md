# FedGTCL Study 3: leakage-free, equal-budget re-evaluation

Contents
- `PLAN_STUDY3.md`: design, hypotheses and decision rules, written and sealed (`SEAL_STUDY3.sha256`) before any run.
- `partitions/`: the ten non-IID partitions (101-110) of WUSTL-IIoT-2021, TON_IoT-Network and Edge-IIoTset, with
  chronological training/validation regions (`code/check_partitions.py` verifies that no window is shared).
- `code/train_s3.py`: trainer (fork of the Study 2 trainer; equal optimizer steps, equal labels, IID arm, optional
  true-sparse attention). `code/sparse_model.py`: O(N*m) attention, numerically equivalent to the dense-masked one.
- `code/run_s3.py`: parallel, resumable runner for the 1,440 runs. `code/timing.py`: speed test and time estimate.
- `code/analyze_s3.py`: confirmatory tests (superiority and non-inferiority, Holm), exploratory tables, figure.
- `code/efficiency.py`, `code/bench_sparse.py`: parameters, uplink payload, latency; dense versus sparse attention.


Run: `python code/timing.py --workers 5`, then `run_study3.ps1` (Windows) or the same commands in a shell.
Smoke tests with R=1 were run while writing the code (to check that every method, regime and label fraction runs);
they are not results and were not kept.
