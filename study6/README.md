# Study 6: budget curve and learning-rate re-tuning (Section 5.2, Supplementary S11)
`PLAN_STUDY6.md` and `SEAL_STUDY6.sha256` were written before any run. `code/run_s6.py` has three phases
(`select`, `curves`, `confirm`); `code/analyze_s6.py` runs the three sealed tests. Results: `results/*.jsonl`
(750 runs) and `results/analysis_s6.json`. The runs read the Study 3 partitions from `partitions/dev/`
(copy or link `../study3/partitions/`). The study ran on a Linux cloud machine and was resumed once after a restart.
