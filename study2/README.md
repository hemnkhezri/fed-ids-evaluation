# Study 2 (Sections 4.9-4.11 of the paper)

## Order of events (all times UTC; see plans_and_seals/ and Supplementary Table S11)
1. `DEV_PROTOCOL.md` sealed (2026-09-26 20:48): 12 configurations per method, inner split of the training region only.
2. `CICAPT_PROTOCOL.md` sealed (2026-09-27 06:28) before any CICAPT-IIoT2024 window was built.
3. `PHASE2_PLAN.md` sealed (08:24): hypotheses H_A-H_C, partitions 101-110, seeds 201-202.
4. `XIIOTID_CRITERIA.md` sealed (12:49) before the X-IIoTID file was opened; addendum with column roles (12:54).
   `XIIOTID_CRITERIA_first_sealed.md` is the text as first sealed.
5. `PHASE3B_PLAN.md` sealed (13:00): baseline B5 and hypotheses H_E1-H_E3. This was written after the first family had
   been analysed, so only its B5 and X-IIoTID runs were blind.
6. Post hoc checks (not in any plan): runs repeated on a second platform; step-matched baselines.

## Folders
- `plans_and_seals/`: every sealed document and its SHA-256 file. Hash files list paths relative to the original
  working folder; code paths map to `code/`, and partition paths to `data_partitions/`.
- `code/`: `train_v3.py` (current), `train_v3_dev_sealed.py` and `train_v3_phase2_sealed.py` (the versions whose hashes
  match the development and phase-2 seals), data preparation, partition building, analyses, the figure, and
  `study2_numbers.py`, which computes every number quoted in the paper from the raw records.
- `results/`: development runs (`dev_results_local.jsonl`, `results_b5dev/`), confirmatory runs
  (`phase2_results/phase2_partition_*.jsonl`, `results_3b/`), second-platform repetitions
  (`phase2_results/cloud_crosscheck_*.jsonl`), post hoc runs (`posthoc/`), and the computed summaries.
- `data_partitions/`: the development-dataset partitions (`*.pkl`) and partition summaries. The CICAPT-IIoT2024 and
  X-IIoTID window and partition files (about 500 MB) are archived on Zenodo; their hashes are in
  `plans_and_seals/HELDOUT*_HASHES.sha256`, and they can be rebuilt from the public datasets with
  `cicapt_prepare.py` / `xiiot_prepare.py` and the partition builders.

## Recomputing the paper's numbers
    python code/study2_numbers.py <folder holding the phase-2 and phase-3b jsonl files>

Paths at the top of the scripts point to the original locations and must be edited.
