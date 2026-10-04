# Evaluation Choices Decide the Winner: code, sealed plans and results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23142334.svg)](https://doi.org/10.5281/zenodo.23142334)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, sealed analysis plans and every run-level result for the paper

> Khezri, H., & Trik, M. *Evaluation Choices Decide the Winner: A Pre-Registered Re-Evaluation of Federated
> Intrusion Detection for Industrial IoT.* (under review; journal and DOI to be added)

The paper re-evaluates FedGTCL, a federated graph-temporal intrusion detection model, and measures how much the
split protocol, the training budget and label access change the result of a comparison. A pilot showed FedGTCL well
ahead of its FedAvg baselines. Under window-disjoint chronological splits and equal budgets it was not better than
FedProx at 1,000 steps and was worse at 3,000 steps or with a re-tuned learning rate. The pilot's lead came from a
30-step training budget at which the baselines collapsed to one class. The repository also holds an audit of 24
published federated IDS studies and a per-record study showing that random record splits raise MCC by about 0.15.

## Layout

| Folder | Paper | Contents |
|---|---|---|
| `study1_pilot/` | Study 1, Supplementary S6 | Pilot code under the original protocol (random split of overlapping sequences, 30 steps). Kept for transparency; its results are not evidence. |
| `preliminary_sealed_retest/` | Supplementary S5 | First sealed re-test (one partition per dataset, ten seeds): plan, addendum, seals, code, 315 runs. Not counted among the seven studies. |
| `study2/` | Study 2, Supplementary S5 | Tuned configurations, held-out tests on CICAPT-IIoT2024 and X-IIoTID: plans, seals, code, results. See `study2/README.md`. |
| `study3/` | Study 3, Section 5.3 | Window-disjoint, equal-budget re-evaluation: plan, seal, code, partitions 101-110, 1,440 runs, post hoc runs on a second platform. |
| `study4/` | Study 4, Section 5.4 | Variant selection, internal re-test, external test on UNSW-NB15: plan, seal, code, development partitions, 1,103 runs. |
| `study5/` | Study 5, Section 5.1 | Split x budget x baseline labels: plan, seal, code, 1,840 new runs and the reused Study 3/4 runs. |
| `study6/` | Study 6, Section 5.2 | Budget curve up to 3,000 steps and learning-rate re-tuning: plan, seal, code, 750 runs, logs. |
| `study7/` | Study 7, Section 5.5 | Per-record classifiers, random vs temporal split: plan with two dated addenda, three seals, code, 720 runs, logs. |
| `audit/` | Section 4, Supplementary S9 | Literature audit: sealed protocol, screening log, two AI codings with evidence, reliability, author check sheets, final codes. |
| `analysis/` | all tables and figures | `collect_numbers.py` (every number in the paper from the result files), `hier.py` (dataset-clustered re-analysis, Supplementary S13), figure scripts. |
| `figures/` | Figs. 1-9, S1-S2 | Figures as PNG and PDF. |

## Sealing

Before the first confirmatory run of each study, a script wrote the SHA-256 hash of the plan, the code, the
configurations and the partition files to a `SEAL_*.sha256` file with a UTC time, and the run scripts refuse to start
if a sealed file differs (`code/verify_seal.py`). Sealed files are kept exactly as executed, so some of them contain
absolute paths of the machines they ran on (for example `study7/code/prepare_s7.py`); edit the paths to re-run, and
expect `verify_seal.py` to report the edit. Superseded seals and dated addenda are kept (Study 7). The seals were
written by the authors (Studies 6 and 7: by the AI assistant acting for them) without a third-party timestamp.

## Reproducing the reported numbers (no training needed)

```bash
pip install -r requirements.txt
python analysis/collect_numbers.py      # rebuilds analysis/numbers.json from the run-level results
python analysis/hier.py                 # Supplementary Table S12
python study3/code/analyze_s3.py study3/results/results_s3.jsonl <out_dir>
python study5/code/analyze_s5.py study5/results/results_s5.jsonl <out_dir>
python study6/code/analyze_s6.py        # writes study6/results/analysis_s6.json
python study7/code/analyze_s7.py        # writes study7/results/analysis_s7.json
python analysis/figs.py; python analysis/figs_new.py leakage audit s3dist curves; python analysis/figs_s67.py s6 s7
```

## Re-running the experiments

Each study folder has a runner (`code/run_s*.py`, resumable). Studies 3-5 ran on one Windows 11 laptop (PyTorch 2.14.0
CPU build, one thread per process); Studies 6 and 7 ran on a Linux cloud machine. Single federated runs can differ
across platforms; partition-level means were stable (Supplementary S8).

- Studies 5 and 6 read the Study 3 partitions from `partitions/dev/`: copy or link `study3/partitions/` there.
- UNSW-NB15 (Studies 4 and 5): the window file and partitions are rebuilt from the four public CSV files with
  `study4/code/unsw_prepare.py` and `unsw_partitions.py`; their hashes are in `study4/results/unsw_eligibility.json`.
- Study 7: `study7/code/prepare_s7.py` builds the record-level arrays from the public datasets (hashes in
  `study7/results/prep_log.txt`).

## Data

The datasets are public and are not redistributed here: WUSTL-IIoT-2021, TON_IoT (network), Edge-IIoTset, UNSW-NB15,
X-IIoTID and CICAPT-IIoT2024. The partition files in `study3/` and `study4/` contain window graphs with scaled
features derived from the first three; they are included because they are part of the sealed packages. Please cite
the dataset papers when using them.

## Use of AI

Claude (Anthropic) was used to write and test code, draft plans, run analyses, code the literature audit and, at the
authors' request, to draft, seal and run Studies 6 and 7. The paper's declaration gives the details.

## Licence and citation

Code: MIT (see `LICENSE`). Cite the paper and this archive (see `CITATION.cff`).

Archive (all versions): https://doi.org/10.5281/zenodo.23142334
