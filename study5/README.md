# FedGTCL Study 5

Controlled experiment: split protocol (random vs chronological) x training budget (30 vs 1,000 steps) x baseline
labels (50% vs all), on WUSTL-IIoT-2021, TON_IoT-Network, Edge-IIoTset and UNSW-NB15. Plan: `PLAN_STUDY5.md`,
sealed in `SEAL_STUDY5.sha256`. Needs the Study 4 folder (for the UNSW-NB15 windows and partitions).

    powershell -ExecutionPolicy Bypass -File .\run_study5.ps1 -Study4Dir "<path to FedGTCL_study4>"

About 3-3.5 h with 5 worker processes. Outputs go to `results/`.
