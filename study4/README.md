# FedGTCL Study 4

Selection of a revised method (graph-temporal encoder without the contrastive loss) by a fixed rule, an internal
re-test on new partitions and seeds, and an external test on UNSW-NB15. The design, rules and decision criteria are in
`PLAN_STUDY4.md`, sealed in `SEAL_STUDY4.sha256` before any run.

Run (Windows PowerShell, in this folder):

    powershell -ExecutionPolicy Bypass -File .\run_study4.ps1 -UnswDir "<folder with UNSW-NB15_1.csv ... _4.csv>"

Steps: seal check, UNSW-NB15 preprocessing (`code/unsw_prepare.py`), partitions and eligibility
(`code/unsw_partitions.py`), all runs (`code/run_s4.py`, resumable), analysis (`code/analyze_s4.py`).
Outputs go to `results/`. About 7-8 h with 5 worker processes on a 6-core laptop CPU.
