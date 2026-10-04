# Study 3: runs everything on this computer. Open PowerShell in this folder and run:
#     powershell -ExecutionPolicy Bypass -File .\run_study3.ps1
# Safe to stop (Ctrl+C) and start again: finished runs are skipped.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$workers = 5
python -c "import torch, numpy, scipy, matplotlib; print('packages OK, torch', torch.__version__)"
python code\check_partitions.py
if (-not (Test-Path .\SEAL_STUDY3.sha256)) { Write-Host "SEAL_STUDY3.sha256 is missing: the package is incomplete."; exit 1 }
python code\run_s3.py --workers $workers
python code\efficiency.py results\efficiency.json
python code\bench_sparse.py results\bench_sparse.json
python code\analyze_s3.py results\results_s3.jsonl results
Write-Host "Finished. Send the whole 'results' folder back (zip it)."
