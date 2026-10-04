# Study 4: runs everything on this computer. Open PowerShell in this folder and run:
#     powershell -ExecutionPolicy Bypass -File .\run_study4.ps1 -UnswDir "C:\path\to\UNSW-NB15 folder"
# Safe to stop (Ctrl+C) and start again: finished steps and runs are skipped.
param([Parameter(Mandatory = $true)][string]$UnswDir)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$workers = 5
function Step($cmd) { Write-Host "`n>> $cmd"; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { Write-Host "Stopped: the step above failed."; exit 1 } }
Step "python -c `"import torch, numpy, scipy, pandas, matplotlib; print('packages OK, torch', torch.__version__)`""
Step "python code\verify_seal.py"
if (-not (Test-Path .\data\unsw_windows.pkl)) { Step "python code\unsw_prepare.py `"$UnswDir`"" }
if (-not (Test-Path .\results\unsw_eligibility.json)) { Step "python code\unsw_partitions.py" }
Step "python code\run_s4.py --workers $workers"
Step "python code\analyze_s4.py results\results_s4.jsonl results"
Copy-Item .\data\unsw_prepare_log.txt .\results\ -Force
Write-Host "`nFinished. Zip the whole 'results' folder and send it back."
