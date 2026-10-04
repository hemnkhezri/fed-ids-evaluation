# Study 5: runs everything on this computer. Open PowerShell in this folder and run:
#     powershell -ExecutionPolicy Bypass -File .\run_study5.ps1 -Study4Dir "C:\path\to\FedGTCL_study4"
# Safe to stop (Ctrl+C) and start again: finished runs are skipped.
param([Parameter(Mandatory = $true)][string]$Study4Dir)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$workers = 5
function Step($cmd) { Write-Host "`n>> $cmd"; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { Write-Host "Stopped: the step above failed."; exit 1 } }
Step "python -c `"import torch, numpy, scipy, matplotlib; print('packages OK, torch', torch.__version__)`""
Step "python code\verify_seal.py"
Step "python code\import_unsw.py `"$Study4Dir`""
Step "python code\run_s5.py --workers $workers"
Step "python code\analyze_s5.py results\results_s5.jsonl results"
Write-Host "`nFinished. Zip the whole 'results' folder and send it back."
