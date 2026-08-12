param([int]$Limit = 16)
$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
& $Python (Join-Path $PSScriptRoot "run_smoke.py") --config (Join-Path $PSScriptRoot "..\configs\hard_vote.yaml") --dataset v22_dev --limit $Limit
if ($LASTEXITCODE -ne 0) { throw "Consensus smoke failed with exit code $LASTEXITCODE" }
