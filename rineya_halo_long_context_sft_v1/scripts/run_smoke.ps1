$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $LabRoot
Set-Location $RepoRoot
$env:PYTHONUTF8 = "1"

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

& $Python (Join-Path $LabRoot "scripts\smoke_babysit.py") @args
exit $LASTEXITCODE
