$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$Lab = Split-Path -Parent $PSScriptRoot
$Repo = Split-Path -Parent $Lab
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = Join-Path (Split-Path -Parent $Repo) "Chat-Bubble\.venv\Scripts\python.exe"
}
& $Py (Join-Path $PSScriptRoot "run_experiment.py") --phase p1
exit $LASTEXITCODE
