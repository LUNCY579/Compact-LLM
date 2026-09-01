$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (Test-Path (Join-Path $Repo ".venv\Scripts\python.exe")) {
    $Py = Join-Path $Repo ".venv\Scripts\python.exe"
} else {
    $Py = Join-Path (Split-Path -Parent $Repo) "Chat-Bubble\.venv\Scripts\python.exe"
}
& $Py (Join-Path $PSScriptRoot "smoke_babysit.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py -m pytest (Join-Path (Split-Path -Parent $PSScriptRoot) "tests\test_model_alpha_zero.py") -q
exit $LASTEXITCODE
