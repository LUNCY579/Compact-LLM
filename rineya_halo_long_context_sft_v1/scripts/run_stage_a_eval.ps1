param(
    [ValidateSet("val", "test", "sealed", "all")]
    [string]$Split = "all",
    [ValidateSet("baseline", "adapter", "both")]
    [string]$Arm = "both",
    [int]$Limit = 0,
    [switch]$BuildSealed
)

$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $LabRoot
Set-Location $RepoRoot
$env:PYTHONUTF8 = "1"

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Config = Join-Path $LabRoot "configs\stage_a_qlora.yaml"
$Eval = Join-Path $LabRoot "scripts\run_stage_a_eval.py"

$Command = @($Python, $Eval, "--config", $Config, "--split", $Split, "--arm", $Arm, "--resume")
if ($Limit -gt 0) { $Command += @("--limit", "$Limit") }
if ($BuildSealed) { $Command += "--build-sealed" }

Write-Host ($Command -join " ")
& $Command[0] $Command[1..($Command.Length - 1)]
exit $LASTEXITCODE
