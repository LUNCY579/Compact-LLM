$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $LabRoot
Set-Location $RepoRoot
$env:PYTHONUTF8 = "1"

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Config = Join-Path $LabRoot "configs\stage_a_qlora.yaml"
$OutputDir = Join-Path $RepoRoot "artifacts\halo_long_context_sft_v1\stage_a"
$TrainMix = Join-Path $OutputDir "mix\train_stage_a.jsonl"

if (-not (Test-Path $TrainMix)) {
    throw "Stage A mix missing at $TrainMix. Run run_stage_a_train.ps1 first (or build_stage_mix.py)."
}

Write-Host "Resume status:"
& $Python (Join-Path $LabRoot "scripts\train_halo_qlora.py") --config $Config --status
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Resuming Halo QLoRA Stage A from latest checkpoint."
& $Python (Join-Path $LabRoot "scripts\train_halo_qlora.py") --config $Config --resume-from-checkpoint latest
exit $LASTEXITCODE
