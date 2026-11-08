$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $LabRoot
Set-Location $RepoRoot
$env:PYTHONUTF8 = "1"

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Config = Join-Path $LabRoot "configs\stage_a_qlora.yaml"
$TrainMix = Join-Path $RepoRoot "artifacts\halo_long_context_sft_v1\stage_a\mix\train_stage_a.jsonl"

if (-not (Test-Path $TrainMix)) {
    Write-Host "Building Stage A mix..."
    & $Python (Join-Path $LabRoot "scripts\build_stage_mix.py") --config $Config --force
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Starting Halo QLoRA Stage A (fresh). Halo vault weights are not written."
& $Python (Join-Path $LabRoot "scripts\train_halo_qlora.py") --config $Config
exit $LASTEXITCODE
