param([string]$RunId = 'train-001', [int]$Timeout = 6000)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$pythonPath = Join-Path (Split-Path $repoRoot -Parent) '.venv-mq001/Scripts/python.exe'
$recordsRoot = Join-Path (Split-Path $repoRoot -Parent) 'rineya-dataset-retention-records'
$preparedPath = Join-Path $recordsRoot 'a1-prepared-003'
$bankRoot = Join-Path $repoRoot 'research/reasoning_posttraining_v1/evidence/inspection-20260915'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
Push-Location $repoRoot
try {
    & $pythonPath -m experiments.reasoning_posttraining_v1.controller train `
        --source (Join-Path $repoRoot 'Supa') `
        --prepared $preparedPath --plan (Join-Path $preparedPath 'PLAN.json') `
        --grant (Join-Path $PSScriptRoot 'GRANT.json') `
        --journal (Join-Path $recordsRoot 'a1-journal') `
        --experiment RRP-004-A1-CORRECTED --run-id $RunId --timeout $Timeout `
        --heldout-bank (Join-Path $bankRoot 'RRP2_DEV_V1_BANK.jsonl') `
        --heldout-bank (Join-Path $bankRoot 'RRP2_DEV_V2_BANK.jsonl') `
        --heldout-bank (Join-Path $bankRoot 'RRP2_PRES_DEV_BANK.jsonl') `
        --confirm-model-execution
    $workerCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $workerCode
