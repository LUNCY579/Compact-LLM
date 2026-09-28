param([string]$RunId = 'baseline-001', [int]$Timeout = 2400, [string]$ResumeEvaluation)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$projectRoot = Split-Path $repoRoot -Parent
$pythonPath = Join-Path $projectRoot '.venv-mq001/Scripts/python.exe'
$recordsRoot = Join-Path $projectRoot 'rineya-dataset-retention-records'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$arguments = @('-m', 'experiments.reasoning_posttraining_v1.controller', 'baseline',
    '--source', (Join-Path $repoRoot 'Supa'), '--prepared', (Join-Path $recordsRoot 'a1-prepared-003'),
    '--plan', (Join-Path $PSScriptRoot 'PLAN_A2.json'), '--grant', (Join-Path $PSScriptRoot 'GRANT_A2.json'),
    '--journal', (Join-Path $recordsRoot 'preservation-baseline-a2-journal'),
    '--bank', (Join-Path $recordsRoot 'preservation-correction-002/RRP5_BASELINE_INPUTS.jsonl'),
    '--experiment', 'RRP-005-PRESERVATION-BASELINE-A2', '--run-id', $RunId, '--timeout', $Timeout,
    '--max-sequence', '2048', '--max-new-tokens', '64', '--confirm-model-execution')
if ($ResumeEvaluation) { $arguments += @('--resume-evaluation', $ResumeEvaluation) }
Push-Location $repoRoot
try {
    & $pythonPath @arguments
    $workerCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $workerCode
