param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$SftAdapter = ".\artifacts\dense_sft\adapter",
    [string]$SftSummary = ".\artifacts\dense_sft_evaluation\summary.json",
    [string]$PreferenceTrain = ".\data\preferences\train.jsonl",
    [string]$PreferenceDev = ".\data\preferences\dev.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_3_dpo",
    [string]$EvaluationRoot = ".\artifacts\phase_3_dpo_evaluation",
    [int]$MaxSteps = 200,
    [int]$Seed = 11,
    [switch]$SkipTraining,
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

$DpoAdapter = Join-Path $OutputRoot "adapter"
$DpoSummary = Join-Path $EvaluationRoot "summary.json"
$ValidationReport = ".\reports\PHASE_3_PREFERENCE_DATA_VALIDATION.json"
$GateReport = ".\reports\PHASE_3_STOP_GATE.md"

Invoke-PythonStep "[1/6] Checking Phase 2 or approved deviation gate..." @(
    ".\scripts\common\check_human_gate.py",
    "--phase", "2",
    "--allow-deviation"
)

foreach ($RequiredPath in @(
    $SftAdapter,
    $SftSummary,
    $PreferenceTrain,
    $PreferenceDev
)) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required Phase 3 input does not exist: $RequiredPath"
    }
}

Invoke-PythonStep "[2/6] Validating Phase 3 preference training data..." @(
    ".\scripts\phase_3_preference_alignment\validate_preferences.py",
    "--input", $PreferenceTrain,
    "--report", $ValidationReport
)

Invoke-PythonStep "[3/6] Validating Phase 3 preference development data..." @(
    ".\scripts\phase_3_preference_alignment\validate_preferences.py",
    "--input", $PreferenceDev,
    "--report", ".\reports\PHASE_3_PREFERENCE_DEV_VALIDATION.json"
)

if (-not $SkipTraining) {
    Invoke-PythonStep "[4/6] Training the DPO adapter..." @(
        ".\scripts\phase_3_preference_alignment\train_dpo.py",
        "--model-id", $ModelId,
        "--sft-adapter", $SftAdapter,
        "--train-file", $PreferenceTrain,
        "--dev-file", $PreferenceDev,
        "--output-dir", $OutputRoot,
        "--max-steps", "$MaxSteps",
        "--seed", "$Seed"
    )
}
elseif (-not (Test-Path $DpoAdapter)) {
    throw "SkipTraining was requested, but the DPO adapter is missing: $DpoAdapter"
}

if (-not $SkipEvaluation) {
    Invoke-PythonStep "[5/6] Evaluating the DPO adapter..." @(
        ".\scripts\phase_1_dense_sft\evaluate_model.py",
        "--model-id", $ModelId,
        "--adapter-path", $DpoAdapter,
        "--benchmark", ".\data\benchmark\shortchatbench_pilot.jsonl",
        "--output-dir", $EvaluationRoot,
        "--max-new-tokens", "32"
    )
}
elseif (-not (Test-Path $DpoSummary)) {
    throw "SkipEvaluation was requested, but the DPO summary is missing: $DpoSummary"
}

Invoke-PythonStep "[6/6] Creating the Phase 3 stop-gate report..." @(
    ".\scripts\phase_3_preference_alignment\check_phase_3_gate.py",
    "--sft-summary", $SftSummary,
    "--dpo-summary", $DpoSummary,
    "--out", $GateReport
)

Write-Host ""
Write-Host "Phase 3 execution completed."
Write-Host "DPO adapter: $DpoAdapter"
Write-Host "Evaluation report: $(Join-Path $EvaluationRoot 'MODEL_EVALUATION.md')"
Write-Host "Stop-gate report: $GateReport"
Write-Host ""
Write-Host "Review the raw SFT and DPO replies before editing the stop-gate report."
Write-Host "Phase 4 remains blocked until PHASE_3_STOP_GATE.md contains:"
Write-Host '  - Human decision: `PASS`'
