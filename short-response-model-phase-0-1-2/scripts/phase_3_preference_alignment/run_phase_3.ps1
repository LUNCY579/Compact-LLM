param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$SftAdapter = ".\artifacts\dense_sft\adapter",
    [string]$SftSummary = ".\artifacts\dense_sft_evaluation\summary.json",
    [string]$Phase2Profile = "",
    [string]$Phase2Catalog = ".\artifacts\phase_2_progressive_reduction\model_catalog.json",
    [string]$Phase2BGate = ".\reports\PHASE_2B_STOP_GATE.md",
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

if (-not [string]::IsNullOrWhiteSpace($Phase2Profile) -and $Phase2Profile -notin @("full_quality", "balanced", "min_memory")) {
    throw "-Phase2Profile must be full_quality, balanced, min_memory, or omitted."
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

Invoke-PythonStep "[1/7] Checking Phase 2 or approved deviation gate..." @(
    ".\scripts\common\check_human_gate.py",
    "--phase", "2",
    "--allow-deviation"
)

# Optional profile resolution keeps the original full-quality path as the
# default while allowing approved balanced/min-memory branches to continue.
if (-not [string]::IsNullOrWhiteSpace($Phase2Profile)) {
    $ProfileJson = & python .\scripts\phase_2_layer_selection\resolve_phase_2_profile.py `
        --profile $Phase2Profile --catalog $Phase2Catalog --gate $Phase2BGate
    if ($LASTEXITCODE -ne 0) { throw "Unable to resolve Phase 2B profile $Phase2Profile." }
    $Profile = $ProfileJson | ConvertFrom-Json
    $ModelId = [string]$Profile.training_model_id
    $SftAdapter = if ($null -eq $Profile.training_adapter) { "" } else { [string]$Profile.training_adapter }
    $SftSummary = [string]$Profile.evaluation_summary

    # Keep profile experiments side by side rather than overwriting the current
    # accepted full-quality Phase 3 artifacts.
    if ($OutputRoot -eq ".\artifacts\phase_3_dpo") {
        $OutputRoot = ".\artifacts\phase_3_dpo_$Phase2Profile"
    }
    if ($EvaluationRoot -eq ".\artifacts\phase_3_dpo_evaluation") {
        $EvaluationRoot = ".\artifacts\phase_3_dpo_${Phase2Profile}_evaluation"
    }
}

$DpoAdapter = Join-Path $OutputRoot "adapter"
$DpoSummary = Join-Path $EvaluationRoot "summary.json"
$ValidationReport = ".\reports\PHASE_3_PREFERENCE_DATA_VALIDATION.json"
$GateReport = if ([string]::IsNullOrWhiteSpace($Phase2Profile)) {
    ".\reports\PHASE_3_STOP_GATE.md"
} else {
    ".\reports\PHASE_3_${Phase2Profile}_STOP_GATE.md"
}

$RequiredPaths = @($SftSummary, $PreferenceTrain, $PreferenceDev)
if (-not [string]::IsNullOrWhiteSpace($SftAdapter)) { $RequiredPaths += $SftAdapter }
foreach ($RequiredPath in $RequiredPaths) {
    if (-not (Test-Path $RequiredPath)) { throw "Required Phase 3 input does not exist: $RequiredPath" }
}

Invoke-PythonStep "[2/7] Validating Phase 3 preference training data..." @(
    ".\scripts\phase_3_preference_alignment\validate_preferences.py",
    "--input", $PreferenceTrain,
    "--report", $ValidationReport
)

Invoke-PythonStep "[3/7] Validating Phase 3 preference development data..." @(
    ".\scripts\phase_3_preference_alignment\validate_preferences.py",
    "--input", $PreferenceDev,
    "--report", ".\reports\PHASE_3_PREFERENCE_DEV_VALIDATION.json"
)

if (-not $SkipTraining) {
    $TrainArgs = @(
        ".\scripts\phase_3_preference_alignment\train_dpo.py",
        "--model-id", $ModelId,
        "--train-file", $PreferenceTrain,
        "--dev-file", $PreferenceDev,
        "--output-dir", $OutputRoot,
        "--max-steps", "$MaxSteps",
        "--seed", "$Seed"
    )
    if (-not [string]::IsNullOrWhiteSpace($SftAdapter)) {
        $TrainArgs += @("--sft-adapter", $SftAdapter)
    }
    Invoke-PythonStep "[4/7] Training the DPO adapter..." $TrainArgs
}
elseif (-not (Test-Path $DpoAdapter)) {
    throw "SkipTraining was requested, but the DPO adapter is missing: $DpoAdapter"
}

if (-not $SkipEvaluation) {
    Invoke-PythonStep "[5/7] Evaluating the DPO adapter..." @(
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

Invoke-PythonStep "[6/7] Creating the Phase 3 stop-gate report..." @(
    ".\scripts\phase_3_preference_alignment\check_phase_3_gate.py",
    "--sft-summary", $SftSummary,
    "--dpo-summary", $DpoSummary,
    "--out", $GateReport
)

# Add provenance without modifying the evaluator's score calculations.
$ManifestPath = Join-Path $OutputRoot "training_manifest.json"
if (Test-Path $ManifestPath) {
    $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $Manifest | Add-Member -NotePropertyName phase_2_profile -NotePropertyValue $(if ([string]::IsNullOrWhiteSpace($Phase2Profile)) { "legacy_full_quality" } else { $Phase2Profile }) -Force
    $Manifest | ConvertTo-Json -Depth 20 | Set-Content $ManifestPath -Encoding UTF8
}

Write-Host ""
Write-Host "Phase 3 execution completed."
Write-Host "Profile: $(if ([string]::IsNullOrWhiteSpace($Phase2Profile)) { 'legacy_full_quality' } else { $Phase2Profile })"
Write-Host "DPO adapter: $DpoAdapter"
Write-Host "Evaluation report: $(Join-Path $EvaluationRoot 'MODEL_EVALUATION.md')"
Write-Host "Stop-gate report: $GateReport"
Write-Host "Review raw SFT-versus-DPO replies before recording a human PASS."
