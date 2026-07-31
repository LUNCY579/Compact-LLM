param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$SftAdapter = ".\artifacts\dense_sft\adapter",
    [string]$SftSummary = ".\artifacts\dense_sft_evaluation\summary.json",
    [string]$Phase2Profile = "",
    [string]$Phase2Catalog = ".\artifacts\phase_2_progressive_reduction\model_catalog.json",
    [string]$Phase2BGate = ".\reports\PHASE_2B_STOP_GATE.md",
    [string]$Phase2CProfile = "",
    [string]$Phase2CCatalog = ".\artifacts\phase_2c_targeted_recovery\model_catalog.json",
    [string]$Phase2CGate = ".\reports\PHASE_2C_STOP_GATE.md",
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
if (-not [string]::IsNullOrWhiteSpace($Phase2CProfile) -and $Phase2CProfile -notin @("maximum_quality", "balanced_training", "minimum_training_memory")) {
    throw "-Phase2CProfile must be maximum_quality, balanced_training, minimum_training_memory, or omitted."
}
if (-not [string]::IsNullOrWhiteSpace($Phase2Profile) -and -not [string]::IsNullOrWhiteSpace($Phase2CProfile)) {
    throw "Use either -Phase2Profile or -Phase2CProfile, not both."
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

Invoke-PythonStep "[1/6] Checking Phase 2 or approved deviation gate..." @(
    ".\scripts\common\check_human_gate.py",
    "--phase", "2",
    "--allow-deviation"
)

# Resolve exactly one upstream branch. A Phase 2B profile supplies a materialized
# reduced checkpoint. A Phase 2C profile supplies that same checkpoint plus a
# targeted SFT adapter; they must stay paired through DPO and Phase 4 soups.
$ResolvedBranch = "legacy_full_quality"
if (-not [string]::IsNullOrWhiteSpace($Phase2CProfile)) {
    $ProfileJson = & python .\scripts\phase_2c_targeted_recovery\resolve_phase_2c_profile.py `
        --profile $Phase2CProfile --catalog $Phase2CCatalog --gate $Phase2CGate
    if ($LASTEXITCODE -ne 0) { throw "Unable to resolve Phase 2C profile $Phase2CProfile." }
    $Profile = $ProfileJson | ConvertFrom-Json
    $ModelId = [string]$Profile.training_model_id
    $SftAdapter = [string]$Profile.training_adapter
    $SftSummary = [string]$Profile.evaluation_summary
    $ResolvedBranch = "phase2c_$Phase2CProfile"
    if ($OutputRoot -eq ".\artifacts\phase_3_dpo") {
        $OutputRoot = ".\artifacts\phase_3_dpo_$ResolvedBranch"
    }
    if ($EvaluationRoot -eq ".\artifacts\phase_3_dpo_evaluation") {
        $EvaluationRoot = ".\artifacts\phase_3_dpo_${ResolvedBranch}_evaluation"
    }
}
elseif (-not [string]::IsNullOrWhiteSpace($Phase2Profile)) {
    $ProfileJson = & python .\scripts\phase_2_layer_selection\resolve_phase_2_profile.py `
        --profile $Phase2Profile --catalog $Phase2Catalog --gate $Phase2BGate
    if ($LASTEXITCODE -ne 0) { throw "Unable to resolve Phase 2B profile $Phase2Profile." }
    $Profile = $ProfileJson | ConvertFrom-Json
    $ModelId = [string]$Profile.training_model_id
    $SftAdapter = if ($null -eq $Profile.training_adapter) { "" } else { [string]$Profile.training_adapter }
    $SftSummary = [string]$Profile.evaluation_summary
    $ResolvedBranch = $Phase2Profile
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
$GateReport = if ($ResolvedBranch -eq "legacy_full_quality") {
    ".\reports\PHASE_3_STOP_GATE.md"
} else {
    ".\reports\PHASE_3_${ResolvedBranch}_STOP_GATE.md"
}

$RequiredPaths = @($SftSummary, $PreferenceTrain, $PreferenceDev)
if (-not [string]::IsNullOrWhiteSpace($SftAdapter)) { $RequiredPaths += $SftAdapter }
foreach ($RequiredPath in $RequiredPaths) {
    if (-not (Test-Path $RequiredPath)) { throw "Required Phase 3 input does not exist: $RequiredPath" }
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
    Invoke-PythonStep "[4/6] Training the DPO adapter..." $TrainArgs
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

# Add provenance without modifying evaluator calculations.
$ManifestPath = Join-Path $OutputRoot "training_manifest.json"
if (Test-Path $ManifestPath) {
    $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $Manifest | Add-Member -NotePropertyName phase_2_profile -NotePropertyValue $(if ([string]::IsNullOrWhiteSpace($Phase2Profile)) { $null } else { $Phase2Profile }) -Force
    $Manifest | Add-Member -NotePropertyName phase_2c_profile -NotePropertyValue $(if ([string]::IsNullOrWhiteSpace($Phase2CProfile)) { $null } else { $Phase2CProfile }) -Force
    $Manifest | Add-Member -NotePropertyName source_branch -NotePropertyValue $ResolvedBranch -Force
    $Manifest | ConvertTo-Json -Depth 20 | Set-Content $ManifestPath -Encoding UTF8
}

Write-Host ""
Write-Host "Phase 3 execution completed."
Write-Host "Profile: $ResolvedBranch"
Write-Host "DPO adapter: $DpoAdapter"
Write-Host "Evaluation report: $(Join-Path $EvaluationRoot 'MODEL_EVALUATION.md')"
Write-Host "Stop-gate report: $GateReport"
Write-Host "Review raw SFT-versus-DPO replies before recording a human PASS."
