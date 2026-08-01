<#
.SYNOPSIS
Train compatible Phase 3 variants, evaluate them, build greedy soups, and
materialize the exact full checkpoint consumed by Phase 4B.

.DESCRIPTION
The previous runner only printed instructions. This runner performs the work:
1. checks the Phase 3 human gate;
2. reuses the accepted Phase 3 adapter as the first ingredient;
3. trains additional same-configuration DPO adapters with new seeds;
4. evaluates every ingredient with the same benchmark/settings;
5. validates adapter compatibility;
6. evaluates greedy uniform soups in reconstructed FP32 delta space;
7. writes MODEL_SOUP_REPORT.md, PHASE_4_STOP_GATE.md, and the Phase 4
   reference manifest.

The script never marks the human gate PASS. Review the raw outputs and replace
PENDING only after accepting the result.
#>
param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$SftAdapter = "",
    [string]$PrimaryAdapter = ".\artifacts\phase_3_dpo\adapter",
    [string]$Phase3GateReport = ".\reports\PHASE_3_STOP_GATE.md",
    [string]$BranchName = "full_quality",
    [string]$PrimarySummary = "", # Deprecated: Phase 4 re-evaluates the primary adapter with the common evaluator.
    [string]$PreferenceTrain = ".\data\preferences\train.jsonl",
    [string]$PreferenceDev = ".\data\preferences\dev.jsonl",
    [string]$Benchmark = ".\data\benchmark\shortchatbench_pilot.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_4_model_soups",
    [int[]]$AdditionalSeeds = @(23, 37, 49),
    [int]$MaxSteps = 200,
    [int]$MaxNewTokens = 32,
    [ValidateSet("balanced", "logic_oriented", "conversational_permissive")]
    [string]$QualityProfile = "balanced",
    [switch]$SkipTraining,
    [switch]$ReuseExisting
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

# Named Phase 2B/3 branches default to their matching adapter, gate, and output
# tree. Explicit caller values still win. This prevents a balanced/min-memory
# experiment from overwriting the accepted full-quality Phase 4 artifacts.
if ($BranchName -ne "full_quality") {
    $SafeBranchDefaults = $BranchName -replace '[^A-Za-z0-9_.-]', '_'
    if (-not ($PSBoundParameters.ContainsKey('PrimaryAdapter'))) {
        $PrimaryAdapter = ".\artifacts\phase_3_dpo_${SafeBranchDefaults}\adapter"
    }
    if (-not ($PSBoundParameters.ContainsKey('Phase3GateReport'))) {
        $Phase3GateReport = ".\reports\PHASE_3_${SafeBranchDefaults}_STOP_GATE.md"
    }
    if (-not ($PSBoundParameters.ContainsKey('OutputRoot'))) {
        $OutputRoot = ".\artifacts\phase_4_model_soups_${SafeBranchDefaults}"
    }
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

Invoke-PythonStep "[1/8] Checking the Phase 3 human gate..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "3", "--report", $Phase3GateReport
)

# Resolve the exact SFT adapter used by Phase 3 when the caller does not
# provide one. This prevents additional seeds from accidentally starting from a
# different Phase 2/SFT checkpoint.
$Phase3Root = Split-Path $PrimaryAdapter -Parent
$Phase3Manifest = Join-Path $Phase3Root "training_manifest.json"
if ([string]::IsNullOrWhiteSpace($SftAdapter)) {
    if (-not (Test-Path $Phase3Manifest)) {
        throw "-SftAdapter was omitted and the Phase 3 training manifest is missing: $Phase3Manifest"
    }
    $ManifestObject = Get-Content $Phase3Manifest -Raw | ConvertFrom-Json
    $SftAdapter = if ($null -eq $ManifestObject.sft_adapter) { "" } else { [string]$ManifestObject.sft_adapter }
    # A Phase 2B reduced profile is already a materialized SFT checkpoint, so
    # Phase 3 correctly records no predecessor adapter and attaches a fresh
    # DPO LoRA. Additional Phase 4 seeds must follow that same initialization.
    if (($PSBoundParameters.ContainsKey('ModelId') -eq $false) -and $ManifestObject.model_id) {
        $ModelId = [string]$ManifestObject.model_id
    }
}

$RequiredPaths = @($PrimaryAdapter, $PreferenceTrain, $PreferenceDev, $Benchmark)
if (-not [string]::IsNullOrWhiteSpace($SftAdapter)) { $RequiredPaths += $SftAdapter }
foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) {
        throw "Required Phase 4 input does not exist: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if ($BranchName -eq "full_quality") {
    $SoupReport = ".\reports\MODEL_SOUP_REPORT.md"
    $Phase4Gate = ".\reports\PHASE_4_STOP_GATE.md"
    $ReferenceManifest = ".\artifacts\references\phase_4_reference.json"
} else {
    $SafeBranch = $BranchName -replace '[^A-Za-z0-9_.-]', '_'
    $SoupReport = ".\reports\MODEL_SOUP_REPORT_$SafeBranch.md"
    $Phase4Gate = ".\reports\PHASE_4_${SafeBranch}_STOP_GATE.md"
    $ReferenceManifest = ".\artifacts\references\phase_4_reference_$SafeBranch.json"
}
$IngredientSpecs = New-Object System.Collections.Generic.List[string]
$SummarySpecs = New-Object System.Collections.Generic.List[string]
$AdapterPaths = New-Object System.Collections.Generic.List[string]

# Re-evaluate the accepted Phase 3 adapter under the exact same evaluator,
# benchmark, and token cap used for every new ingredient and soup trial.
$PrimaryEvalRoot = Join-Path $OutputRoot "ingredients\phase3_primary\evaluation"
$CommonPrimarySummary = Join-Path $PrimaryEvalRoot "summary.json"
if (-not ($ReuseExisting -and (Test-Path $CommonPrimarySummary))) {
    Invoke-PythonStep "[2/8] Evaluating the accepted Phase 3 primary ingredient..." @(
        ".\scripts\phase_1_dense_sft\evaluate_model.py",
        "--model-id", $ModelId,
        "--adapter-path", $PrimaryAdapter,
        "--benchmark", $Benchmark,
        "--output-dir", $PrimaryEvalRoot,
        "--max-new-tokens", "$MaxNewTokens"
    )
}
$IngredientSpecs.Add("phase3_primary=$PrimaryAdapter") | Out-Null
$SummarySpecs.Add("phase3_primary=$CommonPrimarySummary") | Out-Null
$AdapterPaths.Add($PrimaryAdapter) | Out-Null

Invoke-PythonStep "[3/8] Validating preference data before additional training..." @(
    ".\scripts\phase_3_preference_alignment\validate_preferences.py",
    "--input", $PreferenceTrain,
    "--report", (Join-Path $OutputRoot "preference_train_validation.json")
)

foreach ($Seed in $AdditionalSeeds) {
    $RunRoot = Join-Path $OutputRoot "ingredients\seed_$Seed"
    $Adapter = Join-Path $RunRoot "adapter"
    $EvalRoot = Join-Path $RunRoot "evaluation"
    $Summary = Join-Path $EvalRoot "summary.json"

    if (-not $SkipTraining) {
        if (-not ($ReuseExisting -and (Test-Path (Join-Path $Adapter "adapter_config.json")))) {
            $TrainArgs = @(
                ".\scripts\phase_3_preference_alignment\train_dpo.py",
                "--model-id", $ModelId,
                "--train-file", $PreferenceTrain,
                "--dev-file", $PreferenceDev,
                "--output-dir", $RunRoot,
                "--max-steps", "$MaxSteps",
                "--seed", "$Seed"
            )
            if (-not [string]::IsNullOrWhiteSpace($SftAdapter)) {
                $TrainArgs += @("--sft-adapter", $SftAdapter)
            }
            Invoke-PythonStep "[4/8] Training compatible DPO ingredient seed $Seed..." $TrainArgs
        }
    }
    elseif (-not (Test-Path (Join-Path $Adapter "adapter_config.json"))) {
        throw "-SkipTraining was requested, but ingredient seed $Seed is missing: $Adapter"
    }

    if (-not ($ReuseExisting -and (Test-Path $Summary))) {
        Invoke-PythonStep "[5/8] Evaluating ingredient seed $Seed..." @(
            ".\scripts\phase_1_dense_sft\evaluate_model.py",
            "--model-id", $ModelId,
            "--adapter-path", $Adapter,
            "--benchmark", $Benchmark,
            "--output-dir", $EvalRoot,
            "--max-new-tokens", "$MaxNewTokens"
        )
    }

    $Name = "seed_$Seed"
    $IngredientSpecs.Add("$Name=$Adapter") | Out-Null
    $SummarySpecs.Add("$Name=$Summary") | Out-Null
    $AdapterPaths.Add($Adapter) | Out-Null
}

$Compatibility = Join-Path $OutputRoot "compatibility.json"
$ValidateArgs = @(
    ".\scripts\phase_4_model_soups\validate_ingredients.py"
) + $AdapterPaths.ToArray() + @("--out", $Compatibility)
Invoke-PythonStep "[6/8] Validating adapter shape/config compatibility..." $ValidateArgs

$GreedyArgs = @(
    ".\scripts\phase_4_model_soups\build_greedy_soup.py",
    "--model-id", $ModelId,
    "--benchmark", $Benchmark,
    "--output-root", $OutputRoot,
    "--max-new-tokens", "$MaxNewTokens",
    "--quality-profile", $QualityProfile,
    "--reference-manifest", $ReferenceManifest,
    "--report", $SoupReport,
    "--gate-report", $Phase4Gate
)
foreach ($Spec in $IngredientSpecs) { $GreedyArgs += @("--ingredient", $Spec) }
foreach ($Spec in $SummarySpecs) { $GreedyArgs += @("--summary", $Spec) }
if ($ReuseExisting) { $GreedyArgs += "--reuse-existing" }
Invoke-PythonStep "[7/8] Building and evaluating greedy FP32 delta soups..." $GreedyArgs

Invoke-PythonStep "[8/8] Resolving the accepted Phase 4 full checkpoint..." @(
    ".\scripts\phase_4_model_soups\resolve_phase_4_model.py",
    "--manifest", $ReferenceManifest
)

Write-Host ""
Write-Host "Phase 4 execution completed."
Write-Host "Report: $SoupReport"
Write-Host "Stop gate: $Phase4Gate"
Write-Host "Reference manifest: $ReferenceManifest"
Write-Host ""
Write-Host "Review all ingredient and trial replies, then change the stop-gate line to:"
Write-Host '  - Human decision: `PASS`'
Write-Host "Phase 4B remains blocked until that explicit approval exists."
