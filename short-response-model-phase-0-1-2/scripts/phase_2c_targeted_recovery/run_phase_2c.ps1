<#
.SYNOPSIS
Run Phase 2C importance-guided targeted recovery on a human-approved Phase 2B reduced profile.

.DESCRIPTION
This runner builds the deterministic Phase 2C reasoning augmentation, ranks all
surviving layers with gradient/ablation/Phase-2B evidence, trains the required
selected-layer controls, repeats the strongest selective candidates across
seeds, validates retained profiles on the common phase benchmark, and writes a
human stop gate. It never uses sealed evaluation data and never records a human
PASS automatically.
#>
param(
    [ValidateSet("balanced", "min_memory")]
    [string]$Phase2BProfile = "balanced",
    [string]$Phase2BCatalog = ".\artifacts\phase_2_progressive_reduction\model_catalog.json",
    [string]$Phase2BGate = ".\reports\PHASE_2B_STOP_GATE.md",
    [string]$Phase2BRegistry = ".\artifacts\phase_2_progressive_reduction\candidate_registry.json",
    [string]$OutputRoot = ".\artifacts\phase_2c_targeted_recovery",
    [string]$Report = ".\reports\PHASE_2C_TARGETED_RECOVERY_REPORT.md",
    [string]$GateReport = ".\reports\PHASE_2C_STOP_GATE.md",
    [int]$CalibrationLimit = 64,
    [int]$EvalLimit = 180,
    [int]$MaxSteps = 200,
    [int]$BaseSeed = 11,
    [string]$RepeatSeeds = "23,37",
    [int]$RepeatTopK = 2,
    [int]$LoraRank = 16,
    [int]$LoraAlpha = 32,
    [double]$LoraDropout = 0.05,
    [double]$LearningRate = 0.0002,
    [int]$GradientAccumulationSteps = 8,
    [int]$MaxSequenceLength = 1024,
    [double]$GradientWeight = 0.35,
    [double]$AblationWeight = 0.45,
    [double]$StructuralWeight = 0.20,
    [double]$MaxQualityIndexLoss = 0.02,
    [double]$MaxFrozenQualityLoss = 0.01,
    [double]$MaxHeuristicLoss = 0.02,
    [double]$MaxSemanticPassLoss = 0.02,
    [double]$MaxReferenceF1Loss = 0.03,
    [double]$MaxCategoryLoss = 0.05,
    [double]$MaxFrozenCategoryLoss = 0.03,
    [double]$MaxNllIncrease = 0.10,
    [double]$MinCompleteUtteranceRate = 0.99,
    [double]$MinimumTrainableReduction = 0.25,
    [int]$MaxNewMajorIssues = 0,
    [double]$BalancedQualityTolerance = 0.01,
    [switch]$Resume,
    [switch]$QLoRA,
    [switch]$GradientCheckpointing
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

Invoke-PythonStep "[1/3] Building/validating the Phase 2C dataset..." @(
    ".\scripts\phase_2c_targeted_recovery\build_phase_2c_dataset.py"
)

$Args = @(
    ".\scripts\phase_2c_targeted_recovery\targeted_recovery.py",
    "--phase2b-profile", $Phase2BProfile,
    "--phase2b-catalog", $Phase2BCatalog,
    "--phase2b-gate", $Phase2BGate,
    "--phase2b-registry", $Phase2BRegistry,
    "--output-root", $OutputRoot,
    "--report", $Report,
    "--gate-report", $GateReport,
    "--calibration-limit", "$CalibrationLimit",
    "--eval-limit", "$EvalLimit",
    "--max-steps", "$MaxSteps",
    "--base-seed", "$BaseSeed",
    "--repeat-seeds", $RepeatSeeds,
    "--repeat-top-k", "$RepeatTopK",
    "--lora-rank", "$LoraRank",
    "--lora-alpha", "$LoraAlpha",
    "--lora-dropout", "$LoraDropout",
    "--learning-rate", "$LearningRate",
    "--gradient-accumulation-steps", "$GradientAccumulationSteps",
    "--max-sequence-length", "$MaxSequenceLength",
    "--gradient-weight", "$GradientWeight",
    "--ablation-weight", "$AblationWeight",
    "--structural-weight", "$StructuralWeight",
    "--max-quality-index-loss", "$MaxQualityIndexLoss",
    "--max-frozen-quality-loss", "$MaxFrozenQualityLoss",
    "--max-heuristic-loss", "$MaxHeuristicLoss",
    "--max-semantic-pass-loss", "$MaxSemanticPassLoss",
    "--max-reference-f1-loss", "$MaxReferenceF1Loss",
    "--max-category-loss", "$MaxCategoryLoss",
    "--max-frozen-category-loss", "$MaxFrozenCategoryLoss",
    "--max-nll-increase", "$MaxNllIncrease",
    "--min-complete-utterance-rate", "$MinCompleteUtteranceRate",
    "--minimum-trainable-reduction", "$MinimumTrainableReduction",
    "--max-new-major-issues", "$MaxNewMajorIssues",
    "--balanced-quality-tolerance", "$BalancedQualityTolerance"
)
if ($Resume) { $Args += "--resume" }
if ($QLoRA) { $Args += "--qlora" }
if ($GradientCheckpointing) { $Args += "--gradient-checkpointing" }
Invoke-PythonStep "[2/3] Running importance ranking and targeted recovery..." $Args

Invoke-PythonStep "[3/3] Checking automated Phase 2C evidence..." @(
    ".\scripts\phase_2c_targeted_recovery\check_phase_2c_gate.py",
    "--catalog", (Join-Path $OutputRoot "model_catalog.json"),
    "--registry", (Join-Path $OutputRoot "candidate_registry.json")
)

Write-Host ""
Write-Host "Phase 2C execution completed."
Write-Host "Catalog: $(Join-Path $OutputRoot 'model_catalog.json')"
Write-Host "Report: $Report"
Write-Host "Stop gate: $GateReport"
Write-Host "Review raw outputs and repeat seeds before recording Human decision: PASS."
