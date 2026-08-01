<#
.SYNOPSIS
Run Phase 2C importance-guided targeted recovery under Evaluation v3.

.DESCRIPTION
Candidate selection compares proportional trainable-state/training-memory
savings with combined proportional quality opportunity loss. No absolute score,
pass-rate, completion, category, NLL, or issue-count threshold controls promotion.
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
    [ValidateSet("balanced", "logic_oriented", "conversational_permissive")]
    [string]$QualityProfile = "balanced",
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
    [switch]$Resume,
    [switch]$QLoRA,
    [switch]$GradientCheckpointing
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root
$PythonExecutable = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & $PythonExecutable @Arguments
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
    "--quality-profile", $QualityProfile,
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
    "--structural-weight", "$StructuralWeight"
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
Write-Host "Phase 2C execution completed under the relative opportunity policy."
Write-Host "Report: $Report"
Write-Host "Stop gate: $GateReport"
