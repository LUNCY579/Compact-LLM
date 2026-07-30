<#
.SYNOPSIS
Run Phase 2B progressive structural reduction without rerunning the completed
Phase 2A selective-layer training study.

.DESCRIPTION
The runner materializes the accepted Phase 1 all-layer SFT reference, tests
one-layer deletion for every current layer, recovery-tunes the strongest raw
candidates, and continues progressive deletion while measured weight-memory
reduction remains greater than heuristic-score loss and hard floors pass.

Every accepted frontier checkpoint is retained. Rejected candidate outputs and
metrics are retained; their large full checkpoints are reproducible and are
removed by default to avoid consuming hundreds of gigabytes.
#>
param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$SftAdapter = ".\artifacts\dense_sft\adapter",
    [string]$TrainFile = ".\data\train\train.jsonl",
    [string]$DevFile = ".\data\dev\dev.jsonl",
    [string]$ProbeFile = ".\data\phase_2_structural_reduction\probe_50.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_2_progressive_reduction",
    [int]$MaxDrops = 6,
    [int]$RecoveryTopK = 3,
    [int]$RecoverySteps = 50,
    [double]$MaxScoreLoss = 0.10,
    [double]$BalancedMaxScoreLoss = 0.03,
    [switch]$QLoRA,
    [switch]$GradientCheckpointing,
    [switch]$SkipRecovery,
    [switch]$KeepCandidateCheckpoints,
    [switch]$Resume
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

Invoke-PythonStep "[1/2] Checking the Phase 1 human gate..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "1"
)

foreach ($Path in @($SftAdapter, $TrainFile, $DevFile)) {
    if (-not (Test-Path $Path)) { throw "Required Phase 2B input does not exist: $Path" }
}

$Arguments = @(
    ".\scripts\phase_2_layer_selection\progressive_reduction.py",
    "--model-id", $ModelId,
    "--sft-adapter", $SftAdapter,
    "--train-file", $TrainFile,
    "--dev-file", $DevFile,
    "--probe-file", $ProbeFile,
    "--output-root", $OutputRoot,
    "--max-drops", "$MaxDrops",
    "--recovery-top-k", "$RecoveryTopK",
    "--recovery-steps", "$RecoverySteps",
    "--max-score-loss", "$MaxScoreLoss",
    "--balanced-max-score-loss", "$BalancedMaxScoreLoss"
)
if ($QLoRA) { $Arguments += "--qlora" }
if ($GradientCheckpointing) { $Arguments += "--gradient-checkpointing" }
if ($SkipRecovery) { $Arguments += "--skip-recovery" }
if ($KeepCandidateCheckpoints) { $Arguments += "--keep-candidate-checkpoints" }
if ($Resume) { $Arguments += "--resume" }

Invoke-PythonStep "[2/2] Running every-layer and progressive structural reduction..." $Arguments

Write-Host ""
Write-Host "Phase 2B completed. Review:"
Write-Host "  reports\PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md"
Write-Host "  reports\PHASE_2B_STOP_GATE.md"
Write-Host "Do not approve balanced/min_memory profiles until you inspect their side-by-side outputs."
