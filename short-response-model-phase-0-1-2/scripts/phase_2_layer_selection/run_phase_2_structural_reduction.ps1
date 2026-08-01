<#
.SYNOPSIS
Run Phase 2B progressive structural reduction without rerunning the completed
Phase 2A selective-layer training study.

.DESCRIPTION
The runner materializes the accepted Phase 1 all-layer SFT reference, tests
one-layer deletion for every current layer, recovery-tunes the strongest raw
candidates, and continues progressive deletion while measured weight-memory
reduction remains greater than quality-score loss and all Quality Evaluation
v2.0 floors pass.

The PowerShell defaults intentionally mirror progressive_reduction.py and
configs/phase_2_progressive_reduction.yaml. Every quality-v2 setting is exposed
so a command is self-documenting and reproducible.
#>
param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$SftAdapter = ".\artifacts\dense_sft\adapter",
    [string]$TrainFile = ".\data\train\train.jsonl",
    [string]$DevFile = ".\data\dev\dev.jsonl",
    [string]$ProbeFile = ".\data\phase_2_structural_reduction\probe_50.jsonl",
    [string]$PhaseBenchmark = ".\data\benchmark\shortchatbench_pilot.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_2_progressive_reduction",
    [string]$Report = "",
    [string]$GateReport = "",
    [string]$PythonExecutable = "",
    [int]$ProbeLimit = 50,
    [int]$ProbeSeed = 20260728,
    [int]$MaxDrops = 6,
    [int]$RecoveryTopK = 3,
    [int]$RecoverySteps = 50,
    [int]$RecoverySeed = 11,
    [int]$MaxNewTokens = 32,
    [double]$MinimumReductionAdvantage = 0.0,
    [double]$MaxScoreLoss = 0.05,
    [double]$BalancedMaxScoreLoss = 0.02,
    [double]$MaxSemanticPassLoss = 0.02,
    [double]$MaxCategoryLoss = 0.08,
    [double]$MaxNllIncrease = 0.15,
    [double]$MinCompleteUtteranceRate = 0.99,
    [double]$MaxWorstDecileLoss = 0.05,
    [double]$MinQualityPassRate = 0.95,
    [double]$MinIntelligibilityPassRate = 1.0,
    [double]$MaxReferenceAlignmentLoss = 0.08,
    [int]$MaxNewMajorIssues = 0,
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

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $RepositoryVenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $RepositoryVenvPython) {
        $PythonExecutable = $RepositoryVenvPython
    } else {
        $PythonExecutable = "python"
    }
}

function Get-ExperimentSlug {
    param([string]$PathValue)
    $Trimmed = $PathValue.TrimEnd([char[]]'\/')
    $Leaf = Split-Path $Trimmed -Leaf
    if ([string]::IsNullOrWhiteSpace($Leaf)) { $Leaf = "phase_2_progressive_reduction" }
    return ($Leaf -replace '[^A-Za-z0-9_.-]', '_')
}

# Preserve the canonical report locations for the canonical output root. For a
# custom output root, derive unique report names unless the caller supplies them.
$CanonicalOutputRoot = ".\artifacts\phase_2_progressive_reduction"
$UsingCanonicalOutput = $OutputRoot.TrimEnd([char[]]'\/') -ieq $CanonicalOutputRoot.TrimEnd([char[]]'\/')
$ExperimentSlug = Get-ExperimentSlug $OutputRoot
if ([string]::IsNullOrWhiteSpace($Report)) {
    if ($UsingCanonicalOutput) {
        $Report = ".\reports\PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md"
    } else {
        $Report = ".\reports\PHASE_2B_${ExperimentSlug}_REPORT.md"
    }
}
if ([string]::IsNullOrWhiteSpace($GateReport)) {
    if ($UsingCanonicalOutput) {
        $GateReport = ".\reports\PHASE_2B_STOP_GATE.md"
    } else {
        $GateReport = ".\reports\PHASE_2B_${ExperimentSlug}_STOP_GATE.md"
    }
}

function Assert-UnitInterval {
    param([string]$Name, [double]$Value)
    if ($Value -lt 0.0 -or $Value -gt 1.0) {
        throw "$Name must be between 0.0 and 1.0; received $Value."
    }
}

if ($ProbeLimit -le 0) { throw "ProbeLimit must be positive." }
if ($MaxDrops -lt 0) { throw "MaxDrops cannot be negative." }
if ($RecoveryTopK -lt 0) { throw "RecoveryTopK cannot be negative." }
if ($RecoverySteps -lt 0) { throw "RecoverySteps cannot be negative." }
if ($MaxNewTokens -le 0) { throw "MaxNewTokens must be positive." }
if ($MaxNewMajorIssues -lt 0) { throw "MaxNewMajorIssues cannot be negative." }
foreach ($Threshold in @(
    @{ Name = "MaxScoreLoss"; Value = $MaxScoreLoss },
    @{ Name = "BalancedMaxScoreLoss"; Value = $BalancedMaxScoreLoss },
    @{ Name = "MaxSemanticPassLoss"; Value = $MaxSemanticPassLoss },
    @{ Name = "MaxCategoryLoss"; Value = $MaxCategoryLoss },
    @{ Name = "MaxNllIncrease"; Value = $MaxNllIncrease },
    @{ Name = "MinCompleteUtteranceRate"; Value = $MinCompleteUtteranceRate },
    @{ Name = "MaxWorstDecileLoss"; Value = $MaxWorstDecileLoss },
    @{ Name = "MinQualityPassRate"; Value = $MinQualityPassRate },
    @{ Name = "MinIntelligibilityPassRate"; Value = $MinIntelligibilityPassRate },
    @{ Name = "MaxReferenceAlignmentLoss"; Value = $MaxReferenceAlignmentLoss }
)) {
    Assert-UnitInterval -Name $Threshold.Name -Value $Threshold.Value
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host $Name
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

Invoke-PythonStep "[1/2] Checking the Phase 1 human gate..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "1"
)

foreach ($Path in @($SftAdapter, $TrainFile, $DevFile, $PhaseBenchmark)) {
    if (-not (Test-Path $Path)) { throw "Required Phase 2B input does not exist: $Path" }
}

$Arguments = @(
    ".\scripts\phase_2_layer_selection\progressive_reduction.py",
    "--model-id", $ModelId,
    "--sft-adapter", $SftAdapter,
    "--train-file", $TrainFile,
    "--dev-file", $DevFile,
    "--probe-file", $ProbeFile,
    "--phase-benchmark", $PhaseBenchmark,
    "--output-root", $OutputRoot,
    "--report", $Report,
    "--gate-report", $GateReport,
    "--probe-limit", "$ProbeLimit",
    "--probe-seed", "$ProbeSeed",
    "--max-drops", "$MaxDrops",
    "--recovery-top-k", "$RecoveryTopK",
    "--recovery-steps", "$RecoverySteps",
    "--recovery-seed", "$RecoverySeed",
    "--max-new-tokens", "$MaxNewTokens",
    "--minimum-reduction-advantage", "$MinimumReductionAdvantage",
    "--max-score-loss", "$MaxScoreLoss",
    "--balanced-max-score-loss", "$BalancedMaxScoreLoss",
    "--max-semantic-pass-loss", "$MaxSemanticPassLoss",
    "--max-category-loss", "$MaxCategoryLoss",
    "--max-nll-increase", "$MaxNllIncrease",
    "--min-complete-utterance-rate", "$MinCompleteUtteranceRate",
    "--max-worst-decile-loss", "$MaxWorstDecileLoss",
    "--min-quality-pass-rate", "$MinQualityPassRate",
    "--min-intelligibility-pass-rate", "$MinIntelligibilityPassRate",
    "--max-reference-alignment-loss", "$MaxReferenceAlignmentLoss",
    "--max-new-major-issues", "$MaxNewMajorIssues"
)
if ($QLoRA) { $Arguments += "--qlora" }
if ($GradientCheckpointing) { $Arguments += "--gradient-checkpointing" }
if ($SkipRecovery) { $Arguments += "--skip-recovery" }
if ($KeepCandidateCheckpoints) { $Arguments += "--keep-candidate-checkpoints" }
if ($Resume) { $Arguments += "--resume" }

Write-Host ""
Write-Host "Phase 2B resolved configuration:"
Write-Host "  Python:       $PythonExecutable"
Write-Host "  Probe:        $ProbeFile (limit metadata/build count: $ProbeLimit)"
Write-Host "  Output root:  $OutputRoot"
Write-Host "  Report:       $Report"
Write-Host "  Gate report:  $GateReport"
Write-Host "  Quality loss: mean <= $MaxScoreLoss; balanced <= $BalancedMaxScoreLoss"
Write-Host "  Intelligibility pass rate >= $MinIntelligibilityPassRate"

Invoke-PythonStep "[2/2] Running every-layer and progressive structural reduction..." $Arguments

Write-Host ""
Write-Host "Phase 2B completed. Review:"
Write-Host "  $Report"
Write-Host "  $GateReport"
Write-Host "Do not approve balanced/min_memory profiles until you inspect their side-by-side outputs."
