<#
.SYNOPSIS
Run Phase 2B progressive structural reduction under Quality Evaluator v3.1.

.DESCRIPTION
Every current layer is tested as a reversible physical deletion. A candidate
must be worthwhile both cumulatively against the full reference and
incrementally against its immediate accepted parent. With -Resume, a completed
catalog is restored directly from its last accepted frontier stage instead of
replaying earlier stages. If focused recovery cannot find an eligible branch,
the runner recovery-tests all remaining raw branches before declaring a terminal
frontier. No absolute quality floor or issue-count veto controls promotion.
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
    [ValidateSet("balanced", "logic_oriented", "conversational_permissive")]
    [string]$QualityProfile = "balanced",
    [int]$ProbeLimit = 50,
    [int]$ProbeSeed = 20260728,
    [int]$MaxDrops = 6,
    [int]$RecoveryTopK = 3,
    [int]$RecoverySteps = 50,
    [int]$RecoverySeed = 11,
    [int]$MaxNewTokens = 32,
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
    $PythonExecutable = if (Test-Path $RepositoryVenvPython) { $RepositoryVenvPython } else { "python" }
}

function Get-ExperimentSlug {
    param([string]$PathValue)
    $Leaf = Split-Path $PathValue.TrimEnd([char[]]'\/') -Leaf
    if ([string]::IsNullOrWhiteSpace($Leaf)) { $Leaf = "phase_2_progressive_reduction" }
    return ($Leaf -replace '[^A-Za-z0-9_.-]', '_')
}

$CanonicalOutputRoot = ".\artifacts\phase_2_progressive_reduction"
$UsingCanonicalOutput = $OutputRoot.TrimEnd([char[]]'\/') -ieq $CanonicalOutputRoot.TrimEnd([char[]]'\/')
$ExperimentSlug = Get-ExperimentSlug $OutputRoot
if ([string]::IsNullOrWhiteSpace($Report)) {
    $Report = if ($UsingCanonicalOutput) { ".\reports\PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md" } else { ".\reports\PHASE_2B_${ExperimentSlug}_REPORT.md" }
}
if ([string]::IsNullOrWhiteSpace($GateReport)) {
    $GateReport = if ($UsingCanonicalOutput) { ".\reports\PHASE_2B_STOP_GATE.md" } else { ".\reports\PHASE_2B_${ExperimentSlug}_STOP_GATE.md" }
}

if ($ProbeLimit -le 0) { throw "ProbeLimit must be positive." }
if ($MaxDrops -lt 0) { throw "MaxDrops cannot be negative." }
if ($RecoveryTopK -lt 0) { throw "RecoveryTopK cannot be negative." }
if ($RecoverySteps -lt 0) { throw "RecoverySteps cannot be negative." }
if ($MaxNewTokens -le 0) { throw "MaxNewTokens must be positive." }

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
    "--quality-profile", $QualityProfile,
    "--probe-limit", "$ProbeLimit",
    "--probe-seed", "$ProbeSeed",
    "--max-drops", "$MaxDrops",
    "--recovery-top-k", "$RecoveryTopK",
    "--recovery-steps", "$RecoverySteps",
    "--recovery-seed", "$RecoverySeed",
    "--max-new-tokens", "$MaxNewTokens"
)
if ($QLoRA) { $Arguments += "--qlora" }
if ($GradientCheckpointing) { $Arguments += "--gradient-checkpointing" }
if ($SkipRecovery) { $Arguments += "--skip-recovery" }
if ($KeepCandidateCheckpoints) { $Arguments += "--keep-candidate-checkpoints" }
if ($Resume) { $Arguments += "--resume" }

Write-Host ""
Write-Host "Phase 2B resolved configuration:"
Write-Host "  Python:          $PythonExecutable"
Write-Host "  Probe:           $ProbeFile (cases: $ProbeLimit)"
Write-Host "  Output root:     $OutputRoot"
Write-Host "  Quality profile: $QualityProfile"
Write-Host "  Decision:        cumulative AND incremental weight-memory gain > quality loss"
Write-Host "  Recovery:        top-$RecoveryTopK first; exhaustive recovery only if the stage stalls"
if ($Resume) { Write-Host "  Resume:          continue directly from the last accepted frontier stage" }

Invoke-PythonStep "[2/2] Running every-layer and progressive structural reduction..." $Arguments

Write-Host ""
Write-Host "Phase 2B completed. Review:"
Write-Host "  $Report"
Write-Host "  $GateReport"
Write-Host "Human review remains required before downstream promotion."
