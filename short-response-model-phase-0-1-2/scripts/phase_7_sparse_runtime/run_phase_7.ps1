<#
.SYNOPSIS
Train and benchmark sparse routers when Phase 6 found a viable ratio, otherwise
measure and approve the dense runtime path.

.DESCRIPTION
A negative sparse simulation is not a product failure. This runner resolves the
Phase 6 result. For a viable ratio it trains routers and compares dense/routed
decode. For `NONE` it skips router training, measures the dense baseline, and
creates a Phase 7 DENSE_RUNTIME recommendation for Phase 8.
#>
param(
    [string]$Model = "",
    [string]$RouterDataset = ".\artifacts\phase_6_sparse\router_dataset.pt",
    [string]$Benchmark = ".\data\benchmark\shortchatbench_pilot.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_7_router",
    [double]$TargetRatio = 0.0,
    [int]$RouterEpochs = 20,
    [int]$RouterWidth = 128,
    [int]$MaxCases = 50,
    [int]$MaxNewTokens = 24,
    [ValidateSet("eos_only", "one_sentence")][string]$ResponseControl = "one_sentence",
    [int]$Repeats = 3,
    [switch]$ReuseExisting
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

Invoke-PythonStep "[1/5] Checking the Phase 6 human gate..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "6"
)
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = python .\scripts\common\resolve_reference_model.py
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve the selected model." }
}
if (-not (Test-Path $Model)) { throw "Phase 7 requires a local selected checkpoint: $Model" }
if (-not (Test-Path $Benchmark)) { throw "Missing Phase 7 benchmark: $Benchmark" }
if ($Benchmark -match "sealed") { throw "Phase 7 may not use sealed evaluation data." }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Invoke-PythonStep "[2/5] Verifying the strict response-control contract..." @(
    ".\scripts\phase_7_sparse_runtime\verify_response_control.py",
    "--out", (Join-Path $OutputRoot "response_control_verification.json")
)

$ResolvedRatio = ""
if ($TargetRatio -gt 0.0) {
    $ResolvedRatio = "$TargetRatio"
} else {
    $ResolvedRatio = (python .\scripts\phase_7_sparse_runtime\resolve_phase_6_ratio.py --allow-none).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve the Phase 6 active-ratio decision." }
}

$Comparison = Join-Path $OutputRoot "runtime_comparison.json"
if ($ResolvedRatio -eq "NONE") {
    if (-not ($ReuseExisting -and (Test-Path $Comparison))) {
        Invoke-PythonStep "[3/5] Measuring the dense-only runtime baseline..." @(
            ".\scripts\phase_7_sparse_runtime\benchmark_runtime.py",
            "--model", $Model,
            "--benchmark", $Benchmark,
            "--dense-only",
            "--max-cases", "$MaxCases",
            "--max-new-tokens", "$MaxNewTokens",
            "--response-control", $ResponseControl,
            "--repeats", "$Repeats",
            "--out", $Comparison
        )
    }
    Write-Host "[4/5] Router training skipped because Phase 6 selected DENSE_ONLY."
} else {
    if (-not (Test-Path $RouterDataset)) { throw "Missing Phase 6 router dataset: $RouterDataset" }
    $Routers = Join-Path $OutputRoot "routers.pt"
    if (-not ($ReuseExisting -and (Test-Path $Routers))) {
        Invoke-PythonStep "[3/5] Training per-layer routers with validation calibration..." @(
            ".\scripts\phase_7_sparse_runtime\train_router.py",
            "--dataset", $RouterDataset,
            "--target-ratio", $ResolvedRatio,
            "--epochs", "$RouterEpochs",
            "--width", "$RouterWidth",
            "--out", $Routers
        )
    }
    if (-not ($ReuseExisting -and (Test-Path $Comparison))) {
        Invoke-PythonStep "[4/5] Benchmarking dense and routed decode on identical prompts..." @(
            ".\scripts\phase_7_sparse_runtime\benchmark_runtime.py",
            "--model", $Model,
            "--routers", $Routers,
            "--benchmark", $Benchmark,
            "--keep-ratio", $ResolvedRatio,
            "--max-cases", "$MaxCases",
            "--max-new-tokens", "$MaxNewTokens",
            "--response-control", $ResponseControl,
            "--repeats", "$Repeats",
            "--out", $Comparison
        )
    }
}

Invoke-PythonStep "[5/5] Creating Phase 7 runtime report and stop gate..." @(
    ".\scripts\phase_7_sparse_runtime\check_phase_7_gate.py",
    "--comparison", $Comparison
)

Write-Host ""
Write-Host "Phase 7 execution completed."
Write-Host "Comparison: $Comparison"
Write-Host "Report: reports\SPARSE_RUNTIME_REPORT.md"
Write-Host "Stop gate: reports\PHASE_7_STOP_GATE.md"
Write-Host "Review the evidence, then approve either DENSE_RUNTIME or SPARSE_FEATURE_FLAG."
