<#
.SYNOPSIS
Build decode calibration, collect dense activation labels, run masked generation,
and create the Phase 6 quality-frontier report.
#>
param(
    [string]$Model = "",
    [string]$Calibration = ".\data\calibration\phase_6_calibration.jsonl",
    [string]$Benchmark = ".\data\benchmark\shortchatbench_pilot.jsonl",
    [string]$OutputRoot = ".\artifacts\phase_6_sparse",
    [int]$BlockSize = 128,
    [double]$RetainContribution = 0.95,
    [int]$MaxCalibrationRecords = 1000,
    [int]$MaxBenchmarkCases = 100,
    [int]$MaxNewTokens = 32,
    [ValidateSet("eos_only", "one_sentence")][string]$ResponseControl = "one_sentence",
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

Invoke-PythonStep "[1/5] Checking the Phase 5 human gate..." @(
    ".\scripts\common\check_human_gate.py", "--phase", "5"
)
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = python .\scripts\common\resolve_reference_model.py
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve the selected Phase 4B model." }
}
if (-not (Test-Path $Model)) { throw "Phase 6 requires a local selected checkpoint: $Model" }
if (-not (Test-Path $Benchmark)) { throw "Missing Phase 6 benchmark: $Benchmark" }
if ($Benchmark -match "sealed") { throw "Phase 6 must not use sealed evaluation data." }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

if (-not (Test-Path $Calibration)) {
    Invoke-PythonStep "[2/5] Building the non-sealed calibration corpus..." @(
        ".\scripts\phase_6_sparse_simulation\build_calibration_corpus.py",
        "--output", $Calibration,
        "--max-records", "$MaxCalibrationRecords"
    )
}

$Dataset = Join-Path $OutputRoot "router_dataset.pt"
if (-not ($ReuseExisting -and (Test-Path $Dataset))) {
    Invoke-PythonStep "[3/5] Collecting per-layer hidden states and oracle block labels..." @(
        ".\scripts\phase_6_sparse_simulation\profile_activations.py",
        "--model", $Model,
        "--calibration", $Calibration,
        "--block-size", "$BlockSize",
        "--retain-contribution", "$RetainContribution",
        "--max-records", "$MaxCalibrationRecords",
        "--positions", "last",
        "--out", $Dataset
    )
}

$Frontier = Join-Path $OutputRoot "frontier_with_generations.json"
if (-not ($ReuseExisting -and (Test-Path $Frontier))) {
    Invoke-PythonStep "[4/5] Running dense-compute masked generation comparisons..." @(
        ".\scripts\phase_6_sparse_simulation\simulate_sparsity.py",
        "--traces", $Dataset,
        "--model", $Model,
        "--benchmark", $Benchmark,
        "--max-cases", "$MaxBenchmarkCases",
        "--max-new-tokens", "$MaxNewTokens",
        "--response-control", $ResponseControl,
        "--ratios", "1.0", "0.8", "0.65", "0.5", "0.35",
        "--out", $Frontier
    )
}

Invoke-PythonStep "[5/5] Creating Phase 6 report and human stop gate..." @(
    ".\scripts\phase_6_sparse_simulation\check_phase_6_gate.py",
    "--frontier", $Frontier
)

Write-Host ""
Write-Host "Phase 6 execution completed."
Write-Host "Router dataset: $Dataset"
Write-Host "Quality frontier: $Frontier"
Write-Host "Report: reports\SPARSITY_CALIBRATION_REPORT.md"
Write-Host "Stop gate: reports\PHASE_6_STOP_GATE.md"
Write-Host "No speed claim is allowed from this phase. Review raw generations before PASS."
