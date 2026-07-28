<#
.SYNOPSIS
Run source-level checks for the Phase 4 through Phase 8 pipeline.

.DESCRIPTION
This does not run model training or GPU benchmarks. It compiles Python source,
runs the regression suite, verifies the six main PowerShell runners and their
referenced local Python scripts, and optionally verifies heavyweight imports.
#>
param(
    [switch]$VerifyImports
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $PSScriptRoot).Path
Set-Location $Root
$env:PYTHONPATH = $Root

Write-Host "[1/4] Compiling Python source..."
python -m compileall -q .\scripts .\tests
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

Write-Host "[2/4] Running regression tests..."
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Regression tests failed." }

Write-Host "[3/4] Verifying post-Phase-3 runner references..."
$Runners = @(
    ".\scripts\phase_4_model_soups\run_phase_4.ps1",
    ".\scripts\phase_4b_structural_reduction\run_phase_4b.ps1",
    ".\scripts\phase_5_context_system\run_phase_5.ps1",
    ".\scripts\phase_6_sparse_simulation\run_phase_6.ps1",
    ".\scripts\phase_7_sparse_runtime\run_phase_7.ps1",
    ".\scripts\phase_8_release_candidate\run_phase_8.ps1"
)
foreach ($Runner in $Runners) {
    if (-not (Test-Path $Runner)) { throw "Missing main phase runner: $Runner" }
    $Text = Get-Content $Runner -Raw
    if ($Text -notmatch "Invoke-PythonStep") {
        throw "Runner does not contain executable Python orchestration: $Runner"
    }
    $Matches = [regex]::Matches($Text, '\.\\scripts\\[^\s"'']+\.py')
    if ($Matches.Count -eq 0) { throw "Runner references no local Python executable: $Runner" }
    foreach ($Match in $Matches) {
        if (-not (Test-Path $Match.Value)) {
            throw "Runner $Runner references missing executable $($Match.Value)"
        }
    }
}

Write-Host "[4/4] Checking optional dependency imports..."
if ($VerifyImports) {
    python -c "import torch, transformers, peft, trl, datasets, safetensors, rapidfuzz, jsonschema; print('Dependency imports: PASS'); print('CUDA available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "Dependency import verification failed." }
} else {
    Write-Host "Skipped. Rerun with -VerifyImports after installing requirements."
}

Write-Host "Post-Phase-3 source verification: PASS"
