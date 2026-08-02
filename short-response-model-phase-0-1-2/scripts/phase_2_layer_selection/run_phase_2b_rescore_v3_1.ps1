<#
.SYNOPSIS
Re-score stored Phase 2B generations under Quality Evaluator v3.1.

.DESCRIPTION
This command performs no model inference. It backs up the existing generation
results, re-scores every stored reply, rebuilds full-reference deviations,
archives stale frontier decisions, and prepares the same Phase 2B run to be
replayed with -Resume. Matching evidence is reused; work is regenerated only
from the first stage whose corrected parent lineage differs.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [Parameter(Mandatory = $true)]
    [string]$Benchmark,
    [string]$QualityConfig = ".\configs\quality_evaluation.json",
    [string]$PythonExecutable = "",
    [switch]$ReviewOnly
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $RepositoryVenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    $PythonExecutable = if (Test-Path $RepositoryVenvPython) { $RepositoryVenvPython } else { "python" }
}

foreach ($Path in @($RunRoot, $Benchmark, $QualityConfig)) {
    if (-not (Test-Path $Path)) { throw "Required input does not exist: $Path" }
}

$Arguments = @(
    ".\scripts\phase_2_layer_selection\rescore_phase_2b_artifacts.py",
    "--run-root", $RunRoot,
    "--benchmark", $Benchmark,
    "--quality-config", $QualityConfig
)
if (-not $ReviewOnly) {
    $Arguments += @("--in-place", "--prepare-resume")
}

Write-Host "Phase 2B Evaluator v3.1 migration:"
Write-Host "  Python:     $PythonExecutable"
Write-Host "  Run root:   $RunRoot"
Write-Host "  Benchmark:  $Benchmark"
Write-Host "  Mode:       $(if ($ReviewOnly) { 'review-only sibling outputs' } else { 'in-place backup + prepare resume' })"

& $PythonExecutable @Arguments
if ($LASTEXITCODE -ne 0) { throw "Phase 2B rescoring failed with exit code $LASTEXITCODE." }

if ($ReviewOnly) {
    Write-Host "Review-only rescoring completed. Existing official evidence was not changed."
} else {
    Write-Host "Migration completed. Now rerun the exact original Phase 2B command with -Resume."
}
