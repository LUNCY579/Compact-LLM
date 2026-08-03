<#
.SYNOPSIS
Deprecated compatibility alias for run_phase_2b_rescore_v4_5.ps1.

.DESCRIPTION
Evaluator v3.1 is superseded. This wrapper forwards every supplied parameter
to the authoritative Evaluator-v4.5 migration script so old saved commands do
not silently use an obsolete policy.
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

Write-Warning "run_phase_2b_rescore_v3_1.ps1 is deprecated; forwarding to category-neutral Evaluator v4.5."
& (Join-Path $PSScriptRoot "run_phase_2b_rescore_v4_5.ps1") @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
