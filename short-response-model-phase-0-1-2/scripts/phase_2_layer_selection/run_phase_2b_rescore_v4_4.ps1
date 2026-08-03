<# Deprecated compatibility alias for run_phase_2b_rescore_v4_5.ps1. #>
param(
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [string]$Benchmark = ".\data\phase_2_structural_reduction\probe_150_thorough_v4_5.jsonl",
    [string]$QualityConfig = ".\configs\quality_evaluation.json",
    [string]$PythonExecutable = "",
    [switch]$ReviewOnly
)
Write-Warning "run_phase_2b_rescore_v4_4.ps1 is deprecated; forwarding to Evaluator v4.5."
& (Join-Path $PSScriptRoot "run_phase_2b_rescore_v4_5.ps1") @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
