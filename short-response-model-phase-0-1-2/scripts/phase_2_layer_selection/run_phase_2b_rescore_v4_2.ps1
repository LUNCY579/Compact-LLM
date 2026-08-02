<# Deprecated compatibility alias for run_phase_2b_rescore_v4_3.ps1. #>
param(
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [string]$Benchmark = ".\data\phase_2_structural_reduction\probe_150_thorough_v4_3.jsonl",
    [string]$QualityConfig = ".\configs\quality_evaluation.json",
    [string]$PythonExecutable = "",
    [switch]$ReviewOnly
)
& (Join-Path $PSScriptRoot "run_phase_2b_rescore_v4_3.ps1") @PSBoundParameters
