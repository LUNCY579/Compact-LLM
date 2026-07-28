<#
Phase 4 builds compatible adapter soups or records that the best individual is
better. It does not perform structural pruning; the accepted dense checkpoint
becomes the immutable source model for Phase 4B.
#>
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 3
Write-Host 'Train 3-5 compatible Phase 3 adapters, validate ingredients, evaluate each, then build and test greedy FP32 delta soups.'
Write-Host 'Record the accepted best individual or soup in reports/PHASE_4_STOP_GATE.md.'
Write-Host 'Do not prune it here; pass the accepted checkpoint path to Phase 4B.'
