$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 3
Write-Host 'Train 3-5 compatible Phase 3 adapters, validate ingredients, evaluate each, then build and test greedy FP32 delta soups.'
