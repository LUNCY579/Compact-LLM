$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 6
Write-Host 'Follow the engineering ladder: reference PyTorch, kernel prototype, look-ahead routing, placement, integration.'
Write-Host 'Approve only with preserved quality and measured speed or memory gains.'
