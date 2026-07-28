$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 5
Write-Host 'Run profile_activations.py on the frozen dense reference, then simulate_sparsity.py.'
Write-Host 'Separately evaluate generated quality at 100/80/65/50/35 percent before approving Stop Gate 5.'
