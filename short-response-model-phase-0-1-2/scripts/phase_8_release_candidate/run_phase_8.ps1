$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 7
Write-Host 'Run the full A-I ablation matrix, blinded human evaluation, safety checks, reproducibility audit, then build_release_audit.py.'
