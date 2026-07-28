$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 4
python .\scripts\phase_5_context_system\evaluate_context_system.py
Write-Host 'Review retrieval traces, token budgets, contradiction handling, and measured context latency before approving Phase 5.'
