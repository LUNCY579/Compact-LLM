<#
Phase 5 starts only after Phase 4B has selected the exact model that all
remaining model-dependent work must use. The context evaluator itself is mostly
model-independent, but resolving the manifest here catches an incomplete phase
transition before router calibration begins.
#>
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root

python .\scripts\common\check_human_gate.py --phase 4B
$SelectedModel = python .\scripts\common\resolve_reference_model.py
Write-Host "Phase 4B selected reference: $SelectedModel"

python .\scripts\phase_5_context_system\evaluate_context_system.py
Write-Host 'Review retrieval traces, token budgets, contradiction handling, and measured context latency before approving Phase 5.'
