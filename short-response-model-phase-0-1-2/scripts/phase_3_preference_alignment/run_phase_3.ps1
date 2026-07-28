$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 2
python .\scripts\phase_3_preference_alignment\validate_preferences.py
Write-Host 'Run train_dpo.py with the Phase 2 winning adapter, then evaluate it with phase_1_dense_sft/evaluate_model.py.'
Write-Host 'Finally run check_phase_3_gate.py with the SFT and DPO summaries.'
