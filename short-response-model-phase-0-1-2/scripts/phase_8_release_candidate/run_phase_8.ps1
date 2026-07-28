<#
Build the final release audit from the Phase 4B-selected checkpoint unless a
specific model directory is supplied for an explicitly labeled release control.
#>
param(
    [string]$ModelDir = ''
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 7

if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = python .\scripts\common\resolve_reference_model.py
}

python .\scripts\phase_8_release_candidate\build_release_audit.py `
  --model-dir $ModelDir
Write-Host 'Complete the A-I ablation matrix, blinded human evaluation, safety checks, and final human release decision.'
