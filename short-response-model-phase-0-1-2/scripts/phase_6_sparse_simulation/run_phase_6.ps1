<#
Collect activation traces and run masked-quality simulation on the exact model
selected in Phase 4B. -Model is available only for explicitly labeled controls;
normal execution resolves the canonical selected-reference manifest.
#>
param(
    [string]$Model = '',
    [Parameter(Mandatory=$true)][string]$Calibration,
    [string]$Benchmark = 'data\benchmark\shortchatbench_pilot.jsonl',
    [int]$BlockSize = 128
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 5

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = python .\scripts\common\resolve_reference_model.py
}
Write-Host "Phase 6 model: $Model"

$Dataset = 'artifacts\phase_6_sparse\router_dataset.pt'
$Frontier = 'artifacts\phase_6_sparse\frontier_with_generations.json'

python .\scripts\phase_6_sparse_simulation\profile_activations.py `
  --model $Model --calibration $Calibration --block-size $BlockSize `
  --positions last --out $Dataset

python .\scripts\phase_6_sparse_simulation\simulate_sparsity.py `
  --traces $Dataset --model $Model --benchmark $Benchmark `
  --ratios 1.0 0.8 0.65 0.5 0.35 --out $Frontier

Write-Host "Review $Frontier and raw generations. This phase makes no speed claim."
