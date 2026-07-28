<#
Train and benchmark routers against the same selected checkpoint used to create
Phase 6 traces. Supplying -Model is reserved for an explicitly labeled control.
#>
param(
    [string]$Model = '',
    [string]$RouterDataset = 'artifacts\phase_6_sparse\router_dataset.pt',
    [string]$Benchmark = 'data\benchmark\shortchatbench_pilot.jsonl',
    [double]$TargetRatio = 0.65
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = $Root
python .\scripts\common\check_human_gate.py --phase 6

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = python .\scripts\common\resolve_reference_model.py
}
Write-Host "Phase 7 model: $Model"

$Routers = 'artifacts\phase_7_router\routers.pt'
$Comparison = 'artifacts\phase_7_router\runtime_comparison.json'

python .\scripts\phase_7_sparse_runtime\train_router.py `
  --dataset $RouterDataset --target-ratio $TargetRatio --out $Routers

python .\scripts\phase_7_sparse_runtime\benchmark_runtime.py `
  --model $Model --routers $Routers --benchmark $Benchmark `
  --keep-ratio $TargetRatio --out $Comparison

Write-Host "Review $Comparison and its raw JSONL before approving Phase 7."
