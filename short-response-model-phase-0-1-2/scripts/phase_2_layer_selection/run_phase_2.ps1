$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

$GateReport = ".\reports\PHASE_1_STOP_GATE.md"
if (-not (Test-Path $GateReport)) { throw "Missing $GateReport. Complete Phase 1 first." }
$GateText = Get-Content $GateReport -Raw
if ($GateText -notmatch '(?im)^\s*-\s*Human decision:\s*`?PASS`?\s*$') {
  throw "Phase 1 needs an explicit human PASS before Phase 2."
}

python .\scripts\phase_2_layer_selection\run_static_layer_study.py `
  --model-id Qwen/Qwen3-0.6B `
  --train-file .\data\train\train.jsonl `
  --dev-file .\data\dev\dev.jsonl `
  --benchmark .\data\benchmark\shortchatbench_pilot.jsonl `
  --output-dir .\artifacts\phase_2_layer_study `
  --qlora `
  --gradient-checkpointing `
  --max-steps 200 `
  --seed 11 `
  --repeat-seeds 11,23,37 `
  --resume

python .\scripts\phase_2_layer_selection\collect_importance_signals.py
python .\scripts\phase_2_layer_selection\build_phase_2_human_review.py
python .\scripts\phase_2_layer_selection\check_phase_2_gate.py
python .\scripts\phase_2_layer_selection\plan_dynamic_selection.py

Write-Host "Phase 2 static study finished. Review reports\PHASE_2_STOP_GATE.md and record the human decision."
