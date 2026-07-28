$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

$GateReport = ".\reports\PHASE_1_STOP_GATE.md"
if (-not (Test-Path $GateReport)) {
  throw "Missing $GateReport. Complete Phase 1 first."
}
$GateText = Get-Content $GateReport -Raw
if ($GateText -notmatch "Human decision:\s*`?PASS`?") {
  throw "Phase 1 has not received an explicit human PASS decision."
}

python .\scripts\phase_2_layer_selection\scan_layers.py `
  --model-id Qwen/Qwen3-0.6B `
  --train-file .\data\train\train.jsonl `
  --dev-file .\data\dev\dev.jsonl `
  --benchmark .\data\benchmark\shortchatbench_pilot.jsonl `
  --output-dir .\artifacts\layer_scan `
  --qlora `
  --gradient-checkpointing `
  --include-all-layer-reference `
  --max-steps 200 `
  --seed 11
