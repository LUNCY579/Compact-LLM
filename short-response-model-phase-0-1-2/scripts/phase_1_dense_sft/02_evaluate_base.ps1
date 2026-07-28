$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

python .\scripts\phase_1_dense_sft\evaluate_model.py `
  --model-id Qwen/Qwen3-0.6B `
  --benchmark .\data\benchmark\shortchatbench_pilot.jsonl `
  --output-dir .\artifacts\baseline_evaluation
