$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

python .\scripts\phase_1_dense_sft\evaluate_model.py `
  --model-id Qwen/Qwen3-0.6B `
  --adapter-path .\artifacts\dense_sft\adapter `
  --benchmark .\data\benchmark\shortchatbench_pilot.jsonl `
  --response-control eos_only `
  --output-dir .\artifacts\dense_sft_evaluation
