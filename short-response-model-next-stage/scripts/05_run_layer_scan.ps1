$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\scripts\scan_layers.py `
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
