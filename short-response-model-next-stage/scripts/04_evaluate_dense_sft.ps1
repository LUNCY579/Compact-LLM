$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\scripts\evaluate_base.py `
  --model-id Qwen/Qwen3-0.6B `
  --adapter-path .\artifacts\dense_sft\adapter `
  --benchmark .\data\benchmark\shortchatbench_pilot.jsonl `
  --output-dir .\artifacts\dense_sft_evaluation
