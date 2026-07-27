$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\scripts\evaluate_base.py --model-id Qwen/Qwen3-0.6B --benchmark .\data\benchmark\shortchatbench_pilot.jsonl --output-dir .\artifacts\baseline_evaluation
