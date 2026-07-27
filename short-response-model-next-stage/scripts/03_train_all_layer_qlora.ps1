$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\scripts\train_dense_sft.py `
  --model-id Qwen/Qwen3-0.6B `
  --train-file .\data\train\train.jsonl `
  --dev-file .\data\dev\dev.jsonl `
  --output-dir .\artifacts\dense_sft `
  --selected-layers all `
  --qlora `
  --gradient-checkpointing `
  --max-steps 200 `
  --seed 11
