$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

python .\scripts\phase_1_dense_sft\smoke_inference.py `
  --model-id Qwen/Qwen3-0.6B `
  --output-json .\artifacts\phase_1_smoke\smoke_result.json
