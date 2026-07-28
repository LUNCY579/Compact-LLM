$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

python .\scripts\phase_1_dense_sft\check_phase_1_gate.py `
  --base-summary .\artifacts\baseline_evaluation\summary.json `
  --adapter-summary .\artifacts\dense_sft_evaluation\summary.json `
  --output-report .\reports\PHASE_1_STOP_GATE.md
