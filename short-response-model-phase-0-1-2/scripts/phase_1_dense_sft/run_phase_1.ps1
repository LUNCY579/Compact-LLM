$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

& .\scripts\phase_1_dense_sft\01_smoke_inference.ps1
& .\scripts\phase_1_dense_sft\02_evaluate_base.ps1
& .\scripts\phase_1_dense_sft\03_train_all_layer_qlora.ps1
& .\scripts\phase_1_dense_sft\04_evaluate_dense_sft.ps1
& .\scripts\phase_1_dense_sft\05_check_stop_gate.ps1

Write-Host "Phase 1 execution completed."
Write-Host "Review reports\PHASE_1_STOP_GATE.md and record the human gate decision."
