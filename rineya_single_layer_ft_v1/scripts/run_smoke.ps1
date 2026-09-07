$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $Root ".venv\Scripts\python.exe"))) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Lab = Join-Path $Root "rineya_single_layer_ft_v1\scripts"
Set-Location $Root

Write-Host "== build puzzle view =="
& $Py (Join-Path $Lab "build_puzzle_view.py")
Write-Host "== train layer-8 LoRA =="
& $Py (Join-Path $Lab "train_layer_lora.py")
Write-Host "== puzzle eval baseline =="
& $Py (Join-Path $Lab "eval_puzzle_val.py") --condition baseline
Write-Host "== puzzle eval layer8_lora =="
& $Py (Join-Path $Lab "eval_puzzle_val.py") --condition layer8_lora
Write-Host "== ablation baseline =="
& $Py (Join-Path $Lab "rerun_layer8_ablation.py") --condition baseline
Write-Host "== ablation layer8_lora =="
& $Py (Join-Path $Lab "rerun_layer8_ablation.py") --condition layer8_lora
Write-Host "== summary =="
& $Py (Join-Path $Lab "write_smoke_summary.py")
Write-Host "done"
