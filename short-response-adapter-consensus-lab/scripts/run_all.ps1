param([switch]$RunSealed)
$ErrorActionPreference = "Stop"
$Lab = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = (Resolve-Path (Join-Path $Lab "..\.venv\Scripts\python.exe")).Path
Set-Location $Lab

function Invoke-LabPython {
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Lab command failed with exit code ${LASTEXITCODE}: $($Args -join ' ')"
    }
}

Invoke-LabPython -m pytest -q
Invoke-LabPython scripts\discover_candidates.py
Invoke-LabPython scripts\reproduce_baselines.py --limit 8
Invoke-LabPython scripts\run_smoke.py --config configs\hard_vote.yaml --dataset v22_dev --limit 16
Invoke-LabPython scripts\run_smoke.py --config configs\fusion_sweep.yaml --dataset v22_dev --committee-size 7 --limit 8
Invoke-LabPython scripts\run_smoke.py --config configs\negative_controls.yaml --dataset v22_dev --limit 8
Invoke-LabPython scripts\run_smoke.py --config configs\adaptive_consultation.yaml --dataset v22_dev --limit 16
Invoke-LabPython scripts\run_rag_smoke.py --limit 16
Invoke-LabPython scripts\analyze_pilots.py

if ($RunSealed) {
    if (-not (Test-Path artifacts\frozen_configs\frozen_final_config.json)) {
        Invoke-LabPython scripts\freeze_finalists.py --sealed-limit 32
    }
    Invoke-LabPython scripts\run_sealed.py --max-vram-gb 7.5
    Invoke-LabPython scripts\analyze_sealed.py
}
Invoke-LabPython scripts\snapshot_official_sources.py --label after
