$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

python .\scripts\phase_0_data\validate_dataset.py
python .\scripts\phase_0_data\check_leakage.py
python .\scripts\phase_0_data\build_human_audit_packet.py

Write-Host "Phase 0 automated checks completed."
Write-Host "Review reports\HUMAN_DATA_AUDIT_PACKET.md before Phase 1 training."
