$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\scripts\validate_dataset.py
python .\scripts\check_leakage.py
python .\scripts\build_human_audit_packet.py
Write-Host "Review reports\HUMAN_DATA_AUDIT_PACKET.md before training."
