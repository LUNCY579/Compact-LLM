param(
    [string]$Python = "py -3.11",
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Venv)) {
    Invoke-Expression "$Python -m venv `"$Venv`""
}
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
Write-Host "Install the CUDA-enabled PyTorch build appropriate for your machine now if needed."
& "$Venv\Scripts\python.exe" -m pip install -e ".[dev]"
& "$Venv\Scripts\python.exe" scripts\validate_install.py
& "$Venv\Scripts\python.exe" -m pytest -q
