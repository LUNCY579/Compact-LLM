$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# Reuse an existing project virtual environment, or create one.
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating Python 3.11 virtual environment at .venv ..."
    py -3.11 -m venv .venv
}

$Python = Resolve-Path ".\.venv\Scripts\python.exe"

Write-Host "Upgrading pip build tools ..."
& $Python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing CUDA 12.8 PyTorch for the NVIDIA GPU ..."
& $Python -m pip install --upgrade torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing the broad short-response-model research stack ..."
& $Python -m pip install --upgrade -r ".\requirements.txt"

Write-Host "Checking dependency consistency ..."
& $Python -m pip check

Write-Host "Verifying PyTorch and CUDA ..."
& $Python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED')"

Write-Host ""
Write-Host "Installation finished."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
