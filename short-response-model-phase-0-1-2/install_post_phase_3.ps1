<#
.SYNOPSIS
Install and verify dependencies needed by Phases 4 through 8.

.DESCRIPTION
Run this inside the repository virtual environment after installing the
CUDA-enabled PyTorch wheel appropriate for your NVIDIA driver/CUDA stack.
The script intentionally verifies CUDA instead of silently accepting a CPU-only
PyTorch build, because the model-merging, pruning, profiling, and runtime phases
are intended for the target GPU.
#>
param(
    [switch]$AllowCpuOnly,
    [switch]$UpgradePip
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $PSScriptRoot).Path
Set-Location $Root

if ($UpgradePip) {
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
}

python -m pip install -r .\requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

python -c "import torch, transformers, peft, trl, datasets, rapidfuzz, jsonschema; print('torch', torch.__version__); print('transformers', transformers.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda)"
if ($LASTEXITCODE -ne 0) { throw "Dependency import verification failed." }

$CudaAvailable = python -c "import torch; print('true' if torch.cuda.is_available() else 'false')"
if (($CudaAvailable.Trim() -ne 'true') -and (-not $AllowCpuOnly)) {
    throw "PyTorch is not CUDA-enabled. Install the correct CUDA wheel, or rerun with -AllowCpuOnly only for static/CPU tests."
}

Write-Host "Post-Phase-3 environment verification completed."
