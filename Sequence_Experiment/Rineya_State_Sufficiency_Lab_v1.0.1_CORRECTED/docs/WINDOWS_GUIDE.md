# Windows 11 Guide

## Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
```

Install a CUDA-compatible PyTorch build appropriate for your driver before the generic training dependencies when necessary.

## Dataset

```powershell
.\scripts\prepare_data.ps1
python -m rineya_state_lab data validate --data-root data/prepared --strict
```

## Base model

```powershell
$env:RINEYA_BASE_MODEL = "C:\Users\you\models\rineya-base"
python -m rineya_state_lab doctor --model-path $env:RINEYA_BASE_MODEL --data-root data/prepared --require-cuda
```

Quote local paths containing spaces. YAML paths should be enclosed in quotes and use either doubled backslashes or forward slashes.

## Sequential adapter training

```powershell
.\scripts\train_full.ps1
.\scripts\train_split.ps1
```

Close other GPU-heavy applications. On an 8 GB GPU, reduce `max_length`, use QLoRA configs, or increase gradient accumulation rather than allowing OOM retries to change experimental settings silently.

## DEV and TEST

```powershell
.\scripts\evaluate_dev.ps1
```

Only after freezing the protocol:

```powershell
$env:RINEYA_ALLOW_TEST = "I_UNDERSTAND_TEST_IS_SEALED"
.\scripts\evaluate_test.ps1
```
