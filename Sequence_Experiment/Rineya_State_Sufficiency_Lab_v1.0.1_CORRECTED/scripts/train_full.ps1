$ErrorActionPreference = "Stop"
if (-not $env:RINEYA_BASE_MODEL) { throw "Set RINEYA_BASE_MODEL to the immutable base model path first." }
python -m rineya_state_lab train --config configs/train/full_context_lora.yaml
