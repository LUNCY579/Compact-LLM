#!/usr/bin/env bash
set -euo pipefail
: "${RINEYA_BASE_MODEL:?Set RINEYA_BASE_MODEL first}"
python -m rineya_state_lab train --config configs/train/split_stateful_lora.yaml
