#!/usr/bin/env bash
set -euo pipefail
python -m rineya_state_lab data prepare \
  --archive assets/datasets/Rineya_Sequence_Independence_Dataset_v1.0_COMPLETE.zip \
  --output data/prepared
python -m rineya_state_lab data validate --data-root data/prepared --strict --output outputs/data_validation.json
