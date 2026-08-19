#!/usr/bin/env bash
set -euo pipefail
python -m rineya_state_lab smoke-test --work-dir outputs/smoke --force
