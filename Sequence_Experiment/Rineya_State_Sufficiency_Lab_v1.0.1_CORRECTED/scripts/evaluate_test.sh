#!/usr/bin/env bash
set -euo pipefail
[[ "${RINEYA_ALLOW_TEST:-}" == "I_UNDERSTAND_TEST_IS_SEALED" ]] || {
  echo "TEST is sealed. Freeze DEV before unlocking." >&2
  exit 2
}
python -m rineya_state_lab evaluate --config configs/eval/full_adapter_test.yaml
python -m rineya_state_lab evaluate --config configs/eval/split_adapter_test.yaml
python -m rineya_state_lab compare \
  --baseline outputs/eval/full_adapter_test/scored_predictions.jsonl \
  --candidate outputs/eval/split_adapter_test/scored_predictions.jsonl \
  --baseline-protocol direct_solve \
  --candidate-protocol generated_rollout \
  --metric score.correct_strict \
  --noninferiority-margin 0.02 \
  --output outputs/comparison/test_primary
