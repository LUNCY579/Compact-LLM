$ErrorActionPreference = "Stop"
if (-not $env:RINEYA_BASE_MODEL) { throw "Set RINEYA_BASE_MODEL first." }
python -m rineya_state_lab evaluate --config configs/eval/full_adapter_dev.yaml
python -m rineya_state_lab evaluate --config configs/eval/split_adapter_dev.yaml
python -m rineya_state_lab compare `
  --baseline outputs/eval/full_adapter_dev/scored_predictions.jsonl `
  --candidate outputs/eval/split_adapter_dev/scored_predictions.jsonl `
  --baseline-protocol direct_solve `
  --candidate-protocol generated_rollout `
  --metric score.correct_strict `
  --noninferiority-margin 0.02 `
  --output outputs/comparison/dev_primary
