$ErrorActionPreference = "Stop"
if ($env:RINEYA_ALLOW_TEST -ne "I_UNDERSTAND_TEST_IS_SEALED") {
  throw "TEST is sealed. Freeze DEV and set RINEYA_ALLOW_TEST=I_UNDERSTAND_TEST_IS_SEALED deliberately."
}
python -m rineya_state_lab evaluate --config configs/eval/full_adapter_test.yaml
python -m rineya_state_lab evaluate --config configs/eval/split_adapter_test.yaml
python -m rineya_state_lab compare `
  --baseline outputs/eval/full_adapter_test/scored_predictions.jsonl `
  --candidate outputs/eval/split_adapter_test/scored_predictions.jsonl `
  --baseline-protocol direct_solve `
  --candidate-protocol generated_rollout `
  --metric score.correct_strict `
  --noninferiority-margin 0.02 `
  --output outputs/comparison/test_primary
