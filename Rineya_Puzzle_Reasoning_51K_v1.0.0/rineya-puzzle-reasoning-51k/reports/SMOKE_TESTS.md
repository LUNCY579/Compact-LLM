# Release Smoke Tests

All checks below were executed against the packaged release.

## Symbolic validation

- Status: **PASS**
- Rows independently re-solved: **51,000**
- Failures: **0**
- Unique IDs: **51,000**
- Unique prompts: **51,000**
- Unique structures: **51,000**

## Prose lint

- Status: **PASS**
- Rows checked: **51,000**
- Findings: **0**
- Checks: singular/plural agreement, common article errors, one/remain agreement, duplicate words, double spaces, and control characters.

## Training-view builders

The `think_then_short`, `short_only`, `dual_fields`, and `preference_pair` builders each produced the expected 75 rows from `examples/sample_75.jsonl` and parsed successfully as JSONL.

## Evaluator oracle test

The evaluator was given the canonical answer for every row in the frozen test split.

- Accuracy: **100.0%**
- Coverage: **100.0%**
- Correct / total: **4,080 / 4,080**
- Missing predictions: **0**
- Duplicate prediction IDs: **0**

This is a software sanity check, not a model result.

## Clean-room regeneration

A separate `--tiny` corpus was generated from scratch (120 rows: two rows per family in every split), then passed symbolic validation, prose linting, and short-only training-view conversion. This checks that custom-size regeneration works independently of the canonical 51K layout.
