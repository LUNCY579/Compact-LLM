# Rineya Short-Response Behavior Corpus v1.0

A train-only, trainer-compatible dataset package for Rineya.

## Recommended files

| Use | File |
|---|---|
| Phase 1 or Phase 9 behavior SFT | `data/sft/rineya_short_response_combined_train.jsonl` |
| New behavior only ablation | `data/sft/rineya_short_response_new_train.jsonl` |
| Preservation-only control | `data/sft/rineya_short_response_preservation_slice.jsonl` |
| Phase 3 preference alignment | `data/dpo/rineya_short_response_preferences_train.jsonl` |

The combined SFT contains **10,000 new rows + 2,000 preservation rows**. The DPO file contains **5,000 pairs**.

## Hard guarantees

- JSONL: one object per line.
- Every SFT row ends in the assistant target.
- Every SFT row is marked `split: train`.
- No top-level `question` field is present.
- No DEV or SEALED row is included.
- The combined file has 12,000 unique rendered prompts.
- Automated validation and leakage checks pass.
- Human review remains pending.

## Validate

```powershell
python .\scripts\validate_dataset.py `
  --sft .\data\sft\rineya_short_response_new_train.jsonl `
  --sft .\data\sft\rineya_short_response_preservation_slice.jsonl `
  --sft-schema .\schemas\sft_record.schema.json `
  --dpo .\data\dpo\rineya_short_response_preferences_train.jsonl `
  --dpo-schema .\schemas\dpo_record.schema.json
```

From inside the Rineya repository, add the untouched evaluation files to repeat leakage checks:

```powershell
python .\Rineya_Short_Response_Behavior_Corpus_v1.0\scripts\validate_dataset.py `
  --sft .\Rineya_Short_Response_Behavior_Corpus_v1.0\data\sft\rineya_short_response_combined_train.jsonl `
  --sft-schema .\Rineya_Short_Response_Behavior_Corpus_v1.0\schemas\sft_record.schema.json `
  --dpo-schema .\Rineya_Short_Response_Behavior_Corpus_v1.0\schemas\dpo_record.schema.json `
  --forbidden-eval-file .\data\dev\dev.jsonl `
  --forbidden-eval-file .\data\sealed\sealed.jsonl
```

## Training

See `examples/training_commands.md`. Keep the repository DEV/SEALED and preference DEV files untouched.

## Review evidence

- `DATA_CARD.md`
- `reports/validation_report.json`
- `reports/leakage_report.json`
- `reports/statistics.md`
- `reports/sample_audit_packet.md`
- `manifests/dataset_manifest.json`
- `manifests/checksums.sha256`
