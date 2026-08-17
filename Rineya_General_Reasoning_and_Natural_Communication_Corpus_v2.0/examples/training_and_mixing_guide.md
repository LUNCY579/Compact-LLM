# Training and Mixing Guide

## Recommended Rineya1.1 teacher mix

1. Start from the frozen accepted Rineya1 teacher checkpoint; do not overwrite it.
2. Use `data/sft/rineya_v2_recommended_combined_40000.jsonl`.
3. Append the local 4,410-row converted RAG training file once.
4. Shuffle deterministically and train one QLoRA epoch as the first controlled run.
5. Re-run the same 510-case chat DEV and official RAG DEV used for the baseline.
6. Compare category tails, not only aggregate means.
7. Freeze one SFT candidate before DPO or any sealed evaluation.

The result should contain 44,410 rows if the local RAG file still has exactly 4,410 valid records.

## PowerShell mixing example

```powershell
python .\scripts\mix_jsonl.py `
  --input .\data\sft\rineya_v2_recommended_combined_40000.jsonl `
  --input C:\PATH\TO\converted_rag_train_4410.jsonl `
  --output C:\PATH\TO\rineya1_1_dual_sft_train_44410.jsonl `
  --seed 20260816
```

## Re-run the deterministic semantic audit

```powershell
python .\scripts\semantic_audit.py --root .
```

## Measure target lengths with the exact local tokenizer

```powershell
python .\scripts\tokenizer_length_report.py `
  --sft .\data\sft\rineya_v2_recommended_combined_40000.jsonl `
  --tokenizer C:\PATH\TO\LOCAL\QWEN\SNAPSHOT `
  --output .\reports\exact_tokenizer_length_report.json
```

## Run the complete structural audit

```powershell
python .\scripts\semantic_audit.py --root .
python .\scripts\full_quality_audit.py `
  --root . `
  --eval-root C:\PATH\TO\short-response-model-phase-0-1-2
```

## Validate against current evaluation splits

```powershell
python .\scripts\validate_dataset.py `
  --sft .\data\sft\rineya_v2_recommended_combined_40000.jsonl `
  --dpo .\data\dpo\rineya_v2_preferences_train_12000.jsonl `
  --forbidden-eval-file C:\PATH\TO\short-response-model-phase-0-1-2\data\dev\dev.jsonl `
  --forbidden-eval-file C:\PATH\TO\short-response-model-phase-0-1-2\data\sealed\sealed.jsonl
```

## DPO

Use DPO only after the SFT candidate is accepted on DEV. Keep the SFT adapter immutable, train DPO as a new child artifact, and evaluate chat, reasoning, RAG supported, RAG abstention, and procedure correctness again.
