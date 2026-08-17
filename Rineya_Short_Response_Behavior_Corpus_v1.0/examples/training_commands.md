# Example Training Commands

Replace paths in angle brackets. These commands assume the package is copied into the Rineya repository root.

## 1. Phase 1 short-response SFT

Start with one epoch. Keep the original DEV split untouched.

```powershell
python .\scripts\phase_1_dense_sft\train_dense_sft.py `
  --model-id "<PATH_TO_BASE_OR_CURRENT_MODEL>" `
  --train-file ".\Rineya_Short_Response_Behavior_Corpus_v1.0\data\sft\rineya_short_response_combined_train.jsonl" `
  --dev-file ".\data\dev\dev.jsonl" `
  --output-dir ".\artifacts\rineya_behavior_sft_v1" `
  --selected-layers all `
  --target-module-group both `
  --qlora `
  --lora-rank 16 `
  --lora-alpha 32 `
  --lora-dropout 0.05 `
  --learning-rate 1e-4 `
  --epochs 1 `
  --per-device-batch-size 1 `
  --gradient-accumulation-steps 16 `
  --max-sequence-length 1024 `
  --gradient-checkpointing `
  --seed 20260816 `
  --run-label "rineya_short_response_behavior_v1"
```

The conservative `1e-4` starting rate is intentional for a behavior repair on an already capable checkpoint. Preserve the current accepted checkpoint and compare it against this run before increasing epochs or learning rate.

## 2. Phase 9 student LoRA SFT

The same combined SFT is valid because every row contains a complete `messages` list and omits `question`.

```powershell
python .\scripts\phase_9_student_distillation\train_student.py `
  --student "<PATH_TO_RINEYA_STUDENT_CHECKPOINT>" `
  --objective sft `
  --train-file ".\Rineya_Short_Response_Behavior_Corpus_v1.0\data\sft\rineya_short_response_combined_train.jsonl" `
  --dev-file ".\data\dev\dev.jsonl" `
  --output-dir ".\artifacts\rineya_student_behavior_lora_v1" `
  --max-sequence-length 512 `
  --max-completion-tokens 32 `
  --per-device-batch-size 1 `
  --gradient-accumulation-steps 32 `
  --learning-rate 1e-4 `
  --epochs 1 `
  --lora `
  --lora-rank 16 `
  --lora-alpha 32 `
  --lora-dropout 0.05 `
  --target-module-group both `
  --require-cuda `
  --seed 20260816
```

For the first behavior-only run, use `objective sft`; do not introduce teacher KL until the SFT frontier is measured.

## 3. Phase 3 DPO

Use the selected SFT adapter and the repository's existing preference DEV file.

```powershell
python .\scripts\phase_3_preference_alignment\train_dpo.py `
  --model-id "<PATH_TO_BASE_OR_MATERIALIZED_MODEL>" `
  --sft-adapter ".\artifacts\rineya_behavior_sft_v1\adapter" `
  --train-file ".\Rineya_Short_Response_Behavior_Corpus_v1.0\data\dpo\rineya_short_response_preferences_train.jsonl" `
  --dev-file ".\data\preferences\dev.jsonl" `
  --output-dir ".\artifacts\rineya_behavior_dpo_v1" `
  --max-steps 200 `
  --lora-rank 16 `
  --lora-alpha 32 `
  --lora-dropout 0.05 `
  --target-module-group both `
  --seed 20260816
```

## Required evaluation discipline

Do not train on:

```text
data/dev/dev.jsonl
data/sealed/sealed.jsonl
data/preferences/dev.jsonl
```

Evaluate the SFT candidate before DPO. Compare:

- median and P90 response length;
- complete-utterance rate;
- factual and arithmetic correctness;
- context recall/update integrity;
- ambiguity clarification;
- safety regression;
- roleplay adherence;
- original-specialization preservation;
- category worst-decile quality;
- adapter-dose frontier, including the current v2.3.1-relevant range around 0.8–1.2.
