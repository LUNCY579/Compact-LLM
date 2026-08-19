# Operational Runbook

## 1. Verify release

```bash
python tools/verify_release.py .
```

## 2. Create environment and install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train,dev]"
```

## 3. Prepare and validate data

```bash
python -m rineya_state_lab data prepare \
  --archive assets/datasets/Rineya_Sequence_Independence_Dataset_v1.0_COMPLETE.zip \
  --output data/prepared
python -m rineya_state_lab data validate --data-root data/prepared --strict
```

## 4. Run no-model smoke test

```bash
python -m rineya_state_lab smoke-test --work-dir outputs/smoke --force
```

Expected reference strict accuracy is 100% for direct solve, oracle transition, and generated rollout.

## 5. Freeze base model and run training smoke

Set `RINEYA_BASE_MODEL`, then override both configs to two steps:

```bash
python -m rineya_state_lab train --config configs/train/full_context_lora.yaml \
  --set training.max_steps=2 --set data.max_train_examples=64 --set data.max_eval_examples=32 \
  --set run.output_dir=outputs/train_smoke/full
python -m rineya_state_lab train --config configs/train/split_stateful_lora.yaml \
  --set training.max_steps=2 --set data.max_train_examples=64 --set data.max_eval_examples=32 \
  --set run.output_dir=outputs/train_smoke/split
```

Inspect both manifests and adapter files before full training.

## 6. Train primary arms

Run FULL and SPLIT sequentially with the frozen configs. Do not change one arm without making the same non-treatment change to the other.

## 7. Evaluate DEV

Run both DEV configs. Audit parser disagreements and freeze all policies. Choose and document the non-inferiority margin.

## 8. Run ablations

Train ordered and next-step-only SPLIT arms. Add matched-token and one-transition-per-problem controls if the primary result is close or positive.

## 9. Unlock TEST once

Set the explicit environment token and run the two TEST configs. Archive outputs read-only.

## 10. Compare and report

Use strict paired correctness as primary, lenient correctness as sensitivity analysis, and include all subgroup/failure reports.
