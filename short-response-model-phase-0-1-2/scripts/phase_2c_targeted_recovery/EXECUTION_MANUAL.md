# Phase 2C Execution Manual

## Purpose

Phase 2C starts from a human-approved Phase 2B reduced checkpoint. It ranks the surviving transformer layers, trains independent LoRA/QLoRA recovery candidates on controlled layer subsets, and compares their quality recovery with their trainable-state, training-memory, runtime, and adapter-size cost.

Each candidate starts from the same frozen Phase 2B checkpoint. Candidate adapters never accumulate into later candidates.

## Prerequisites

1. Phase 1 is complete.
2. Phase 2B completed and its stop gate contains `Human decision: PASS`.
3. The selected source profile (`balanced` or `min_memory`) is listed in `Approved profiles`.
4. The Phase 2B reduced checkpoint, catalog, candidate registry, and stop gate remain in place.
5. CUDA PyTorch, Transformers, PEFT, bitsandbytes, and repository requirements are installed.
6. The Phase 2C train and dev paths are sibling files named `train.jsonl` and `dev.jsonl`; their directory also contains `dataset_manifest.json`.
7. Sufficient free disk space remains for final adapters and evaluation evidence. Intermediate Trainer checkpoints are disabled by default in v1.4.7.

## Normal run

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -Phase2BCatalog .\artifacts\phase_2b_quality_v4_full_20260802_213049\model_catalog.json `
  -Phase2BRegistry .\artifacts\phase_2b_quality_v4_full_20260802_213049\candidate_registry.json `
  -Phase2BGate .\reports\PHASE_2B_quality_v4_full_20260802_213049_STOP_GATE.md `
  -OutputRoot .\artifacts\phase_2c_targeted_recovery_20260804 `
  -Report .\reports\PHASE_2C_TARGETED_RECOVERY_20260804_REPORT.md `
  -GateReport .\reports\PHASE_2C_TARGETED_RECOVERY_20260804_STOP_GATE.md `
  -QualityProfile balanced `
  -CalibrationLimit 64 `
  -EvalLimit 180 `
  -MaxSteps 200 `
  -BaseSeed 11 `
  -RepeatSeeds "23,37" `
  -RepeatTopK 2 `
  -LoraRank 16 `
  -LoraAlpha 32 `
  -LearningRate 0.0002 `
  -QLoRA `
  -GradientCheckpointing
```

The gate compares proportional training-resource savings with combined proportional quality opportunity loss. Quality dimensions, tail/category behavior, issue cost, output deviation, and target NLL contribute continuously. No absolute score, pass-rate, issue-count, or NLL floor independently controls promotion.

## Resume an interrupted v1.4.7 run

Use the exact original command and append:

```powershell
-Resume
```

Resume is end-to-end:

- completed adapters are validated and reused;
- partial generation resumes from durable stable-ID rows;
- teacher-forced NLL resumes from durable per-case rows;
- corrupted or incomplete JSON sentinels are quarantined and rebuilt;
- output comparisons are regenerated when their input-result hashes changed;
- transient Windows destination locks are retried;
- reconstructible progress sidecars cannot abort completed scientific work.

Do not manually invoke child evaluators after installing v1.4.7.

## Resume an existing v1.4.6 Phase 2C run

v1.4.6 generated deterministic reasoning cases without explicit correctness contracts. Under Evaluator v4.5, fluent but incorrect reasoning replies could therefore receive high scores. v1.4.7 performs a narrowly authorized contract-only migration.

Run the same Phase 2C wrapper command with `-Resume`. The wrapper will:

1. rebuild the Phase 2C dataset with explicit prompt-grounded contracts;
2. verify that stable IDs, visible messages, and assistant training targets are unchanged;
3. update only the approved dataset hashes in `run_config.json` and record a migration audit;
4. reuse every valid completed adapter;
5. rescore durable generated replies without regenerating them;
6. continue partial generation from the next missing stable ID;
7. reuse teacher-forcing rows because model inputs and targets did not change;
8. rebuild summaries and output-deviation evidence before candidate selection.

The migration is rejected if any model, message, assistant target, benchmark, seed, LoRA setting, ranking weight, or other decision-bearing input changed.

Do not delete the existing Phase 2C output root. Do not restart from scratch merely because the dataset hashes changed from v1.4.6 to v1.4.7.

## Windows file-lock behavior

The shared atomic writer retries transient access, sharing, and lock violations instead of terminating after one failed `os.replace`. Defaults can be overridden only for troubleshooting:

```powershell
$env:SRM_ATOMIC_REPLACE_ATTEMPTS = "60"
$env:SRM_ATOMIC_REPLACE_INITIAL_DELAY = "0.05"
$env:SRM_ATOMIC_REPLACE_MAX_DELAY = "0.50"
```

Persistent filesystem failures still raise an error. Close editors, Explorer previews, sync clients, or antivirus exclusions that continuously hold artifact files open.

## Disk-space and checkpoint policy

Phase 2C disables intermediate Trainer checkpoints by default and safely stages the final adapter before publication. The default pre-training free-space requirement is 1 GB.

Adjust it with:

```powershell
-MinimumFreeDiskGB 2
```

Retain intermediate checkpoints only when deliberately needed:

```powershell
-KeepTrainingCheckpoints
```

Keeping checkpoints can substantially increase disk use across the candidate-and-seed matrix.

## Alternate isolated run

Use unique output/report paths so the existing run remains immutable:

```powershell
.\scripts\phase_2c_targeted_recovery\run_phase_2c.ps1 `
  -Phase2BProfile balanced `
  -OutputRoot .\artifacts\phase_2c_targeted_recovery_v2 `
  -Report .\reports\PHASE_2C_V2_REPORT.md `
  -GateReport .\reports\PHASE_2C_V2_STOP_GATE.md `
  -MaxSteps 300 `
  -RepeatSeeds "23,37,49" `
  -QLoRA `
  -GradientCheckpointing
```

A changed decision-bearing configuration requires a new output root. Do not force a mismatched run into an existing root.

## Clean restart

A clean restart is appropriate only when intentionally discarding the previous Phase 2C experiment:

```powershell
Remove-Item -Recurse -Force .\artifacts\phase_2c_targeted_recovery -ErrorAction SilentlyContinue
Remove-Item -Force .\reports\PHASE_2C_TARGETED_RECOVERY_REPORT.md, .\reports\PHASE_2C_STOP_GATE.md -ErrorAction SilentlyContinue
```

Never delete Phase 2B artifacts required by the selected source profile.

## Review sequence

1. Inspect `run_config.json` and any `RUN_CONFIG_CONTRACT_MIGRATION_v1.4.7.json`.
2. Inspect `importance/combined_layer_ranking.json`; distinguish current reduced-layer indices from original Qwen layer indices.
3. Inspect all `candidate_result.json` and `OUTPUT_DEVIATION.md` files.
4. Review arithmetic, implication, ordering, comparison, contradiction, set, temporal, clarification, context, factual, and safety cases.
5. Confirm objectively wrong reasoning outputs now receive explicit contract penalties.
6. Confirm each recommended selective profile has repeated-seed evidence and same-seed `all_remaining` controls.
7. Compare trainable parameters, peak training VRAM, runtime, and adapter bytes against `all_remaining`.
8. Inspect common ShortChatBench profile-validation outputs before human approval.
9. Keep each winning adapter paired with the exact reduced checkpoint recorded in the catalog.

The `legacy_heuristic_score` is a historical surface diagnostic only. It does not represent the Phase 2B baseline and does not select Phase 2C profiles.

## Audit completed artifacts

```powershell
python .\scripts\phase_2c_targeted_recovery\audit_phase_2c_artifacts.py `
  --artifact-root .\artifacts\phase_2c_targeted_recovery_20260804 `
  --json-out .\artifacts\phase_2c_targeted_recovery_20260804\artifact_audit.json
```

The audit validates candidate identities, complete adapters, generation/NLL case coverage, unique stable IDs, output comparisons, profile eligibility, repeated seeds, profile-validation summaries, and catalog/registry consistency.

## Approve profiles

Edit only the human fields in the generated Phase 2C stop gate:

```text
- Human decision: `PASS`
- Approved profiles: `maximum_quality,balanced_training`
```

The resolver rejects profiles that are not human-approved, automated-eligible, technically valid, and repeated-seed stable.

## Continue to Phase 3

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 `
  -Phase2CProfile balanced_training
```

The Phase 3 branch keeps the exact reduced model and selected Phase 2C adapter together and does not overwrite the Phase 2B-only branch.
