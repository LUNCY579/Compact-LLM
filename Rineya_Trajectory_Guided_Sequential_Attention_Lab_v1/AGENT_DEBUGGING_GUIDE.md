# TGSA Phase 1 developer and agent debugging guide

## First command

Run the doctor before starting expensive training:

```powershell
python .\scripts\phase1_doctor.py `
  --config .\configs\phase1_model_identifiability.json `
  --dataset-dir .\artifacts\phase1_full\dataset `
  --repo-root .. `
  --output .\artifacts\phase1_full\doctor.json
```

The doctor validates dataset hashes, parent-commit ancestry, exact history/delta initialization, parameter parity, finite forward/backward passes for all required arms, all trajectory interventions, and hook cleanup.

## Artifact map

```text
study/
  preflight.json
  matrix_manifest.json
  analysis.json
  analysis.md
  runs/
    <arm>/seed_<seed>/
      checkpoint_last.pt       # resumable; removed after success
      checkpoint_final.pt      # fixed endpoint
      train_log.jsonl
      predictions_dev__*.jsonl
      predictions_test__*.jsonl
      run_manifest.json
      failure_bundle.json      # only after failure
```

Every JSONL training record contains the exact example IDs and a batch signature. Use those fields to reproduce the first divergent update across arms or machines.

## Reproduce one paired seed

```powershell
python .\scripts\run_phase1_model_study.py `
  --config .\configs\phase1_model_identifiability.json `
  --dataset-dir .\artifacts\phase1_full\dataset `
  --output-dir .\artifacts\phase1_debug `
  --repo-root .. `
  --arms cascade,history_control,learned_delta `
  --seeds 101
```

This is diagnostic and not the full confirmatory matrix. Do not merge its scores with another output directory.

## Common failure paths

### Initialization audit fails

Do not train. Compare state-dictionary keys, dtypes, shapes, and the reported left/right-only keys. The history and learned-delta arm definitions must remain structurally identical. A parameter-count-only check is insufficient.

### Dataset manifest fails

Do not regenerate over the evidence directory. Preserve it, inspect the mismatched file and manifest hash, then materialize into a new directory. `--force` is for deliberate regeneration before evidence collection, not repair after outcomes are visible.

### Loss or gradient becomes non-finite

Open `failure_bundle.json` and locate the last step, batch signature, and example IDs. Re-run the same arm/seed with anomaly detection in a separate debug output directory. Do not delete the failed bundle or silently lower the learning rate under the same study ID.

### Arms see different batches

Compare `train_log.jsonl` records at the same paired seed and step. `batch_signature`, `example_ids`, `examples_seen`, and `tokens_seen` must match. Runtime RNG seeds may differ by arm for isolation, but dropout is fixed at zero and data order is arm-independent.

### Batch-roll has no effect

Inspect `intervention_records` in the learned-delta run manifest. A valid batch-roll trace should preserve input/output RMS closely while usually changing input-output cosine. If the records are empty, the model has no intercepted `TrajectoryEncoder`; if hooks remain after evaluation, the doctor fails cleanup.

### Resume is rejected

A checkpoint is resumed only when arm, seed, initialization seed, data-order seed, dataset hash, and config hash match. This prevents accidental continuation after code/config drift. Start a new output directory rather than bypassing the check.

## Agent rules

1. Never edit the registered config after reading confirmatory test results.
2. Never use dev score to choose a checkpoint; only the fixed final update is admissible.
3. Never summarize only aggregate accuracy. Preserve and analyze per-instance paired rows.
4. Never treat the smoke workflow as evidence; its config is explicitly non-confirmatory.
5. Never reinterpret a delta-versus-cascade win as finite-difference evidence unless delta also beats history control.
6. Never interpret zero/negate damage as alignment evidence; the batch-roll falsifier is required.
7. Preserve source commit, environment, manifests, checkpoints, logs, predictions, and failure bundles together.
8. Open a new study ID for any post-outcome change to data, architecture, training, or analysis.

## Useful integrity checks

```powershell
Get-FileHash .\artifacts\phase1_full\dataset\dataset_manifest.json -Algorithm SHA256
Get-Content .\artifacts\phase1_full\study\matrix_manifest.json | ConvertFrom-Json
Get-Content .\artifacts\phase1_full\study\analysis.json | ConvertFrom-Json
```

The frozen analyzer independently checks the prediction hashes referenced by every run manifest before computing confidence intervals.
