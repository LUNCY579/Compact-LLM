# TGSA Phase 1 v3 — operator and agent runbook

Every command fails closed. A non-zero exit means stop and read the error; it
never means "probably fine".

**Study:** `tgsa-phase1-identifiability-v3-2026-08-29`
**Branch:** `research/tgsa-phase1-identifiability-v3-powered-confirmatory`
**Registered config:** `configs/phase1_identifiability_v3.json`
**Registered matrix:** 3 arms x 20 seeds = 60 cells, 2,500 updates each.

## Setup (PowerShell)

```powershell
$ErrorActionPreference = "Stop"

$Branch       = "research/tgsa-phase1-identifiability-v3-powered-confirmatory"
$LaunchCommit = "<PINNED_LAUNCH_SHA>"      # fill from the frozen launch commit
$ParentCommit = "aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e"
$StudyId      = "tgsa-phase1-identifiability-v3-2026-08-29"

git fetch origin --prune
git checkout $Branch
git pull --ff-only origin $Branch

$Head = (git rev-parse HEAD).Trim()
if ($Head -ne $LaunchCommit) { throw "Wrong launch commit: $Head" }
git merge-base --is-ancestor $ParentCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Pinned parent is not an ancestor" }
if ((git status --porcelain).Length -ne 0) { throw "Launch checkout is dirty" }

Set-Location .\Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
python -m pip install -e ".[dev]"
python -m pytest .\tests -q --tb=short --durations=20

$Config       = (Resolve-Path ".\configs\phase1_identifiability_v3.json").Path
$Repo         = (Resolve-Path "..").Path
$EvidenceBase = "D:\tgsa_evidence"
$Root         = Join-Path $EvidenceBase (Join-Path $StudyId $LaunchCommit)
$Data         = Join-Path $Root "dataset"
$Output       = Join-Path $Root "study"

# Never reuse or delete an evidence root. Archive a prior one deliberately.
if (Test-Path $Root) { throw "Evidence root already exists: $Root" }
New-Item -ItemType Directory -Path $Root -Force | Out-Null
```

## 1. Doctor / preflight

```powershell
python .\scripts\prepare_phase1_data.py --config $Config --output-dir $Data

python .\scripts\phase1_doctor.py `
  --config $Config --dataset-dir $Data --repo-root $Repo `
  --output (Join-Path $Root "doctor.json")
```

## 2. Outcome-blind design audits (run and freeze BEFORE training)

```powershell
python .\scripts\phase1_task_difficulty_report.py `
  --config $Config --dataset-dir $Data `
  --output (Join-Path $Root "task_difficulty.json")

python .\scripts\phase1_power_simulation.py `
  --config $Config `
  --output (Join-Path $Root "power_simulation.json") `
  --primary-gate lower_bound_above_zero --trials 20000

python .\scripts\phase1_inference_calibration.py `
  --output (Join-Path $Root "inference_calibration.json") `
  --trials 20000 --seed-counts "5,10,16,20"
```

## 3. Train only — stops before any confirmatory test outcome exists

```powershell
python .\scripts\run_phase1_model_study.py `
  --config $Config --dataset-dir $Data --output-dir $Output --repo-root $Repo `
  --phase train
```

Writes `matrix_barrier.json`. No test prediction is produced by this phase.
Development predictions are written for debugging.

## 4. Matrix barrier verification

```powershell
python .\scripts\run_phase1_model_study.py `
  --config $Config --dataset-dir $Data --output-dir $Output --repo-root $Repo `
  --phase barrier
```

Confirms all 60 cells are trained, each to exactly 2,500 updates, with matching
checkpoint and train-log hashes and a single clean source head. Anything else
aborts before the confirmatory phase can begin.

## 5. Confirmatory evaluate only — the single frozen exposure

```powershell
python .\scripts\run_phase1_model_study.py `
  --config $Config --dataset-dir $Data --output-dir $Output --repo-root $Repo `
  --phase evaluate
```

**This is the point of no return for blinding.** Run it once, after step 4 passes.

## 6. Evidence-chain verification

```powershell
python .\scripts\verify_phase1_evidence.py `
  --config $Config --dataset-dir $Data --output-dir $Output --repo-root $Repo `
  --expect-confirmatory true `
  --report (Join-Path $Root "evidence_verification.json")
```

## 7. Frozen analysis

```powershell
python .\scripts\analyze_phase1_model_study.py `
  --config $Config --dataset-dir $Data --output-dir $Output --repo-root $Repo
```

Emits `analysis.json` / `analysis.md`, including `hypothesis_hierarchy` with the
preregistered interpretation, all four estimators, and the non-gating
difficulty-response and efficiency sections.

## 8. Focused single-example trajectory trace

```powershell
python .\scripts\phase1_trace_example.py `
  --config $Config --dataset-dir $Data --output-dir $Output --repo-root $Repo `
  --arm learned_delta --seed 101 --split test --index 0 --gradients `
  --report (Join-Path $Root "trace_learned_delta_101.json")
```

Also accepts `--example-id <exact id>`. Reports study ID, source SHA, arm, seed,
completed steps, dataset/config/checkpoint hashes, restored-state hash, resume
lineage, preserved failed attempts, per-head trajectory-control statistics,
per-block telemetry, and (with `--gradients`) gradient-group norms and finiteness.

Fails closed on an unknown example ID, a checkpoint hash mismatch, or a restored
state that disagrees with the recorded final state.

## Forbidden during confirmatory execution

`--force-data` (refused outright for a confirmatory study), `--overwrite`
combined with `--force-data`, `--allow-partial`, a reused output root, a dirty
checkout, and any inspection of test predictions between steps 3 and 5.

## What a failure gives you

Every failure path writes an immutable, content-addressed record under
`runs/<arm>/seed_<n>/failures/attempt_XXXX_<digest>.json` containing the study ID,
source SHA and dirty state, arm, seed, last completed step, last batch signature
and example IDs, exception type, message, and full traceback. Nothing is ever
overwritten or deleted, including across retries and resumes.

## Debugging index

| Question | Where |
|---|---|
| study ID, source SHA, arm, seed, step | `run_manifest.json`, any `failures/attempt_*.json`, trace `lineage` |
| dataset / config / checkpoint hashes | `run_manifest.json`, `evidence_verification.json`, trace `lineage` |
| last valid event before a crash | `failures/attempt_*.json` -> `last_step`, `last_batch_signature`, `last_example_ids` |
| gradient health | trace `--gradients` -> `gradient_health` |
| trajectory-control statistics | trace -> `trajectory_control_statistics`, `blocks[].summary` |
| resume lineage | `run_manifest.json` -> `training.resumed_from_step`, `training.train_log_repair` |
| evidence-chain state | `evidence_verification.json` -> `checks`, `errors` |
| why a gate failed | `analysis.json` -> `gates[].reason`, `hypothesis_hierarchy` |
| whether a conclusion is estimator-dependent | `analysis.json` -> `comparisons[].sensitivity_estimators` |
