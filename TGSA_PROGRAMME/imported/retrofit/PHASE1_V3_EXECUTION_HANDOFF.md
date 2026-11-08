# Phase-1 v3 confirmatory execution — handoff

This file lives on the retrofit branch on purpose. The v3 execution worktree
must stay **byte-clean** while the matrix runs, because the runner records
`git.dirty` into every run manifest and the evidence verifier fails a
confirmatory run produced from a dirty checkout. Writing anything into that
worktree mid-flight would invalidate every cell that completes afterwards.

## Launch record

| item | value |
|---|---|
| branch | `research/tgsa-phase1-identifiability-v3-powered-confirmatory` |
| design-freeze commit | `04d3c02b6f9c7398fede8416deea81ce170ea7c2` |
| design-freeze tag | `tgsa-phase1-v3-launch-2026-08-29` |
| execution commit | `72395b69beb96dc71dc80bc8c73ef75205ea51a1` |
| v2 frozen parent | `e0063f0e784da69a33a7669956ff0ac13daa2cd6` (tag `tgsa-phase1-v2-launch-ready-frozen`) |
| Phase-0 ancestor | `aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e` |
| CI run | 33271024242, **success**, confirmatory job skipped |
| study id | `tgsa-phase1-identifiability-v3-2026-08-29` |
| config SHA-256 | `9dadb157b04fefad6a4d1ed8cfee1d5ed23916e4c12e13f2dce61b097a7b8163` |
| preregistration SHA-256 | `6cfa608a510672181a847241b4b6e91fabe8d12032071c346230efb456361085` |
| dataset manifest SHA-256 | `cad5851eeba887584c9c6aa38cef11f00d752c71fb2c71c6877cdc44a3e52d54` |
| environment | Python 3.11.15, torch 2.13.0+cpu, numpy 2.4.6, CPU-only, 32 cores |

Full record: `evidence/v3_design/launch_record.json` on the v3 branch.

## Where execution stopped

| stage | state |
|---|---|
| PREPARE | **done** — dataset materialized and hashed |
| TRAIN | **in flight** — sharded across 8 processes |
| BARRIER | not started |
| CONFIRMATORY EVALUATE | not started |
| EVIDENCE VERIFY | not started |
| ANALYZE | not started |

**No confirmatory test prediction exists, and none has been inspected.** The
train phase writes development predictions only; test predictions are produced
solely by the evaluate phase, which runs only after the barrier passes.

Evidence root: `%TEMP%\claude\...\scratchpad\v3full\` (`dataset/`, `study/`, `logs/`).

## Sharding, and why it is not a design change

The 60 registered cells are independent: each builds its own model from its own
derived seeds and writes its own run directory. Training was therefore sharded
across processes with `--phase train --arms <arm> --seeds <seed>`. No registered
arm, seed, step budget, data order, or estimator is affected, and the barrier
still requires all 60 cells before any confirmatory evaluation.

Each shard invocation rewrites `preflight.json` for *its* seed only. The
unrestricted `--phase barrier` pass below regenerates the complete preflight
covering all 20 registered seeds, which is what the verifier checks.

Measured cost: ~218 ms/step (cascade), ~288 (history_control), ~258
(learned_delta) solo. All 32 cores saturate at 8-way concurrency, so extra
parallelism does not help. Total ≈ 8 h wall clock for 150,000 steps.

## Resuming and completing

Run from the v3 worktree with a **clean tree**.

```powershell
$ErrorActionPreference = "Stop"
Set-Location C:\Users\clclc\BusinessProject\Chat-Bubble-tgsa-launch-audit
if ((git status --porcelain).Length -ne 0) { throw "worktree is dirty; every cell completed now would record dirty=true" }
$Head = (git rev-parse HEAD).Trim()
if ($Head -ne "72395b69beb96dc71dc80bc8c73ef75205ea51a1") { throw "wrong execution commit: $Head" }

Set-Location .\Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
$Config = ".\configs\phase1_identifiability_v3.json"
$Root   = "$env:TEMP\claude\C--Users-clclc-BusinessProject-Chat-Bubble\8b037530-92dc-40fa-93b0-7d2d5ea2749e\scratchpad\v3full"
$Data   = Join-Path $Root "dataset"
$Study  = Join-Path $Root "study"
$Repo   = (Resolve-Path "..").Path
```

Any interrupted cell resumes from `checkpoint_last.pt` (written every 250
steps); the train log is truncated to the checkpoint boundary automatically, and
that path is covered by `test_interrupted_and_uninterrupted_training_are_equivalent`.

```powershell
# 1. finish training (idempotent; completed cells short-circuit)
python .\scripts\run_phase1_model_study.py --config $Config --dataset-dir $Data `
  --output-dir $Study --repo-root $Repo --phase train

# 2. barrier - regenerates the full preflight and verifies all 60 cells
python .\scripts\run_phase1_model_study.py --config $Config --dataset-dir $Data `
  --output-dir $Study --repo-root $Repo --phase barrier

# 3. THE SINGLE CONFIRMATORY EXPOSURE - run once, only after step 2 passes
python .\scripts\run_phase1_model_study.py --config $Config --dataset-dir $Data `
  --output-dir $Study --repo-root $Repo --phase evaluate

# 4. evidence verification
python .\scripts\verify_phase1_evidence.py --config $Config --dataset-dir $Data `
  --output-dir $Study --repo-root $Repo --expect-confirmatory true `
  --report (Join-Path $Root "evidence_verification.json")

# 5. frozen analysis
python .\scripts\analyze_phase1_model_study.py --config $Config --dataset-dir $Data `
  --output-dir $Study --repo-root $Repo
```

## Reading the result without spinning it

`analysis.json` → `hypothesis_hierarchy.interpretation` states the registered
reading directly, so it cannot be renegotiated afterwards:

| outcome | registered interpretation |
|---|---|
| H1a + H1b + H1c | trajectory-specific, example-aligned information adds capability |
| H1b only | an extra learned recurrent pathway helps — **not** trajectory-specific |
| H1a + H1b, not H1c | parameterization may help; aligned content **not** shown causal |
| interval includes zero | **inconclusive, not success** |
| negative | evidence **against** the mechanism in this setting |

Then follow the matching Track C case. Do not force a TGSA-positive reading.

If the realised seed spread is at the elevated end (`sigma_a ≈ 0.04`), a null at
+0.020 is **underpowered**, not evidence against the mechanism: measured power
there is 0.492 at 20 seeds. Report it that way. Do **not** change the registered
seed count after the fact — high realised variance is an interpretation issue,
not permission to alter the experiment.

## Do not

- write into the v3 worktree while cells are running;
- inspect `predictions_test__*.jsonl` before step 5;
- rerun `--phase evaluate` to "check" an early cell;
- use `--force-data`, `--allow-partial`, or a reused output root;
- change thresholds, seeds, or the estimator after outcomes are visible.
