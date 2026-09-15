# MQ operational successor runbook

Current decision authority: [MODEL_QUALITY_STATUS.md](../../MODEL_QUALITY_STATUS.md). Historical instructions are [archived](../../docs/history/model-quality-audit-2026-09-15/experiments/model_quality_gpu_v1/RUNBOOK.md).

## Scope

Use a clean full checkout of **`codex/model-quality-audit-v2`**, descended from `8c3ed88a821b83e531ab682b3201879ea4fc928b`, and the separately pinned MQ runtime. This branch changes operational supervision only. It has no official-source native qualification or trained candidate of its own. Historical MQ reviews/qualifications cannot be transplanted to this changed commit.

The six-cell design, source lock, FP32 precision, masks/normalization, optimizer and resource minima remain fixed. The public supervisor is a foreground process; no retry loop or asynchronous launch is supported. Do not use archived shell wrappers or the internal worker flag.

## Before native execution

1. Stage exact local Supa assets matching `source_lock.json`; no network/model download or source substitution is automatic. Source parameters: 596,049,920; trainable tail: 314,618,880 parameters / 220 tensors. Opallo and gatedA1.1 are different sources.
2. Prepare an independently reviewed bank and evaluator. The historical v3 pool is not the proposed complete quality bank. QTB-001 and a larger DEV contract need a separately registered scientific successor; this code retains the64-row DEV cap. Never approve an unaudited manifest to get past a gate.
3. Measure real current host/GPU/disk headroom and other model jobs. The cooperative lease cannot protect against old runners. Do not evict processes, lower guards or delete checkpoints for space. Provision adequate storage instead.
4. Training preload minima are6 GiB host / 6 GiB GPU; qualification4.6 GiB host / 3 GiB GPU. These are refusal thresholds, not maximum-shape fit certificates. Qualification includes backward/gradient checks but no optimizer step. Post-load guards remain1.5 GiB host / 0.5 GiB GPU.
5. Record source/runtime/code, representative and maximum planned shapes, and actual peak resources. The old 20-token qualification is insufficient for a changed long or maximum-shape plan. Complete bounded official-source full-state resume validation before relying on restart.

## Public commands

Run from this checkout with the dedicated Python executable. Absolute paths below are illustrative and must identify immutable existing assets or **new** output locations.

```text
python -m pytest experiments/model_quality_gpu_v1/tests -q
python -m experiments.model_quality_gpu_v1 environment
python -m experiments.model_quality_gpu_v1 prepare --source <source> --train <train.jsonl> --dev <dev.jsonl> --excluded <exclusions.jsonl> --selection-seed 17 --order-seed 17 --out <new-preparation>
```

Preparation compiles full native token IDs, response/EOS masks, selected rows, order and update groups. An independent reviewer must bind the exact plan hash in a new review JSON after rights/labels/semantics/generator checks. Preserve pending/rejected reviews.

```text
python -m experiments.model_quality_gpu_v1 qualify --source <source> --plan <plan.json> --review <review.json> --out <new-qualification>
```

This is zero-update native qualification. Inspect launch, worker log, supervisor status, binding/resource snapshots and qualification metrics. A test fixture or synthetically constructed envelope is insufficient. Qualification and training must share the same clean commit, source, plan, review, code and runtime; commit documentation changes before qualification, not between qualification and training.

Only when the independently registered scientific prerequisites are satisfied, select and inspect **one** cell at a time:

```text
python -m experiments.model_quality_gpu_v1 train --source <source> --plan <plan.json> --review <review.json> --qualification <qualification.json> --arm whole_1e-5 --confirm-training --out <new-attempt>
```

The supervisor creates a new output directory and owns the watchdog and child context. Direct worker bypass is rejected. Preserve `launch.json`, `worker.log`, `supervisor.json`, `binding.json`, result/failure, raw baseline/final outputs and step logs and **every checkpoint directory**. Process startup failures remain diagnosed; existing output paths are rejected rather than reused. No wrapper automatically cleans weights or moments.

## Resume and analysis

Use only a complete, immutable checkpoint with verified manifest and shards, source/plan/arm/code/runtime binding and cleared-gradient cursor. Resume into a **new** attempt using the same public train command plus `--resume <complete-checkpoint-directory>`. Never invent a missing checkpoint or remove a possibly live lease. Inspect a stale PID/lock and arrange its removal separately only after its process is known gone. Historical MQ checkpoint directories are absent, so those records cannot currently be resumed.

```text
python -m experiments.model_quality_gpu_v1.analyze <whole-attempt> <response-attempt> --out <new-comparison.json>
```

Analyze one recorded attempt per arm, with explicit failed/retry accounting; compare identical source/plan/runtime/code bindings. Report processed and loss-bearing tokens separately. Raw DEV tradeoffs are not automatic promotion, statistical confirmation or release permission. The new corpus, proposed 512-row DEV, independent training draws and confirmation need a separate prospective study; this operational patch does not silently enable them.
