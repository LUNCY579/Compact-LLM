# Phase 1: model-backed trajectory identifiability

## Purpose

Phase 0 established that the trajectory-guided sequential-attention (TGSA) operator is differentiable, mask-correct, instrumented, and expressible through the registered ablation family. It did **not** show that a trained model gains capability from the trajectory signal.

Phase 1 asks the narrowest model-backed causal question that should be answered before a Qwen/GQA retrofit:

> When the sequential cascade, optimizer, examples, update count, initialization, and learned parameter budget are held fixed, does explicitly representing the finite difference between consecutive head carriers improve held-out capability beyond an information-matched history representation?

This study deliberately remains smaller and more controlled than the later natural-language and sealed long-context phases. A positive result advances the mechanism; it does not establish a Qwen-scale result.

## Why `history_control` is the primary baseline

At sequential head index `j`, define the current and prior carrier states as `c_j` and `c_{j-1}`.

- trajectory arm input: `d_j = c_j - c_{j-1}`
- history control input: `m_j = (c_j + c_{j-1}) / 2`

The current carrier `c_j` is already present in the gate input in both arms. Therefore the history-control arm can recover the same two-state information:

- `c_{j-1} = 2m_j - c_j`
- `d_j = 2(c_j - m_j)`

`history_control` and `learned_delta` also instantiate the same trajectory-encoder structure and parameter count. The runner proves their full initial state dictionaries are byte-identical for every paired seed. This makes the contrast substantially stronger than comparing trajectory guidance only with a smaller cascade arm: it tests whether the finite-difference coordinate itself is a useful inductive bias, not merely whether an extra learned pathway helps.

## Registered contrasts

### Primary causal contrast

`learned_delta - history_control` on held-out `pointer_chase` and `state_update` strata.

### Required supporting contrast

`learned_delta - cascade` on the same held-out strata. This establishes whether trajectory guidance adds capability beyond sequential recurrence alone, while the history contrast controls additional learned capacity.

### Alignment falsifier

Evaluate the trained `learned_delta` model after rolling every trajectory-control tensor by one example along the batch dimension. The distribution, shape, sequence positions, and model weights are preserved, but each example receives another example's control signal. No token is moved across positions, so this intervention cannot create future-token leakage.

A trajectory interpretation requires aligned inference to outperform this batch-rolled intervention. Zero and sign-negated interventions are preserved as diagnostic sensitivity checks, but they are not substitutes for the distribution-preserving alignment falsifier.

### Guardrail

`direct_lookup` must remain non-inferior to `history_control`. This detects a model that improves iterative tasks only by damaging ordinary one-step retrieval.

## Capability suite

All examples are generated once, materialized to JSONL, and hash-locked before any arm is trained.

- **Pointer chasing:** follow a randomly ordered key-to-key mapping for multiple hops.
- **State update:** compose a sequence of signed modular state transitions.
- **Direct lookup:** retrieve one mapped value; this is the short-horizon guardrail.

Development and test splits use longer chains and larger contexts than training. Latent task signatures are rejected across splits, and every prediction keeps the instance ID, family, exact stratum, seed, arm, and intervention.

These are controlled synthetic mechanism tasks. They are not a sealed natural-language benchmark.

## Registered full study

The confirmatory configuration is `configs/phase1_model_identifiability.json`.

- arms: `cascade`, `history_control`, `learned_delta`
- paired seeds: `101, 202, 303, 404, 505`
- train/dev/test examples: `5,000 / 900 / 1,800`
- model: two decoder blocks, width 96, six sequential heads, trajectory rank 16
- training: 2,500 fixed AdamW updates per arm/seed, batch 64, dropout 0
- selection: final fixed-step checkpoint only; no early stopping
- interventions: zero, negate, and batch-roll
- analysis: hierarchical paired bootstrap over seeds and within-stratum examples

The smoke configuration is intentionally tiny and marked `confirmatory: false`. Its analysis decision is always `smoke_only`, regardless of scores.

## Advancement gates

The frozen analyzer requires all of the following:

1. the full registered arm-by-seed matrix is present and hash-valid;
2. learned delta exceeds history control by at least 0.02 macro accuracy, with a 95% paired-bootstrap lower bound above zero;
3. learned delta exceeds cascade by at least 0.02, with the lower bound above zero;
4. aligned learned delta exceeds batch-rolled learned delta by at least 0.015, with the lower bound above zero;
5. the 95% lower bound for direct-lookup delta versus history is at least -0.01;
6. exact initialization, parameter-parity, lineage, and artifact-integrity checks pass.

The possible decisions are:

- `supports_phase1_mechanism_hypothesis`
- `inconclusive`
- `contradicted`
- `smoke_only`

## Evidence outputs

Each run writes:

- fixed final checkpoint and SHA-256 hash;
- run manifest with branch lineage, environment, seeds, parameter groups, update count, examples/tokens seen, and checkpoint policy;
- per-step loss, gradient norms, batch IDs, and batch signatures;
- per-example dev/test predictions for every registered condition;
- pre/post-training TGSA telemetry on a fixed debug batch;
- intervention RMS/cosine traces;
- resumable intermediate checkpoint;
- failure bundle with traceback, last step, last batch signature, and example IDs.

The matrix writes a preflight proof of exact paired initialization. The analyzer revalidates all recorded prediction hashes before computing a result and records its own source hash.

## Windows PowerShell workflow

From the repository root:

```powershell
git fetch origin
git checkout research/tgsa-phase1-model-identifiability
cd .\Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
python -m pip install -e ".[dev]"

$Config = (Resolve-Path ".\configs\phase1_model_identifiability.json").Path
$Data = Join-Path $PWD "artifacts\phase1_full\dataset"
$Output = Join-Path $PWD "artifacts\phase1_full\study"
$Repo = (Resolve-Path "..").Path

python .\scripts\prepare_phase1_data.py `
  --config $Config `
  --output-dir $Data

python .\scripts\phase1_doctor.py `
  --config $Config `
  --dataset-dir $Data `
  --repo-root $Repo `
  --output (Join-Path $Output "doctor.json")

python .\scripts\run_phase1_model_study.py `
  --config $Config `
  --dataset-dir $Data `
  --output-dir $Output `
  --repo-root $Repo

python .\scripts\analyze_phase1_model_study.py `
  --config $Config `
  --output-dir $Output
```

Rerunning the matrix command resumes from hash-compatible checkpoints. Use `--overwrite` only when deliberately restarting the entire selected runs; preserve the original artifact directory first.

## Interpretation and next stage

A passing Phase 1 result justifies a Phase 2 Qwen/GQA retrofit with matched trainable capacity, natural-language iterative tasks, ordinary-capability guardrails, latency/memory measurements, and the existing raw/QKV/predictor-corrector/full decomposition. It still should not touch the independently sealed Phase 3 evaluation until the migration implementation and pilot configuration are frozen.

An inconclusive result calls for power or optimization diagnostics without changing the primary endpoint after seeing test outcomes. A contradiction should stop the Qwen retrofit and first determine whether the finite-difference coordinate is actively harmful, redundant with the carrier, or simply not learnable through the present injection path.
