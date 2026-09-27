# Exact-host handoff — CSS-001

## Current boundary

The shipped plan has native/training authorization false. Local mechanics use tiny random
matrices and a tiny CPU model. No real Rineya checkpoint, native forward, optimizer step, Windows
memory qualification, successful inherited endpoint preflight or GPU timing is claimed.
The public Python training/router APIs are usable research components, not a grant bypass.

Use the new branch only. Keep the parent branch and existing experiment records unchanged.
No downloads, automatic installation, model discovery, checkpoint cleanup, remote training,
job termination, batch sweep, release or reuse of another experiment's grant is implemented.

## Preparation and operator review

1. Run the CPU tests and demo from the repository root. Copy `plan.example.json` to a new local
   plan. Freeze donors, coefficients, matrix scope, ranks, private gain, arms and decision margins.
2. Bind the inherited `experiments/titan_model_soup_mini_v1/manifest.example.json` locally using
   its exact published full-weight and sidecar hashes. Do not invent paths, treat partial trained
   tensors as deltas, add a child's inherited delta twice, average raw LoRA A/B factors, or relax pins.
3. Review all donor ancestry and supplied TRAIN/CAL/CHECK ledgers. A fresh independent holdout,
   prompt-only collection suites and separately vetted scorer are required. Historical exposed,
   spent or quarantined pools cannot be relabeled as confirmation data.
4. Provision the host-wide shared lease and a persistent attempt directory. Issue a distinct
   CSS-001 single-action/single-arm grant, bound to `digest(plan)`, `source_digest()`, the manifest,
   expiry and output parent. The example is deliberately invalid. Native collection additionally
   requires reviewed exact-runtime and ancestry receipts, all model-file hashes and the frozen
   suite/artifact record. Changing a plan or source invalidates prior bindings.

Plan/code digests are **canonical-object/source-map** hashes, not the raw JSON file hash. Model,
manifest, suite and receipt file hashes are raw-file SHA-256. Use the library functions, not a
mixture of the two conventions. `validate` prints the plan/code identities.

Native permissions must be registered **before** building/freezing artifacts intended for native
collection; editing the plan to enable native later changes its hash and requires a fresh binding
and build. A true plan flag alone never authorizes execution: a reviewed per-action grant is also
required. Training needs a separate grant and exact-host backend qualification.

## Preflight and one artifact

```bash
python -m experiments.rineya_composable_subspace_stack_v1 preflight --plan css-plan.local.json --grant css-preflight.local.json --manifest soup-manifest.local.json
python -m experiments.rineya_composable_subspace_stack_v1 build --plan css-plan.local.json --grant css-build.local.json --manifest soup-manifest.local.json --arm P1S1C1 --output /absolute/new/P1S1C1
```

Each command consumes its own grant. Failed attempts stay spent; no automatic retry.
Admission requires two consecutive >=7.4 GiB free host-RAM checks, then a >=6.0 GiB worker
floor. Build requires output bytes plus a 1 GiB disk margin. Native CUDA collection requires
>=7 GiB free GPU memory; these conservative floors are not measured peak-memory guarantees.
Dense float64 SVD workspace is bounded by selected tensor size but LAPACK/OS allocations are
not an exact RAM cap. Start with the one explicitly registered matrix, not all 196 matrices.

The output is an ordinary FP32 safetensors checkpoint with immutable-source provenance and
base tokenizer/config sidecars. `INCOMPLETE.json` or `FAILED.json` makes it ineligible. Unselected
weights are bit-exact FP32 copies of the base. Full endpoint controls copy the entire endpoint.
The bridge checks inherited `soup.py` git-blob identity before importing its auditor.

Build only predeclared arms, one at a time. Never delete a checkpoint automatically to make room.
A small first calibration set can include `base`, `P0S0C0`, `P1S1C1` and matched component-dose
controls; the complete eight-cell design is required later for factorial attribution. A frozen
subset is supported, so all 19 configurations need not reside on disk simultaneously.

## Freeze, collect, score and confirm

`freeze` reads a direct artifact-map JSON object, for example
`{"base":"<sha>","P1S1C1":"<sha>"}`; do not wrap it in an `artifacts` field.
The freeze also binds the raw CAL/CHECK suite hashes and reviewed ancestry-audit file hash.

```bash
python -m experiments.rineya_composable_subspace_stack_v1 freeze --plan css-plan.local.json --artifacts artifact-hashes.local.json --cal-sha256 <CAL_SHA256> --check-sha256 <CHECK_SHA256> --audit-sha256 <AUDIT_SHA256> --output freeze.local.json
python -m experiments.rineya_composable_subspace_stack_v1 collect --plan css-plan.local.json --grant css-cal.local.json --model /absolute/P1S1C1 --suite CAL.local.json --freeze freeze.local.json --output P1S1C1.raw.local.json
python -m experiments.rineya_composable_subspace_stack_v1 select --plan css-plan.local.json --reference base.scored.local.json --candidates P1S1C1.scored.local.json --freeze freeze.local.json --output selection.local.json
python -m experiments.rineya_composable_subspace_stack_v1 confirm --run selected.check.scored.local.json --freeze freeze.local.json --selection selection.local.json --output confirmation.local.json
```

Scored runs are produced by the reviewed external scorer, not by `collect`. Preserve all common
metadata from the bound runtime/decoder/tokenizer/scorer and use actual per-row losses if testing
loss interactions. `demo.py` and the evaluation tests give executable scored-run examples.
CHECK collection uses a separate grant with `split: CHECK`, the selected artifact and
`selection_sha256`; pass `--selection selection.local.json`. A rejected/empty CAL selection does
not trigger a CHECK run. Collection never silently switches to a cheaper dtype/runtime.

## Stage 2 and routing APIs

`training.FunctionalSkill` accepts a supplied model returning `(prediction, hidden)` and a bounded
list of trainable parameter names. First run `FunctionalSkill.qualify` on the exact architecture. Then use
`ordinary_skill`, `distill_skill` or `train_composition_study` with TRAIN-only batches, registered
step limits and a common merger. Review the code/test examples; no native loader or training
launch is hidden inside these APIs. Native Qwen3 hidden-state wrapping remains an exact-host
integration/qualification task. Do not call a tiny fixture's JVP evidence native qualification.

`routing.fit_router` consumes CAL `PromptFeature` records and task labels. Deploy `TaskRouter.route`
or `route_and_predict` using independently generated prompt-only features. `LinearSkillBank`
provides reversible per-Linear additive hooks for qualified effective deltas. Adapter banks use
more memory than a static merge; measure latency, feature cost and memory separately. Oracle
results never choose deployed routes. Tokenizer-derived feature generation is host supplied.
