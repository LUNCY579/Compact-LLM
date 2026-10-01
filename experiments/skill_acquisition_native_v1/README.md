# Prospective native skill acquisition

This package implements the proposed COMP-RS S1 update, data admission,
checkpoint, accounting, and single-arm runner. It does not register a recipe or
authorize training. The existing S0 competence read uses a different, unchanged
checkout.

## Current status

The native S1 recipe is an **unapproved proposal**. Native qualification, measured
operation budgets, completed S0 prerequisites, and separate S1 authorization
are required before a production invocation. Pure fixtures and tiny CPU models
do not supply those requirements. This package currently has no native
qualification entry point.

The production runner rejects an unapproved packet before importing a machine
learning framework. It accepts one named independent arm. Decisions about H1,
later arms, composition, and retention remain governed by the registered study.
A completed donor file is not evidence that its skill was acquired.

## Components

| File | Responsibility |
| --- | --- |
| `data.py` | Verify TRAIN manifests, canonical and policy files, row roles, IDs, prompts, and rendered token census. |
| `plan.py` | Fix exposure order, sample seeds, and separate exposure count C from applied Adam count U. |
| `ce_plan.py` | Bind unapproved 3,000-row CE8 exposure/label plans; permit checkpoint metadata export only for explicit synthetic fixtures. |
| `protocol.py` | Validate authorization, S0 ancestry, source/runtime bindings, qualification, and measured budgets. |
| `update.py` | Verify sampled-policy identity and raw log probabilities; apply the proposed MaxRL loss; extract a donor relative to the actual warm anchor. |
| `checkpoint.py` | Stream parameters, Adam state, module modes, and process RNG state into two verified rolling checkpoints. |
| `ce_checkpoint.py` | Admit synthetic CPU CE8 storage mechanics with exact row/token/recipe bindings and C=U; deny native scope before IO. |
| `journal.py` | Preserve physical attempts and costs across rollback; bind returned update and restore proofs. |
| `runner.py` | Hold one shared GPU lease, enforce resource budgets, collect 16 trajectories per boundary, and publish a sealed result. |

The proposed atomic unit is 3,000 TRAIN prompts with B=2 and G=8. Uniform groups
contribute zero while the loss denominator remains 16. An entirely uniform
boundary advances C without an Adam step. This is the proposal's behavior, not
a claim about effective learning. MT has two balanced units.

All independent arms start from the same deployed FP32 warm anchor. Extracted
donors use that anchor as their common reference. Sequential arms and alternate
objectives require separate integration and are rejected by production
admission.

## Verification

Run the framework-free suite from the repository root:

```powershell
python -m unittest discover -s experiments/skill_acquisition_native_v1/tests -t . -v
```

Tiny actual CPU tests in `test_update.py` and `test_checkpoint.py` are skipped by
default. Enable them only with resource headroom and no competing native model
worker:

```powershell
$env:COMP_RS_RUN_NATIVE_CPU_TESTS = '1'
python -m unittest experiments.skill_acquisition_native_v1.tests.test_update experiments.skill_acquisition_native_v1.tests.test_checkpoint -v
Remove-Item Env:COMP_RS_RUN_NATIVE_CPU_TESTS
```

These tests verify actual gradients, EOS/length masks, frozen state, donor
extraction, parameters/Adam/RNG resume, and corrupt-checkpoint refusal. Passing
them does not constitute qualification on the registered CUDA model.

## Prospective supervised core

`ce_update.py` supplies the common eight-row teacher-CE boundary for the
unapproved R oracle-supervision alternatives. It uses a labeled-token mean per
row, equal row weights, one-shift target/EOS masks, global norm clipping before
Adam, and frozen-state/U proofs. Reset and failure cleanup remain observable;
an exception after an Adam attempt requires discarding or restoring live state.

Its five pure and ten actual CPU checks passed in the superseding
`evidence/CE_CORE_CPU_CHECKS_V2_20261001.json` receipt. Review found and fixed a
second-moment sign check that covered only the final chunk. The new multi-chunk
regression proves earlier corruption is rejected before forward, clipping or
Adam work, preserves parameters and optimizer state, and admits the repaired
nonnegative control. The earlier 14-check receipt is retained as historical
evidence. It always refuses native
execution before model access, including forged approval flags. It has no model
loader, experiment driver, evaluation, target policy, retry defaults or grant.
The existing B2 checkpoint/progress adapter must not be presented as a qualified
CE8 adapter. Full integration and native qualification still require the new
operator decision and a separately registered scope.

The separate `ce_checkpoint.py` adapter now verifies eight globally unique row
identities per boundary, every completed boundary updated, exact static CE
recipe and prepared-token digest, and CPU parameters/buffers with uninitialized
CUDA. The base storage adapter retains its B2 rules. The combined CE/core/legacy
checkpoint suite passed 43 checks, including 19 actual CPU cases; see the
superseding `evidence/CE8_CHECKPOINT_CPU_CHECKS_V2_20261001.json` receipt.
Review strengthened the fixture's nested restore-progress and ancestor checks;
the original 42-check receipt is preserved.

The new stochastic CE8 integration case uses fresh model, optimizer and adapter
objects in one OS process. Parameters, buffers, Adam moments/steps, Python/NumPy/
CPU RNG and module modes match the uninterrupted second update bit for bit.
The caller installs the returned fence and recomputes actual policy identity;
old CE results and reused restore proofs are rejected by the synthetic fixture.
The corruption control rejects the latest payload without mutation or fallback.
This is mechanics evidence. The in-memory test observer supplies no durable
production CE ledger, fresh-OS-process qualification, native resource census,
experiment driver, acquisition result or training grant.

`ce_plan.py` verified both existing 3,000-row target files and the bound atomic R
manifest into 375 identical eight-row exposure boundaries. Label bindings remain
distinct between canonical and permuted teachers. The independent root export
also pins the completed warm-anchor receipt; recipe, native runtime and full
trainer source remain unresolved. Nine plan controls and three permutation-math
controls passed without framework imports. See
`evidence/CE8_PLAN_PREPARATION_20261001.json` and
`evidence/CE8_PLAN_SOURCE_CHECKS_20261001.json`.

Proposal plans always deny execution and cannot export fixture checkpoint scope.
The preparation references full canonical/policy TRAIN hashes; future native
ingestion must revalidate those files. A target-plan binding and tokenizer census
do not constitute a native recipe, measured resource envelope or learning result.

## Production interface

```powershell
python -m experiments.skill_acquisition_native_v1.runner --packet APPROVED_PACKET.json
python -m experiments.skill_acquisition_native_v1.runner --packet APPROVED_PACKET.json --resume --resume-reason 'documented interruption'
```

The approved packet must bind the clean source checkout, complete warm
competence/pilot/enrollment prerequisites, native qualification, numeric cost
limits, measured IO and RAM transients, and an owned output directory. The runner
rechecks host resources after waiting for the lease and before model allocation.
It preserves interrupted attempts and refuses already sealed completion.

`RESULT.json` is complete only with its returned publication operation and the
matching `DONOR_COMPLETE` invocation end in `PHYSICAL_COSTS.jsonl`. Unsealed
results are preserved as partial evidence during a documented resume.
