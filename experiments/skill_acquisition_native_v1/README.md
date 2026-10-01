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
| `protocol.py` | Validate authorization, S0 ancestry, source/runtime bindings, qualification, and measured budgets. |
| `update.py` | Verify sampled-policy identity and raw log probabilities; apply the proposed MaxRL loss; extract a donor relative to the actual warm anchor. |
| `checkpoint.py` | Stream parameters, Adam state, module modes, and process RNG state into two verified rolling checkpoints. |
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
