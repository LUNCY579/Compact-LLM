# SYNTH-CRS-001: centralized verified supervision for Rineya

**Branch:** `research/rineya-synth-centralized-reasoning-v1`  
**Status:** design recorded; native implementation and qualification pending.  
**Target:** one pinned approximately 0.6B Rineya student, not a new from-scratch model.

## Read in this order

1. [Owner-directed preservation charter](../../RINEYA_SYNTH_CENTRALIZED_REASONING.md).
2. [Architecture and boundaries](ARCHITECTURE.md).
3. [Controlled experimental plan](EXPERIMENT_PLAN.md).
4. [Data and evidence contracts](DATA_CONTRACT.md).
5. [Current status and next handoff](STATUS.md), [manifest](manifest.json), and [sources](SOURCES.md).

The core proposal is: grounded sources -> controlled task variants -> independently
verified targets -> student adaptation -> fresh acquisition/retention/composition
measurement. A later system-assisted comparison is a separate experiment.

This supplements the inherited [CRS-001 package](../rineya_centralized_reasoning_system_v1/README.md).
It does not silently replace its procedure-memory hypothesis or import its gates
as proof of qualification. Prior recurrence, retrieval, skill-delta composition,
abstention, and MiMo work retain their identities and evidence.

## What can run now

From the repository root, Python 3.10+; standard library only:

```bash
python experiments/rineya_synth_centralized_reasoning_v1/validate_manifest.py
python -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/tests -v
```

These commands check the founding registration, preservation policy fields,
closed execution permissions, source bindings, and separation of planned arms.
They are NOT a dataset audit, runtime integration test, native-model test,
training job, or benchmark. See [local evidence](evidence/local_validation.json).
There are no automatic network calls, paid services, model downloads, or GPU jobs.

## First implementable question

At a fixed student-training budget, compare surface-varied verified atomic tasks
against condition-diverse verified atomic tasks. Start with reference resolution
(R) and exact-size target-sum selection (S); reserve RS combinations. Use the
same final-answer-only target representation in both arms. Study prose versus
structured intermediate states only in a subsequent, separately controlled test.

The manifest is a **design registration, not an execution permit or completed
preregistration**. Checkpoint bindings, data hashes, resource ceilings, numerical
qualification, outcome margins, and launch approval must be filled in a separate
future run manifest. Never convert missing bindings into permissive defaults.
