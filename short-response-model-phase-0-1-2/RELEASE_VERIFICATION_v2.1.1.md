# Release Verification — v2.1.1 Preservation-First Procedural Recovery

## Scope

This release is a corrective follow-up to the observed v2.1 recovery run. It does **not** claim a new GPU-trained student inside this packaging environment. It changes the recovery curriculum, schedule, candidate-selection logic, documentation, and test suite so the user's next GPU run can preserve the already-strong query-aware path instead of overwriting it.

## Observed v2.1 evidence incorporated

The preserved v2.1 comparison reports:

- legacy + old weights procedure correctness: **0.0000**;
- query-aware + same weights: **0.9533**;
- query-aware + recovered weights: **0.2267**;
- challenge procedure correctness: **0.00154 → 0.15432**;
- supported correctness: **0.63396 → 0.48868**;
- evidence faithfulness: **0.97938 → 0.83929**;
- correct abstention: **0.81295 → 0.80576**;
- conflict-safe and social usability remained at **1.0**;
- reported v2.1 training opportunity margin was approximately **+0.00018**.

The collision audit also preserves the earlier finding that all 150 historical answerable procedure cases had authored support and an equivalent trusted procedure, all 150 legacy prompts had sibling collisions, and query-aware selection removed those collisions.

v2.1.1 interprets those results as: **retrieval is the main historical fix; training is a residual intervention that must be preservation-positive.**

## New data verification

Command:

```text
python scripts/phase_9_student_distillation/validate_rag_reasoning_corpus.py --corpus-root data/phase_9_student_distillation/procedural_recovery_v211
```

Result:

```text
PASS
canonical rows: 6560
train/dev/test: 4976 / 792 / 792
knowledge documents: 5040
procedure records: 2320
problems: []
```

The corpus contains **24** deterministic procedure families and mixed evidence difficulty:

```text
clean binding:          1440
moderate distractor:    1440
hard sibling collision: 2880
boundary:                800
```

The generated training views are:

```text
Stage A short hard-binding view:      768
Stage B preservation-first mix:      4820
  original preservation replay:      3120 (64.7%)
  new v2.1.1 procedure rows:          1700
DEV preservation probe:                512
```

Stage B preservation includes factual answers, factual boundaries, social behavior, **verified original procedure answers**, and original procedure boundary cases. The **18** known historical inconsistent latency-tie rows are excluded from replay but remain untouched in the historical corpus.

## Procedural-gold audit

Command:

```text
python scripts/phase_9_student_distillation/audit_procedural_gold_consistency_v21.py --input data/phase_9_student_distillation/procedural_recovery_v211/reviewed_splits/train.reviewed.jsonl --require-clean
```

Result:

```text
checked latency answer rows: 186
violations: 0
```

## Selection-policy regression check

The new selector was fed the observed v2.1 before/after metrics as a synthetic policy check. Under the official `preservation_guarded` policy, the recovered v2.1-style candidate receives:

```text
positive challenge gain:       +0.15278
weighted preservation cost:     0.31710
opportunity margin:            -0.16432
selected path:                  query_aware_only
```

That is intentional. The v2.1.1 selector would reject the observed destructive recovery despite its challenge improvement.

## Conservative schedule

Defaults:

```text
Stage A:  60 steps @ 1e-5
Stage B1: 60 steps @ 5e-6
Stage B2: 60 steps @ 3e-6
LoRA rank: 16
```

The PowerShell runner refuses, by default, any recovery schedule with:

- LR above `3e-5`, or
- more than `400` total recovery steps.

An explicit `-AllowAggressiveTraining` switch is required to bypass the preservation guard for a deliberate ablation.

## Code verification

Commands:

```text
python -m compileall -q scripts tests
pytest -q
```

Result:

```text
255 passed
```

The new focused test file contributes six tests covering:

- corpus validity/counts;
- 24-family real-world diversity;
- 768-row hard-only Stage A;
- 64.7% preservation-first Stage B including original procedure replay;
- conservative checkpoint-ladder runner/config contracts;
- selector rejection of v2.1-style catastrophic forgetting;
- selector acceptance of a small residual gain when preservation cost is lower.

PowerShell is not installed in this packaging environment, so the `.ps1` workflow is covered by static regression contracts rather than executed here.

## Sealed TEST policy

The first v2.1.1 run should be DEV-only. The workflow evaluates candidate zero, Stage A, B1, and B2 on both original Phase 9 DEV and the new v2.1.1 challenge DEV, then writes:

```text
PROCEDURAL_RECOVERY_V211_SELECTION.md
procedural_recovery_v211_selection.json
```

Only after one DEV path is frozen should `-ReuseExisting -RunSealedTest` be used. The runner evaluates exactly the selected path, even when that path is the **unchanged original adapter**. TEST never retrains or recalibrates the evidence gate.

## Package limitation

No new model accuracy is pre-claimed. The release is an executable research redesign whose central hypothesis is measurable: a smaller residual curriculum plus preservation-dominated consolidation and DEV checkpoint selection should retain the strong query-aware original path better than the v2.1 recovery run. If it does not, the correct result is `query_aware_only` rather than forcing a new adapter.
