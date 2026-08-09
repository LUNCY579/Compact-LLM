# Release Verification — v2.1 Procedural Transfer Recovery

## Scope

This package is a focused update of the supplied v2.0 repository. It does **not** claim that a new student checkpoint was trained inside this packaging environment. Instead, it preserves the observed Phase 9 evidence, performs model-free forensic analysis, adds the procedural-recovery data and code path, and provides a DEV-first runner that can measure the repair on the user's actual selected student.

## Phase 9 evidence inspected

The packaged Phase 9 assessment/stop-gate identifies `sft` as the strongest compressed candidate but does not approve any production candidate because of the Real-RAG functional bottleneck.

Selected `sft` evidence:

- deployed-weight reduction vs teacher: **0.520189**;
- generic relative quality loss: **0.006214**;
- Oracle supported correctness: **1.000000**;
- Real-RAG supported correctness: **0.359615 (187/520)**;
- Real-RAG evidence faithfulness: **0.960905**;
- Real-RAG correct abstention: **0.764706 (104/136)**;
- Real-RAG unsupported confident answer rate: **0.235294**;
- Real-RAG procedure correctness: **0.000000 (0/150)**;
- Real-RAG social usability: **1.000000 (20/20)**;
- prompt/decode integrity failures: **0 / 0**.

The package does not expose a separate Oracle-procedure scalar in `PHASE_9_ASSESSMENT.json`. The user's reported Oracle procedural score of 1.0 is therefore treated as supplied experimental context; the package independently confirms Oracle supported correctness of 1.0.

## Model-free forensic findings

`reports/PHASE_9_PROCEDURAL_RETRIEVAL_COLLISION_AUDIT_v2.1.json` audits the already-exposed 150 answerable historical procedural TEST cases:

- support recall in legacy top-3: **150/150 = 1.0**;
- support rank 1: **143/150**;
- support rank 2: **7/150**;
- equivalent trusted-procedure retrieval recall: **1.0**;
- legacy same-topic sibling-record collisions: **150/150 cases**;
- legacy sibling documents in those prompts: **300**;
- query-aware v2.1 static selection sibling collisions: **0/150**;
- mean selected knowledge passages after query-aware locator filtering: **1**.

This is evidence that the historical failure cannot be explained as simple support absence. It motivates a same-weight retrieval-policy A/B before attributing all 0/150 failures to missing learned procedural skill.

## Historical supervision defect preserved, not rewritten

`reports/PHASE_9_V175_TRAIN_PROCEDURAL_GOLD_AUDIT_v2.1.json` finds **18 violations among 114 checked answerable latency rows** in the historical v1.7.5 training corpus: equal target latencies were paired with a unique-label gold even though the stated rule was to choose the smaller latency.

The historical data is intentionally left unchanged for reproducibility. The v2.1 recovery generator prevents implicit ties; the v2.1 training audit checked **297 latency answer rows with 0 violations**.

## New recovery corpus verification

Command:

```text
python scripts/phase_9_student_distillation/validate_rag_reasoning_corpus.py --corpus-root data/phase_9_student_distillation/procedural_recovery_v21
```

Result:

```text
PASS
canonical/reviewed rows: 6120 / 6120
train/dev/test: 4648 / 736 / 736
knowledge documents: 2880
procedure records: 1560
problems: []
```

The recovery corpus includes 14 deterministic procedure families, hard sibling-collision examples, missing-input cases, conflict cases, and a train-only non-procedural preservation sample. Its newly generated TEST topics remain sealed until the DEV recipe is frozen.

## Code verification

Commands executed after the v2.1 changes:

```text
python -m compileall -q scripts tests
pytest -q
```

Result:

```text
249 passed
```

The v2.1 gold consistency command also passed with `--require-clean` and reported 0 violations for the new training split.

PowerShell itself is not installed in this packaging environment, so the `.ps1` runners were validated through repository regression tests/static contract checks rather than by executing PowerShell here.

## Behavioral/research decision encoded by v2.1

1. Keep the selected student as the primary recovery target rather than discarding it immediately.
2. Run the **same weights** under legacy retrieval and `query_aware_v21` retrieval first.
3. If a gap remains, apply the fixed two-stage rank-16 procedural recovery recipe rather than starting a broad LoRA sweep.
4. Decompose measured improvement into retrieval-policy gain and training-only gain.
5. Re-check supported answering, abstention, conflict handling, evidence faithfulness, social behavior, and prompt/decode integrity to detect catastrophic forgetting.
6. Only after the mechanism is demonstrated on the primary student, replicate the same fixed recipe on one or two other students. Do not tune each student independently at first.
7. Keep structural pruning/depth-width reduction/contextual sparsity downstream until the critical procedural capability is recovered and approved.

## Important limitation

The static query-aware audit demonstrates that the known sibling-collision pattern can be removed **at retrieval-selection time**. It does **not** prove that the selected model's Real-RAG procedure correctness will rise until the same-weight A/B is actually run. Likewise, the new training corpus and two-stage recipe are engineering hypotheses to be measured, not pre-claimed accuracy gains.
