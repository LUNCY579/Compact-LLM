# Release Verification — v1.7.5 Large Corrective Datasets

## Release scope

v1.7.5 extends v1.7.4 with large deterministic corrective corpora for the two behaviors currently needing the most supervision:

1. Phase 9 evidence-grounded supported answering and trusted-procedure application.
2. Phase 9B current-state memory reconstruction and negative-transfer avoidance.

The corpora are generated from fictional structured source records and are labeled `auto_verified`; this is deterministic repository-validated gold, not a claim of human review.

## Phase 9 RAG corpus verification

Command:

```text
python scripts/phase_9_student_distillation/validate_rag_reasoning_corpus.py \
  --corpus-root data/phase_9_student_distillation/rag_reasoning_v3_large
```

Result:

- status: PASS
- schema: 3.0
- canonical rows: 5,900
- train/dev/test: 4,410 / 752 / 738
- knowledge documents: 6,000
- trusted procedures: 200
- problems: 0

Research-ready split materialization also passed with the same 4,410 / 752 / 738 authored split.

### FTS5 support-retrieval integrity

The final 6,000-document + 200-procedure store indexed 6,200 records.

The DEV support audit passed:

- supported cases: 530
- support hits in selected top 3: 530
- support recall: 1.000
- misses: 0

Category recall was 1.000 for every supported category, including all 125 `procedure_supported` DEV rows and all 25 `procedure_with_distractor` DEV rows.

This is a technical corpus/index integrity assertion, not a model-quality floor.

## Phase 9B reconstruction corpus verification

Command:

```text
python scripts/phase_9b_memory_reconstruction/validate_memory_reconstruction_corpus.py \
  --corpus-root data/phase_9b_memory_reconstruction/reconstruction_corpus_v2_large
```

Result:

- status: PASS
- memory records: 1,500
- canonical rows: 3,600
- topic families: 300
- train/dev/test: 2,700 / 456 / 444
- ACCEPT / ADAPT / REJECT / CONFLICT / INSUFFICIENT: 900 / 1,200 / 600 / 300 / 600
- counterfactual pairs: 600
- curated preferences: 8,100 train / 1,368 dev / 1,332 test
- evaluator candidates: 14,400
- problems: 0

Research-ready reconstruction materialization passed:

- train: 2,700 canonical / 8,100 combined SFT / 8,100 curated preferences
- dev: 456 canonical / 1,368 combined SFT / 1,368 curated preferences
- test: 444 canonical / 1,332 combined SFT / 1,332 curated preferences

### Reconstruction evaluator sanity check

DEV packet:

- approved mean: 1.000
- raw replay mean: ~0.297
- over-rejection mean: 0.325
- wrong-source mean: ~0.293
- ordering violations: 0

## Python and regression verification

```text
python -m compileall -q scripts tests
PASS

pytest -q
219 passed in 13.85s
```

This includes all 213 inherited v1.7.4 tests plus six new v1.7.5 corrective-dataset tests.

## PowerShell note

A PowerShell executable is not installed in the packaging environment. The `.ps1` wrappers therefore were not GPU-launched here. Their new defaults, corpus paths, profile wiring, Phase 9/9B/10/11 handoffs, and retrieval-audit references are covered by regression/static tests and manual inspection.

No GPU training result, student-quality improvement, or phase PASS is fabricated by this release. v1.7.5 validates the **data and workflow**, while the actual student must still prove the behavior through Oracle, overfit, DEV, sealed-test, and human gates.
