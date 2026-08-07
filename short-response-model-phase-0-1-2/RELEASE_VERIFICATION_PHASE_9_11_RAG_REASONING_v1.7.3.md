# Release Verification — Phase 9–11 RAG Reasoning Integration v1.7.3

## Scope

This verification covers repository/data integration, deterministic corpus contracts, cross-phase workflow behavior, Python compilation, and automated tests. It does not claim GPU-trained model quality or a human Phase 9/9B/10/11 PASS.

## Canonical corpus validation

Command:

```text
python scripts/phase_9_student_distillation/validate_rag_reasoning_corpus.py \
  --corpus-root data/phase_9_student_distillation/rag_reasoning_v2
```

Result:

```text
PASS
reviewed rows: 530
train: 392
dev: 67
test: 71
knowledge documents: 464
problems: []
```

The authored topic-family split is preserved and leakage-validated.

## Phase 9 dataset build smoke

The complete reviewed corpus was passed through `build_pilot_dataset.py --split-policy intended`.

Result:

```text
train: 392
dev: 67
test: 71
research_ready: true
preserved_intended_splits: true
```

Decision-class counts are present for supported, unsupported, conflict, and social behavior. The two repaired procedure disagreements are supported answers; true conflicts remain abstention cases.

## Phase 9B reconstruction candidate smoke

The full shared prompt/memory corpus was processed by `build_reconstruction_candidates.py`.

Result:

```text
candidate rows: 512
accept: 306
adapt: 96
conflict: 54
insufficient: 56
```

The remaining 18 memory-free social rows are intentionally not reconstruction candidates. Regression coverage verifies that superseded/current evidence produces applicability adaptation rather than a false equal-authority conflict.

## Phase 11 snapshot and answerability calibration smoke

The integrated 464 knowledge documents plus 8 procedures were indexed into an immutable SQLite/FTS5 snapshot. The v1.7.3 dev split was then used for answerability calibration.

Selected calibration packet:

```text
threshold: 0.6142857142857142
true positive: 41
false positive: 2
true negative: 12
false negative: 9
answer precision: 0.9534883720930233
supported recall: 0.82
unsupported rejection: 0.8571428571428571
false answer rate: 0.14285714285714285
unnecessary abstention rate: 0.18
```

These numbers verify calibration behavior only; they are not student-generation quality scores.

## Automated test suite

Command:

```text
python -m pytest -q
```

Result:

```text
207 passed in 21.30s
```

Coverage includes prior evaluator/phase regressions plus v1.7.3 tests for:

- corpus manifest/hash integrity;
- procedure ontology repairs;
- all 530 reviewed cases building successfully;
- exact authored split counts;
- canonical safe conflict abstention;
- trust-aware structured conflict detection;
- stale-memory ADAPT reconstruction;
- preference/evaluator asset counts;
- both Phase 9 runners using dev-first selection and sealed final-test ordering;
- Phase 10 dev-freeze/final-test ordering;
- Phase 11 supported/procedure decision semantics and specialized metrics.

## Python compilation

Command:

```text
python -m compileall -q scripts tests
```

Result:

```text
PASS
```

## PowerShell wrappers

A PowerShell executable is not available in the packaging environment. The wrappers were therefore not launched here. Their paths, arguments, ordering, and delimiter/reference contracts are covered by repository regression/static tests, including the full 207-test pass above.

## Packaging policy

The final archive excludes:

- `.pytest_cache`;
- `__pycache__`;
- `.pyc` files;
- temporary smoke artifacts outside the repository.

A post-package CRC test and SHA-256 are computed on the final ZIP before delivery.

## Release interpretation

v1.7.3 is fully integrated under the repository's automated checks. It does **not** assert that any actual student, KD adapter, reconstruction adapter, structurally reduced checkpoint, or production deployment has passed its quality/human stop gate. Those require the corresponding real training/evaluation runs.
