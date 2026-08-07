# Phase 9 RAG Reasoning Corpus v2.1 — integrated for model v1.7.3

This is the repository-native form of the synthetic Phase 9 RAG Reasoning Corpus v2.

## Integration repairs

- Successful procedure tasks use `mode=procedure` with `decision=answer`.
- The semantic-version and ISO-date disagreement examples are answered because the trusted procedure resolves the requested comparison.
- Remaining procedure conflicts are explicit conflict scenarios and carry structured claim metadata when possible.
- Evidence memories include source-state / validity metadata for Phase 9B.
- Authored train/dev/test topic-family splits are preserved.
- Preferences and evaluator calibration data are regenerated from the repaired canonical rows.

## Phase boundaries

- **Phase 9:** train only on train; tune/calibrate on dev; test is sealed.
- **Phase 9B:** derive reconstruction candidates from train/dev, with human review before labels become training data.
- **Phase 10:** architecture screening and recovery selection use dev; only the selected candidate is evaluated on test.
- **Phase 11:** answerability calibrates on dev; release benchmark uses test.

Do not train on any test or held-out evaluator file.
