# Release verification — Phase 9 student distillation v1.5.0

## Source-level verification performed

- Python compilation: PASS for `scripts/` and `tests/`.
- Regression suite: PASS, 144 tests.
- Evaluator v4.5 comprehensive calibration: PASS, 639/639 cases.
- Relative-opportunity policy audit: PASS; no private absolute promotion floors detected.
- Phase 9 dependency-light tests cover routing, injection filtering, calibrated gating, conflict detection, numeric grounding, SQLite FTS retrieval, group-disjoint splitting, threshold selection, mandatory teacher review, configuration defaults, and runner references.

## Implementation safeguards

- Phase 9 requires an accepted Phase 8 teacher unless the operator explicitly selects a research-only override.
- Randomly initialized narrow-Qwen students require explicit acknowledgement and remain marked controls.
- Direct logit KL verifies identical tokenizer fingerprints and rejects cross-tokenizer use.
- Teacher generations require human acceptance before entering the training splits.
- Pilot splitting is by topic/document group and checks support-document leakage.
- Evidence thresholds are calibrated on development data and are unset in the default configuration.
- The first retrieval implementation uses SQLite FTS5 and reports total process memory, not only generator weight size.
- Completion-only KD uses selected vocabulary-logit positions where supported to avoid materializing prompt-position logits.
- Student-generated on-policy rows carry zero hard-label weight, preventing cross-entropy from teaching the student's own draft as truth.
- Phase 9 writes a pending human gate and cannot self-approve a deployment student.

## Execution not performed in this environment

Full model loading, CUDA training, QLoRA, online teacher/student KD, and runtime VRAM benchmarking were not executed here because this validation environment does not contain the repository's complete Transformers/PEFT/TRL CUDA stack or the user's teacher checkpoint. Those operations must be run on the target Windows/CUDA machine. The source package therefore proves code structure, regression behavior, and orchestration references—not model quality or GPU memory outcomes.
## Phase 9 response-contract regression correction

A target-machine Bootstrap run exposed that the original Phase 9 sample records stored `response_contract.requirements` as strings while Evaluator v4.5 expects explicit objects. Phase 9 now performs compatibility normalization at its ingestion boundary, the sample data uses the canonical object schema, and the full teacher-data quality-evaluation call is covered by regression tests. The shared evaluator remains unchanged and strict.

