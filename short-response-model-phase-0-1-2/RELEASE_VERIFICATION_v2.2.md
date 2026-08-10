# v2.2 Release Verification

## Lineage

v2.2 was built directly on the uploaded `Output(20260810-063134).zip`, not on an older repository snapshot.

- input repository SHA-256: `3c5623ec807d9c48aa9728a8691248afca46450ae09eccab200cabbdee7661f8`
- design-source PDF SHA-256: `b92838141e10efe471a7e2f0df008591c273bb7499de0d379d1e3bae0fdfe280`
- repository copy of the design PDF has the same SHA-256.

The original source ZIP contains 744 files. After cache cleanup, all non-cache source files are retained; the only omitted source entries are 15 disposable Python/pytest cache files. Existing source files are otherwise either byte-identical or explicitly listed in `CHANGED_FILES_v2.2.md`.

Final normalized lineage comparison before packaging:

- source files: **744**;
- final repository files: **763**;
- common files: **729**;
- byte-identical common files: **720**;
- intentionally modified existing files: **9**;
- newly added v2.2 files: **34**;
- omitted source files: **15**, all disposable cache artifacts.

## Baseline verification before v2.2 changes

The unmodified normalized source repository passed:

```text
261 passed
```

## Final code verification

The final v2.2 repository was compiled and tested with:

```text
python -m compileall -q scripts tests
pytest -q
```

Expected final result for this release:

```text
271 passed
```

v2.2-specific tests cover:

- packaged corpus counts and train/DEV/TEST isolation;
- 27-domain / 15-archetype coverage;
- TRAIN-only specialist and social-rehearsal provenance;
- no visible chain-of-thought targets;
- non-overlapping central layer plan;
- missing protected metrics becoming technical invalidity;
- positive broad gain being selectable when preservation holds;
- v2.1.2 null-preservation regression backport;
- frozen-anchor and layer-scoped LoRA CLI integration;
- deterministic stratified greedy probes;
- weighted LoRA-B scaling behavior;
- presence of all four v2.2 tracks and sealed-test gate in the PowerShell runner.

## Dataset verification

Command:

```text
python scripts/phase_9_student_distillation/validate_multidomain_reasoning_v22.py \
  --corpus-root data/phase_9_student_distillation/multidomain_reasoning_v22 \
  --require-rows 40000
```

Expected result:

```text
status: PASS
canonical_rows: 40500
domains: 27
archetypes: 15
train: 32400
dev: 4050
test: 4050
problems: []
```

Additional deterministic audit:

- 40,500 / 40,500 canonical user prompts are exact-string unique;
- 3,000 social-rehearsal rows are TRAIN-only;
- Track A broad view: 15,000 rows;
- Track B anchored mix: 15,000 rows = 12,000 broad + 2,000 original Phase 9 rehearsal + 1,000 v2.1.1 procedural rehearsal;
- specialist general: 10,000;
- specialist grounding: 5,500;
- specialist procedure: 5,500;
- specialist abstention: 4,500;
- specialist social: 4,500;
- dataset file hashes are verified by the validator;
- config and dataset manifest counts are cross-checked programmatically.

## Implemented research tracks

### Track A - broad multi-domain adapter

Fresh LoRA from the immutable anchor on the 15,000-row broad view.

### Track B - preservation-mixed + KL-anchored adapter

Fresh LoRA from the same anchor using supervised preservation rehearsal plus forward `KL(anchor || student)` on supervised completion positions. `-AnchorDevice auto|cpu` is exposed for the memory/speed tradeoff.

### Track C - parallel specialists

Independent `general`, `grounding`, `procedure`, `abstention`, and `social` LoRAs, all starting from the exact same anchor. No specialist is initialized from another specialist.

### Track D - composition

- exact weighted static LoRA-delta composition;
- true DEV-probe greedy composition, where each trial mixture is materialized and evaluated before acceptance;
- layer-separated specialist training on non-overlapping central-layer groups, followed by exact composition.

## Evaluation integrity

Every full DEV candidate uses the same four axes:

1. v2.2 multi-domain reasoning;
2. v2.1.2 reasoning continuity;
3. original Phase 9 Real-RAG with `query_aware_v21`;
4. v2.1.1 procedural challenge with `query_aware_v21`.

The original and v2.1.1 retrieval calibrations are frozen on DEV and reused for every candidate. Protected original metrics must be non-null. The frozen anchor remains a first-class fallback and participates in the Pareto frontier.

Greedy search uses only deterministic stratified DEV probes. TEST is not used for weight selection, layer selection, greedy ingredient selection, or candidate switching.

A sealed run evaluates only the already-frozen DEV winner and performs no additional training or recalibration. Policy-sensitivity disagreement blocks sealed TEST.

## PowerShell verification note

This Linux verification environment does not provide `pwsh`/Windows PowerShell, so the `.ps1` runner could not be executed end-to-end here. It was statically inspected, covered by repository tests for all required track/gate paths, and tokenized with Pygments' PowerShell lexer with zero lexer error tokens. The Python programs invoked by the runner compile and their non-GPU logic is covered by tests.

## GPU-training note

No GPU fine-tuning was performed in this packaging environment. This release is a fully wired, training-ready research workflow; it does not claim experimental wins for Track A/B/C/D before the user runs them on the target student.
