# v1.7.5 — Large Corrective RAG + Memory-Reconstruction Curricula

## Purpose

v1.7.5 responds to a concrete Phase 9 failure mode: students can improve on broad quality while remaining nearly unable to answer supported factual or procedural rows. The release therefore increases **behavioral supervision density** in exactly those weak capabilities, while preserving abstention, conflict, injection, social, and reconstruction controls so the student is not trained into blind answering.

The release does not assume that data scarcity is the only possible cause. If the tiny overfit / Oracle-evidence stages still fail, the workflow continues to treat training labels, completion masking, checkpoint reload, prompt parity, and model capacity as possible causes.

## Phase 9 RAG Reasoning Corpus v3 Large

Location:

`data/phase_9_student_distillation/rag_reasoning_v3_large`

### Canonical corpus

- 5,900 canonical examples
- 6,000 factual knowledge documents
- 200 trusted procedures
- 600 topic families
- 4,410 train / 752 dev / 738 sealed test
- 4,200 answer rows / 1,600 abstention rows / 100 social rows
- 4,200 grounded factual / 1,600 procedure / 100 social-roleplay rows

### Behavior coverage

| Behavior | Canonical rows |
|---|---:|
| Direct supported extraction | 1,200 |
| Supported paraphrase | 300 |
| Multi-document supported | 300 |
| Supported with distractor | 300 |
| Stale/current precedence | 300 |
| Prompt-injection resistance | 300 |
| Qualified/tentative answer | 300 |
| Ambiguous evidence | 300 |
| Missing answer | 300 |
| Equal-authority conflict | 300 |
| Partial support | 300 |
| Procedure supported | 1,000 |
| Procedure with distractor | 200 |
| Procedure missing input | 200 |
| Procedure conflict | 200 |
| Social rehearsal | 100 |

### Corrective views

`curriculum_views/support_procedure_recovery.train.reviewed.jsonl` contains 3,543 answer-heavy corrective rows.

`curriculum_views/procedure_bootstrap.train.reviewed.jsonl` contains 900 answerable procedure examples.

The normal grounded-recovery runner uses the new `support_procedure_recovery` balance by default:

- supported: 55%
- procedure: 25%
- unsupported: 8%
- conflict: 4%
- injection: 5%
- social: 3%

This means **80% of the corrective training mixture directly exercises supported evidence use or procedure application**, while retaining enough negative examples to protect abstention and security behavior.

### Retrieval integrity

The v1.7.5 corpus is designed so retrieval itself does not make the supported benchmark impossible. A new technical audit runs before model evaluation and requires every supported DEV row's authored support document to occur in the FTS5 top-3 selection.

Final release audit:

- supported DEV cases: 530
- authored support retrieved in top 3: 530
- top-3 support recall: 100%

This is a **dataset/index integrity check**, not a model-quality promotion threshold. The answerability score gate can still be imperfect even when the correct document is retrievable.

### Gold provenance

The large generated corpus uses `review_status: auto_verified`. This is intentional. The examples are deterministic fictional gold generated from explicit structured templates, then validated by repository invariants. They are **not falsely represented as human-reviewed teacher generations**.

## Phase 9B Memory Reconstruction Corpus v2 Large

Location:

`data/phase_9b_memory_reconstruction/reconstruction_corpus_v2_large`

### Canonical corpus

- 1,500 source-state memories
- 3,600 canonical reconstruction examples
- 300 topic families
- 2,700 train / 456 dev / 444 sealed test
- 600 counterfactual pairs
- 10,800 curated preference pairs
- 14,400 evaluator candidates

Decision coverage:

- ACCEPT: 900
- ADAPT: 1,200
- REJECT: 600
- CONFLICT: 300
- INSUFFICIENT: 600

Every canonical reconstruction row has a current state. Every memory retains source state and validity conditions.

### Scenario coverage

The corpus contains 300 rows in each of twelve scenario families:

1. exact state match / ACCEPT
2. paraphrased state match / ACCEPT
3. multi-memory applicability
4. micro-adaptation
5. counterfactual adaptation
6. source-state shift / ADAPT
7. retrieval prompt injection
8. premise invalidation
9. goal mismatch
10. equal-authority conflict
11. insufficient current state
12. semantically relevant but currently inapplicable memory

### Reconstruction training views

Each canonical row can yield three supervised views:

1. current state + memories -> structured reconstruction
2. reconstructed guidance -> concise final answer
3. current state + raw memories -> direct answer for later one-pass distillation

For the authored split this materializes to:

- train: 2,700 canonical / 8,100 combined SFT rows / 8,100 curated preference rows
- dev: 456 canonical / 1,368 combined rows / 1,368 preference rows
- test: 444 canonical / 1,332 combined rows / 1,332 preference rows

Known-wrong preference/evaluator alternatives remain isolated from ordinary SFT.

### Evaluator sanity check

On the DEV evaluator packet, the approved reconstruction receives mean score 1.0. The deliberately wrong variants are materially lower, and there are zero ordering violations:

- approved: 1.000
- raw replay: ~0.297
- over-rejection: 0.325
- wrong source: ~0.293

This is a structural calibration sanity test, not a claim that the heuristic evaluator replaces model or human evaluation.

## Workflow integration

### Phase 9

1. Validate RAG corpus and authored split.
2. Build FTS5 index.
3. Build train/dev/test without resplitting topic families.
4. Audit top-3 support retrieval on DEV.
5. Build/merge curriculum.
6. Apply `support_procedure_recovery` behavior profile.
7. Run teacher Oracle preflight and warm initialization baseline on DEV.
8. Run tiny deliberate overfit wiring test.
9. Run warm-SFT sweep and select on DEV Oracle behavior.
10. Evaluate selected SFT on DEV Real/adversarial RAG.
11. Optionally run progressive KD / evidence-preference refinement.
12. Freeze exactly one candidate, then touch sealed test.

### Phase 9B

1. Require accepted Phase 9 student.
2. Validate v2-large reconstruction corpus and evaluator calibration.
3. Materialize three reconstruction training views.
4. Train reconstruction SFT.
5. Optionally run progressive KD.
6. Optionally run curated preference alignment.
7. Compare reconstruction strategies on DEV.
8. Freeze exactly one strategy.
9. Run sealed test and human gate.

### Phase 10

Phase 10 defaults now point at the v1.7.5 RAG and reconstruction corpora and consumes the human-approved Phase 9B student when available. Structural candidate/recovery selection remains DEV-only; sealed test remains after the final architecture is frozen.

### Phase 11

Phase 11 builds its production snapshot from the 6,000 knowledge documents and 200 trusted procedures. It retains `off`, `prompted`, and `trained` memory-reconstruction modes, calibrates on DEV, and measures the frozen runtime on sealed test.

## Recommended first run

For a clean corrective Phase 9 experiment:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Preflight `
  -TeacherModel <YOUR_STRONG_TEACHER_OR_BASE_MODEL> `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_5 `
  -UseLora `
  -AllowUnreleasedTeacher
```

If Preflight passes:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Overfit `
  -TeacherModel <YOUR_STRONG_TEACHER_OR_BASE_MODEL> `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_5 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

Then:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Sweep `
  -TeacherModel <YOUR_STRONG_TEACHER_OR_BASE_MODEL> `
  -OutputRoot .\artifacts\phase_9_grounded_recovery_v1_7_5 `
  -UseLora `
  -AllowUnreleasedTeacher `
  -ReuseExisting
```

Finally run `-Mode Full` only after the earlier ladder stages succeed.

For Phase 9B, after Phase 9 receives human approval:

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Train `
  -UseLora `
  -RunPreferenceAlignment `
  -OutputRoot .\artifacts\phase_9b_memory_reconstruction_v1_7_5
```

Then run the same wrapper with `-Mode Evaluate -ReuseExisting`.

## Interpretation

v1.7.5 deliberately increases **useful repetitions of the underlying skill**, not repetitions of the same factual trivia. Product names, values, evidence order, distractors, source status, procedure type, counterfactual state, and decision class vary while the core behavior remains stable.

If supported/procedure Oracle performance remains near zero even after the overfit probe and answer-heavy SFT curriculum, the correct conclusion is not automatically “generate still more data.” At that point the likely investigation shifts to training wiring, label masking, checkpoint loading, prompt mismatch, or student capacity.
