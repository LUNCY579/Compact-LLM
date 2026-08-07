# Phase 9–11 RAG Reasoning Integration — v1.7.3

## Status

Implemented and repository-integrated. This document defines the v1.7.3 data and workflow contract for evidence-specialized student training, memory reconstruction, architecture reduction, and production RAG.

## Central design decision

The `rag_reasoning_v2` corpus is no longer an external optional packet. It is the canonical evidence-reasoning curriculum shared by Phases 9, 9B, 10, and 11:

```text
rag_reasoning_v2
    ↓
Phase 9: evidence use / abstention / injection / procedure behavior
    ↓
Phase 9B: source-state applicability reconstruction
    ↓
Phase 10: dev-selected structural compression and recovery
    ↓
Phase 11: calibrated production RAG and sealed runtime benchmark
```

The original uploaded corpus is preserved conceptually, but v1.7.3 produces a repository-native normalized copy under:

```text
data/phase_9_student_distillation/rag_reasoning_v2/
```

## Integrated corpus inventory

| Asset | Count |
|---|---:|
| Knowledge documents | 464 |
| Trusted procedures | 8 |
| Reviewed canonical cases | 530 |
| Train | 392 |
| Dev | 67 |
| Sealed test | 71 |
| Train DPO preference pairs | 1,176 |
| Dev DPO preference pairs | 201 |
| Evaluator calibration examples | 1,836 |
| Evaluator heldout cases | 284 |

The authored topic-family split is preserved. Phase selection is forbidden from using the sealed test split.

## Dataset normalization and repair

### Procedure ontology

`mode` and `decision` now describe different concepts:

```text
mode = procedure
```

means a trusted procedure is needed, while:

```text
decision = answer | abstain | social
```

controls whether the final response should answer. Legacy `decision=procedure` rows are normalized to `decision=answer` and remain backwards-compatible in the reader.

### Two resolvable procedure disagreements

Two rows originally labeled as conflicts were inconsistent with their own trusted procedures and questions:

- semantic versions `2.9.0` vs `2.10.0` with the question “Which version is newer?”;
- ISO dates `2032-04-09` vs `2032-11-02` with the question “Which build date is later?”.

The normalized corpus changes these to supported answers:

```text
Version 2.10.0.
2032-11-02.
```

True procedure conflicts remain abstention cases.

### Conflict handling

A genuine equal-authority conflict may be handled safely with the canonical Phase 9 abstention:

```text
I don't know based on the available information.
```

Explicitly saying that the sources conflict is recorded separately as a diagnostic, but it is not required for the response to be functionally safe.

### Trust and freshness precedence

Structured conflict detection now ignores sources marked:

```text
trust_level = superseded
trust_level = untrusted
```

before determining whether still-applicable sources conflict. Grounded prompt headers expose relevant trust and update metadata. This lets the model distinguish:

```text
old superseded fact + current replacement → use current replacement
```

from:

```text
two current equal-authority incompatible facts → abstain/conflict
```

### Reconstruction metadata

Where applicable, the normalized corpus adds or preserves:

- `current_state`;
- `source_state`;
- `validity_conditions`;
- `memory_type`;
- trust level;
- timestamps/version metadata;
- source IDs and claim metadata.

These fields are inputs for Phase 9B applicability reconstruction; they do not turn deterministic conversion into automatic ground truth. Human review remains mandatory before reconstruction training.

## Phase 9 integration

### Capability ladder

The official grounded-recovery runner defaults to the new corpus:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grounded_recovery.ps1 `
  -Mode Full `
  -TeacherModel <teacher-or-strong-base-model> `
  -UseLora
```

For a deliberately unapproved diagnostic teacher/base model, add:

```powershell
-AllowUnreleasedTeacher
```

The runner validates the corpus before training.

### Sealed-test discipline

Phase 9 now uses:

```text
train → SFT/KD optimization
dev   → teacher/warm preflight, SFT selection, Real-RAG diagnosis, KD selection
test  → one final frozen-candidate evaluation
```

The final SFT/KD choice is frozen on dev before test is touched.

### Foundation rehearsal without leakage

Foundation examples may still provide conversational diversity, but exact normalized questions present in dev or test are excluded before mixing. The merge manifest records holdout hashes and excluded IDs.

### Expanded behavior metrics

Phase 9 reports real numerator/denominator evidence for:

- supported-answer correctness;
- evidence faithfulness;
- missing-evidence abstention;
- unsupported confident answering;
- safe conflict handling;
- explicit conflict acknowledgment (diagnostic);
- procedure correctness on answerable procedure rows;
- social usability;
- qualified/ambiguous evidence calibration;
- stale/current precedence;
- prompt-injection task success;
- prompt/decode integrity.

These functional dimensions participate in the proportional opportunity calculation where applicable.

### Optional preference refinement

The corpus's preference pairs are not silently mixed into SFT. After a proven SFT candidate, they can be used explicitly:

```powershell
.\scripts\phase_9_student_distillation\run_evidence_preference_refinement.ps1 `
  -Model <selected-sft-base-model> `
  -SftAdapter <selected-sft-adapter> `
  -TrainPreferences .\data\phase_9_student_distillation\rag_reasoning_v2\preferences\train.jsonl
```

This is an optional refinement/ablation, not a prerequisite for Phase 9 success.

## Phase 9B integration

Phase 9B now defaults to:

```text
SourcePrompts = rag_reasoning_v2/prompts.all.jsonl
MemoryFile    = rag_reasoning_v2/knowledge.jsonl
```

The deterministic reconstruction proposal follows:

```text
1. reject untrusted / injection memories
2. invalidate superseded memories for current scope
3. check explicit validity conditions
4. detect conflicts among remaining applicable memories
5. propose ACCEPT / ADAPT / REJECT / CONFLICT / INSUFFICIENT
```

This fixes the previous stale/current failure mode where a superseded old value and its current replacement could be mislabeled as a conflict.

The authored split is preserved when building the reviewed reconstruction dataset.

Prepare the review packet:

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Prepare `
  -TeacherModel <teacher-model>
```

Training remains blocked until reviewed reconstruction targets are supplied.

## Phase 10 integration

Phase 10 defaults to the v1.7.3 Phase 9 dataset and retrieval artifacts.

The selection contract is now:

```text
dev:
  reference screening
  raw structural screening
  recovery selection
  recovered candidate comparison
  challenger comparison
  final candidate freeze

sealed test:
  reference evaluation
  one frozen final candidate evaluation
```

`select_recovery_candidates.py --purpose final` freezes the final candidate on dev. If no dev candidate is eligible, the best research-only candidate can still be frozen for a sealed failure measurement, but test results may not trigger candidate swapping.

## Phase 11 integration

Phase 11 defaults to the shared knowledge/procedure corpus and v1.7.3 dev/test artifacts.

Answerability calibration now treats normalized supported semantics correctly, including answerable procedure rows. It is calibrated on dev only.

The production benchmark uses functional correctness rather than a generic quality-score cutoff and separately reports:

- supported correctness;
- correct abstention;
- safe conflict handling;
- procedure correctness;
- evidence calibration;
- stale/current precedence;
- injection task success;
- social usability;
- denominator counts.

The final deployment benchmark uses the sealed test split.

## Preparation and validation commands

The integrated copy can be regenerated deterministically from a compatible source corpus:

```powershell
python .\scripts\phase_9_student_distillation\prepare_rag_reasoning_corpus.py `
  --source-root <source-corpus-directory> `
  --output-root .\data\phase_9_student_distillation\rag_reasoning_v2
```

Validate before any expensive run:

```powershell
python .\scripts\phase_9_student_distillation\validate_rag_reasoning_corpus.py `
  --corpus-root .\data\phase_9_student_distillation\rag_reasoning_v2
```

Expected v1.7.3 validation counts are 530 total reviewed rows with 392/67/71 train/dev/test.

## Safety and research interpretation

This release validates code, data contracts, split discipline, and deterministic pipeline behavior. It does not claim that a particular student, reconstruction adapter, structural candidate, or production runtime has passed its human quality gate until the corresponding GPU experiment and human review are completed.
