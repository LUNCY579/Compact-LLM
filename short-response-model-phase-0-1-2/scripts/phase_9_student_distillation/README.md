# Phase 9 — Tiny student distillation and RAG specialization

Phase 9 begins only after the official Phase 0–8 teacher pipeline has produced an explicitly approved teacher checkpoint. It does not replace the teacher pipeline. It consumes the accepted teacher and builds smaller students whose main job is to understand questions, use supplied evidence, follow reusable procedures, respond naturally and briefly, and abstain when evidence is insufficient.

## Why this phase exists

Further pruning of the 0.6B teacher can reduce depth or width, but a much smaller deployment model needs a student-native architecture. Phase 9 separates factual memory from reusable behavior:

```text
Student weights                 External RAG
-----------------------------   -----------------------------
language/question understanding factual knowledge
evidence use                    project documentation
reusable procedures             character/world lore
conversational behavior         user memories
role-playing behavior           recent/changing information
concise EOS behavior            long-tail entities
abstention behavior             domain-specific procedures
```

## Official student tracks

1. **Warm-started depth-copy intermediate** — copies complete teacher layers and preserves the teacher tokenizer/width. This is the default pipeline-validation track.
2. **Aggressive same-tokenizer random control** — constructs the 80M–120M-style Qwen architecture. It is research-only until substantial training and the full gate prove language readiness.
3. **Pretrained challenger** — clones a pretrained compact model such as SmolLM2. It uses sequence-level SFT first because direct Qwen vocabulary KL is invalid when tokenizers differ.

The original teacher remains the authoritative supervisor at every compression rung. Intermediate students may assist later students but do not replace the teacher.

## Two-stage runtime

```text
User message
  -> mode router
     -> social/role-play: direct concise student response, optional memory/lore retrieval
     -> factual/procedure: lexical retrieval -> filtering -> calibrated evidence pre-gate
        -> insufficient: deterministic abstention
        -> sufficient: tiny student -> grounding diagnostics -> answer or abstain
```

The first retriever is SQLite FTS5 so the retrieval stack does not silently consume more memory than the generator. Neural retrieval/reranking is a later measured challenger, not an assumed requirement.

## Bootstrap

Bootstrap creates the student initialization, knowledge index, teacher candidates, and human review packet:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -TeacherModel .\artifacts\<accepted-teacher-checkpoint> `
  -AllowUnreleasedTeacher
```

Omit `-AllowUnreleasedTeacher` after Phase 8 is formally approved. Bootstrap never auto-approves teacher generations.

## Pilot

After editing the candidate JSONL so approved rows contain `review_status: accepted` and a selected/approved answer:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\<accepted-teacher-checkpoint> `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -MaxSteps 200 `
  -AllowUnreleasedTeacher
```

The pilot builds document/topic-disjoint splits, calibrates the evidence threshold on development data, trains matched SFT and SFT+KD candidates from the same student initialization, evaluates teacher and students under four RAG conditions, and writes `reports/PHASE_9_STOP_GATE.md`.

## Optional on-policy round

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\<accepted-teacher-checkpoint> `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -RunOnPolicy `
  -OnPolicyStudentFraction 0.20
```

Student-generated trajectories use `hard_label_weight=0`, preventing hard-label cross-entropy from reinforcing the student's own mistakes. The frozen teacher still supplies the distribution target.

## Preference alignment

After RAG SFT/KD is stable, create RAG-specific preference files and reuse Phase 3's DPO trainer against the selected full student checkpoint or student adapter. Preference pairs should cover supported versus unsupported claims, concise versus verbose answers, correct abstention, clean EOS, in-character behavior, and prompt-injection resistance. Do not apply preference tuning before evidence use is stable.

## Evaluation interpretation

- Poor oracle RAG: generator cannot use supplied evidence.
- Good oracle but poor real RAG: retrieval/filtering/gating is failing.
- High correctness but poor abstention: sufficiency training or the external gate is failing.
- Correct facts but excessive output: EOS/brevity training is failing.
- Good generator footprint but high total process memory: retriever, index, KV cache, or runtime overhead is failing.

No Phase 9 candidate advances solely because it is small. The shared relative-opportunity engine compares measured model-weight reduction with proportional quality loss, while human review inspects unsupported claims, false abstentions, role-play behavior, latency, total process memory, and quantized deployment evidence.
