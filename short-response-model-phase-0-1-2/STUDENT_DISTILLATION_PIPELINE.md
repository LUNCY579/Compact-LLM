# Student Distillation Pipeline — appended Phase 9 workflow

Version 1.7.6 | August 2026

## v1.7.6 exploratory student playground amendment

A separate `Student_Distillation_Playground_v1.0/` now provides non-gating logic, supplied-evidence, and source-state-aware memory demonstrations. It reuses the official Phase 9 generation path and Phase 9/9B prompt builders, but writes only to `playground_artifacts/` and explicitly marks every summary as `official_gate_evidence: false`.

Recommended timing: first finish Phase 9 through the selected student's full Oracle/Real/adversarial evaluation and human selection, then record the playground before Phase 10. Memory reconstruction in two-pass mode is best demonstrated after Phase 9B training/evaluation. Playground cases must not be used to tune/select an official candidate and then be represented as held-out official evidence.

## Phase 9 grounded-recovery amendment — v1.7.2

The official recovery path is now capability-first rather than architecture-first. It evaluates the accepted teacher with Oracle evidence, records the untrained warm_344m Oracle baseline, requires a deliberate 12–24-row overfit wiring proof, runs only three warm SFT settings, and unlocks Real RAG and progressive KD only after Oracle capability improves. The narrow 100M/130M/170M profiles remain research controls and are blocked until the ladder JSON explicitly unlocks architecture comparison.

Training uses a versioned 45/25/10/10/5/5 behavior mixture and simple completion-token emphasis for the first answer token, evidence anchors, abstention tokens, and the completion boundary. Evaluation keeps numerator/denominator evidence and reports candidate-versus-teacher, SFT-versus-initialization, KD-versus-SFT, and KD-versus-initialization separately.

## Phase 9 RAG integrity amendment — v1.7.1

The current gate treats prompt construction and decoding as measured parts of the experiment. Retrieved support and prompted support are separate metrics. Tokenizer truncation is forbidden for official Phase 9 generation; passages are budgeted before chat rendering, while the grounding contract and final question are preserved. Training uses the same rule and skips a row rather than retaining only an ungrounded tail.

Promotion combines generic relative quality loss with a bounded task-contract deficiency. Core RAG dimensions use a proportional bottleneck cost, so complete failure on supported correctness or unsupported-answer control contributes a full functional cost rather than being averaged away by compression. This also adds direct cost for missing grounding, conflict failure, prompt/decode corruption, and adversarial instruction following. The teacher must pass a raw-output preflight before student training begins.


## Behavior Curriculum v2 amendment

The first hybrid smoke run proved that the pipeline executed but did not prove that any student learned evidence use: every candidate reported zero supported Real-RAG successes, and the old cross-profile calculation became numerically unstable when a reference metric was zero. The official Phase 9 implementation is therefore amended as follows:

- a minimum-size, class-covered real data gate precedes training;
- a versioned fictional behavior foundation supplements—but never replaces—human-reviewed real rows;
- curriculum SFT precedes KD;
- progressive KD begins from the SFT artifact, uses a hard-label warmup, filters low-confidence teacher tokens, and emphasizes the completion boundary;
- direct KD from initialization is an optional ablation;
- social conversation is evaluated separately from unsupported factual questions;
- zero denominators produce `null` metrics;
- functional correctness uses explicit response contracts plus grounding;
- bounded absolute deltas prevent division-by-zero explosions in training comparisons;
- smoke evaluations are never production-recommendable;
- adversarial evaluation distinguishes system-filtered evidence from malicious evidence passed through to the model.

The bundled synthetic foundation contains fictional, authored evidence/answer pairs. It teaches reusable evidence behavior and concise conversation, not broad language competence. Narrow randomly initialized students remain research controls until separate pretraining or language-readiness evidence exists.


## Phase 9B extension — source-state-aware memory reconstruction

Phase 9 proves that the student can answer from supplied evidence. Phase 9B tests a stricter capability: whether the student can compare the historical state and validity conditions of a retrieved memory with the current state, then accept, adapt, reject, flag conflict, or abstain before answering. Semantic retrieval remains external; reconstruction is an inference-time model operation that can be taught through reviewed SFT/KD and later distilled into one pass.

Phase 9B does not alter the accepted Phase 9 artifacts. It writes a separate reference manifest and retains the exact Phase 9 student as rollback. See `MEMORY_RECONSTRUCTION_PIPELINE.md`.

## 1. Relationship to the official teacher pipeline

The official Phase 0–8 pipeline remains the authoritative teacher-development program:

```text
Phase 0   Data contract and leakage control
Phase 1   Dense short-response SFT
Phase 2A  Selected-layer training study
Phase 2B  Progressive structural depth reduction
Phase 2C  Targeted low-resource recovery
Phase 3   Preference alignment
Phase 4   Compatible model soups
Phase 4B  Post-alignment structural revalidation
Phase 5   Context retrieval and prefix/KV management
Phase 6   Sparse-inference simulation
Phase 7   Measured runtime implementation
Phase 8   Sealed release audit and human approval
```

Phase 9 is appended after Phase 8. It does not rewrite, bypass, or replace the teacher phases. Its input is an exact, human-approved Phase 8 teacher checkpoint plus the teacher tokenizer, response-control contract, evaluator version, and provenance manifests.

The teacher pipeline optimizes the best practical short-response model derived from Qwen3-0.6B. The student pipeline asks a different question:

> How small can a model become when broad factual memory is moved into retrieval, while the model retains language understanding, evidence use, concise conversation, role-playing, reusable procedures, and safe abstention?

## 2. Product contract

The student is not expected to be a miniature closed-book general assistant. Its weights specialize in reusable behavior:

```text
Student weights                    External retrieval memory
-------------------------------    --------------------------------
language and question understanding factual knowledge
evidence selection and synthesis    product and project documents
procedure application               character and world lore
conversational behavior             user memories
role-playing behavior               recent/changing information
concise response and natural EOS     long-tail entities
abstention and conflict behavior     trusted domain procedures
```

The deployment target is a fast, batch-one, output-light model that usually emits fewer than 32 tokens. The first practical target is hundreds of megabytes unquantized or roughly 40–70 MB of idealized INT4 weights for an 80M–120M architecture. Actual process memory must include weights, quantization metadata, KV cache, retriever, index, tokenizer, buffers, and application overhead.

## 3. Runtime architecture

```text
User message
    ↓
Mode routing
    ├── Social / role-play
    │      ↓
    │   Optional user-memory or lore retrieval
    │      ↓
    │   Tiny student produces a natural concise reply
    │
    ├── Grounded factual
    │      ↓
    │   Query normalization
    │      ↓
    │   Retrieval → filtering → optional reranking
    │      ↓
    │   Calibrated evidence-sufficiency pre-gate
    │      ├── Insufficient → deterministic abstention
    │      └── Sufficient → tiny student
    │                         ↓
    │                    grounding diagnostics
    │                         ├── supported → answer
    │                         └── unsupported → abstain/review
    │
    ├── Procedure / task
    │      ↓
    │   Trusted procedure retrieval + factual evidence retrieval
    │      ↓
    │   Student applies the procedure only to supported facts
    │
    └── Complex / high-risk
           ↓
        clarify, abstain, or route to a larger approved model
```

The initial router is deliberately lightweight. Deterministic rules and explicit dataset modes are acceptable for the pilot. A separate neural router is added only when measured errors justify its memory and latency.

## 4. Student tracks

### Track A — warm-started Qwen-compatible intermediate

This is the official first track. It preserves the teacher tokenizer and hidden width and copies complete selected teacher layers. It validates the distillation objective without asking a randomly initialized tiny network to relearn language from a small RAG dataset.

Recommended first rung:

```text
teacher: accepted modified Qwen3-0.6B checkpoint
student: same tokenizer and width, reduced depth
purpose: prove transfer, evidence use, abstention, and evaluation
```

### Track B — aggressive same-tokenizer Qwen control

This constructs a narrow Qwen3 architecture, for example hidden size 384 with 16 layers. It is randomly initialized unless a separate structured projection or continued-pretraining process is supplied. It is a research control, not a production candidate by default.

The Qwen vocabulary is large. With vocabulary 151,936 and hidden size 384, the tied embedding/output matrix alone contains about 58.3M parameters. This makes same-tokenizer KL easy but leaves relatively little transformer capacity. The same-tokenizer track is therefore a proof path, not an assumption that the final tokenizer is optimal.

### Track C — pretrained compact challenger

A compact pretrained model such as SmolLM2-135M is trained first with sequence-level RAG SFT because its tokenizer differs from Qwen. Cross-tokenizer probability distillation is a later, separately validated experiment. The challenger measures quality per runtime megabyte against the Qwen-compatible students.

### Compression ladder rule

Intermediate students may initialize or assist later students, but the original accepted teacher remains an authoritative supervisor at every rung:

```text
Accepted teacher ───────→ Intermediate student
       │                         ↓
       ├──────────────────→ Small student
       │                         ↓
       └──────────────────→ Deployment student
```

This avoids compounding approximation errors through a teacher chain.

## 5. Data contract

### Knowledge records

Each factual record contains a stable document ID, text, source, version/update timestamp, and optional topic/group ID. Procedures are stored separately and must come from trusted curated sources.

### Prompt records

Every prompt has a stable ID, split group, mode, question, expected decision, oracle evidence/procedures when available, and explicit response contract. Dataset splitting is by document, topic, or source group—not random question rows—so nearly identical evidence cannot leak across train/dev/test.

### Required categories

The pilot includes supported answers, missing answers, irrelevant retrieval, distractors, partial support, conflicting sources, malicious instructions inside evidence, social/role-play cases, and procedure cases. The exact proportions are diagnostics and may be adjusted from measured failures; they are not universal fixed standards.

### Teacher records and review

Teacher generation stores multiple candidates, evidence IDs, grounding diagnostics, quality diagnostics, stop reason, and provenance. No generated row becomes training data until a human marks it accepted and selects an approved answer. Rejected candidates remain preserved for preference construction and audit.

## 6. Stable prompting contract

Teacher generation, SFT, KD, on-policy training, evaluation, and production use the same logical structure:

```text
SYSTEM: concise evidence-grounded behavior and abstention rules
<TRUSTED_PROCEDURES> ... </TRUSTED_PROCEDURES>
<FACTUAL_EVIDENCE> ... </FACTUAL_EVIDENCE>
<QUESTION> ... </QUESTION>
<ANSWER>
```

Procedures and factual evidence remain separate. Instructions embedded inside retrieved documents are untrusted data and cannot redefine system behavior. Social/role-play mode uses its own concise conversational contract rather than forcing an evidence abstention response for every ordinary message.

## 7. Retrieval baseline

Phase 9 starts with SQLite FTS5 lexical retrieval because a neural embedding model or cross-encoder can consume more memory than the tiny generator. The complete process records generator memory, retriever memory, index size, reranker memory, and total RSS.

The external gate does not treat raw lexical similarity as proof that an answer exists. It combines calibrated retrieval evidence with answerability labels and conflict handling. The threshold remains unset until development-set calibration. Any change to index content, tokenizer, query normalization, retriever, or score normalization requires recalibration.

Neural retrieval, vector search, and reranking are challenger configurations added only after the lexical baseline identifies a measurable recall or ranking bottleneck.

## 8. Official Phase 9 subphases

### D0 — teacher freeze and compatibility audit

Record the exact teacher path, tokenizer fingerprint, parameter count, accepted Phase 8 gate, response-control mode, evaluator version, and baseline RAG behavior. The student pipeline refuses an unapproved teacher unless the operator explicitly uses a research-only override.

### D1 — student initialization

Create at least:

1. a warm-started depth-copy student;
2. optionally, an aggressive random narrow-Qwen control;
3. later, a pretrained compact challenger.

Every initialization writes a manifest containing configuration, copied layers, tokenizer fingerprint, parameter breakdown, idealized weight sizes, and initialization status.

### D2 — knowledge/procedure index

Validate JSONL records, reject duplicate IDs, build the FTS5 index, and write an index manifest. Knowledge and procedure stores remain versioned and auditable.

### D3 — teacher data generation and review

Generate multiple concise candidates with thinking disabled where supported. Evaluate evidence support, observable defects, EOS/response control, and provenance. Produce both JSONL evidence and a Markdown review packet. Human acceptance is mandatory.

### D4 — disjoint pilot construction and readiness

Convert accepted teacher rows into chat-formatted records. Create connected-component train/dev/test splits over topics and support/retrieved documents, then verify no leakage. The research path requires at least 30 accepted rows, eight independent connected components, and development/test coverage for both supported and unsupported cases. A six-row packet is allowed only with an explicit smoke flag and cannot support promotion.

### D5 — behavior curriculum and evidence-gate calibration

Create controlled variants from human-approved answers: concise prompt variants, distractors, missing-evidence abstentions, retrieval-injection examples, and social rehearsals. Materialize the bundled fictional behavior foundation, expand it with the same controlled transforms, then deterministically mix it with a higher repeat weight for real reviewed rows. The foundation is training-only.

Search the real development records, sweep candidate thresholds, and report supported-question recall, unsupported rejection, false-answer risk, unnecessary abstention, and every denominator. Product costs determine the operating point. No arbitrary universal score is embedded in code.

### D6 — curriculum SFT then progressive KD

The official comparison is sequential rather than direct-KD-first:

```text
Initialization
   └─→ Candidate A: completion-only curriculum SFT
                         └─→ Candidate B: confidence-filtered completion-only KD

Optional ablation: direct KD from initialization
```

Progressive KD verifies exact tokenizer fingerprints, begins with a hard-label-only warmup, gradually shifts weight toward KL, ignores teacher tokens that assign insufficient probability to the approved target, and falls back to hard labels when the teacher disagrees. Completion positions use `logits_to_keep` where supported. Targets exceeding the completion budget are skipped rather than mechanically truncated. The EOS/end-of-turn boundary receives extra loss weight. Batch size one plus gradient accumulation remains the official low-memory baseline.

### D7 — five-condition evaluation

Evaluate teacher and students under:

- closed book;
- oracle RAG;
- real RAG;
- adversarial RAG with the sanitizer active;
- adversarial passthrough RAG with malicious evidence retained.

Report retrieval recall, evidence coverage, explicit numerator/denominator pairs, supported correctness, faithfulness, unsupported-confident-answer rate, abstention, conflict detection, social usability, distractor and injection robustness, median/P90 completion tokens, EOS success, repetition, broken words, foreign-character leakage, latency, model bytes, peak generator memory, index bytes, and total process RSS. Zero-applicability metrics are `null`; they are never fabricated by dividing by one.

Interpret failures causally:

```text
poor oracle RAG                → generator/evidence-use failure
good oracle, poor real RAG     → retrieval/filtering/gate failure
good answers, weak abstention  → answerability/gate failure
correct but verbose            → EOS/preference failure
good model, high total memory  → retrieval/KV/runtime failure
```

### D8 — optional on-policy KD

Generate a controlled mixture of teacher-reference and student-produced trajectories. Student trajectories carry hard-label weight zero so the student is not rewarded for copying its own mistakes; the frozen teacher supplies the distribution target along those states. Begin around 20% student trajectories and increase only after measured improvement.

### D9 — preference alignment

After evidence use and abstention are stable, reuse the existing Phase 3 DPO machinery with student-specific pairs:

- supported versus unsupported answer;
- evidence-based versus memorized claim;
- concise versus unnecessarily verbose;
- correct abstention versus invention;
- clean EOS versus continuation;
- in-character versus generic language;
- injection-resistant versus evidence-instruction-following.

Preference tuning is not allowed to conceal degraded grounding.

### D10 — proportional gate and human selection

The shared opportunity engine compares each student with the teacher and with same-architecture baselines. Primary benefit is measured model-weight reduction; quality opportunity loss includes oracle/real/adversarial behavior, faithfulness, abstention, tail quality, and brevity. Same-architecture SFT versus KD/on-policy comparisons use quality improvement at unchanged architecture.

No private absolute quality floor is introduced. An explicit unsupported-confident-answer cap may be configured only as a declared, calibrated product-safety constraint. Human review names the selected artifact and classifies it as production candidate, intermediate teacher, or research-only control.

### D11 — quantization and measured deployment

Quantize only an accepted full or merged student. Measure actual checkpoint bytes and complete runtime memory. Compare dense unquantized, dense quantized, and any optional sparse variant on the target hardware. Idealized parameter arithmetic is reported only as an estimate, never as measured deployment memory.

### D12 — compact-tokenizer challenger

After same-tokenizer KD is proven, test a compact pretrained tokenizer/model. Use sequence-level SFT first. Cross-tokenizer span-alignment distillation is experimental and must have its own tokenizer-alignment tests and ablations.

## 9. Commands

Bootstrap initialization, index, teacher candidates, and review packet:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model
```

After human review:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -MaxSteps 200
```

Optional on-policy recovery:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Pilot `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -ReviewedTeacherData .\artifacts\phase_9_student_distillation\teacher_candidates.reviewed.jsonl `
  -RunOnPolicy `
  -OnPolicyStudentFraction 0.20
```

Use `-AllowUnreleasedTeacher` only for research before the official Phase 8 human gate. Use `-AcknowledgeRandomControl` only for the explicitly random narrow-Qwen control.

## 10. Immediate milestone and decision

The first official experiment produces four deliverables:

1. an inherited Qwen-compatible student plus an optional random control;
2. a reviewed, document/topic-disjoint RAG pilot;
3. oracle and real-RAG evaluation harnesses;
4. a matched SFT versus completion-only KD comparison.

The first research question is:

> Can a warm-started 80M–300M Qwen-compatible student use supplied evidence, abstain appropriately, and remain concise—and does completion-only teacher-distribution imitation preserve that behavior better than ordinary SFT at the same architecture and training budget?

Only after this question is answered should the project invest in hidden-state alignment, attention matching, neural rerankers, compact-tokenizer migration, or further structural pruning.

## 11. Implemented scope in v1.5.0

Implemented now:

- explicit student creation with depth-copy, random-control, and pretrained-clone modes;
- low-memory SQLite FTS5 knowledge/procedure index;
- mode routing, passage sanitization, stable prompts, and evidence gate;
- teacher candidate generation with review packet;
- accepted-row pilot construction and group-leakage checks;
- dev-set evidence-threshold calibration;
- completion-only SFT and same-tokenizer online KD;
- optional mixed on-policy trajectories;
- closed/oracle/real/adversarial evaluation;
- proportional assessment and pending human gate;
- PowerShell orchestration, sample data, tests, and documentation.

Deferred until baseline evidence justifies them:

- learned neural reranker;
- cross-tokenizer GOLD-style training;
- hidden-state or attention-relation distillation;
- compact tokenizer construction;
- automatic DPO pair generation;
- production quantizer integration;
- target-runtime native kernels.

## 12. v2.2 capability-composition research branch

After the v2.1.x procedural/reasoning experiments, v2.2 adds a governed capability-composition branch before Student Optimization v2. This branch does not overwrite the established Phase 9 release contract; it asks whether the same tiny student can retain specialized behavior while gaining broader prompt-contained reasoning.

The branch is intentionally parallel rather than sequential:

```text
exact selected student
        |
        v
materialized frozen anchor
        |
        +--> Track A: broad multi-domain LoRA
        |
        +--> Track B: broad + supervised preservation replay + frozen-anchor KL
        |
        +--> Track C: independent general / grounding / procedure / abstention / social LoRAs
        |
        +--> Track D: weighted / greedy / layer-separated compositions
        |
        v
four-axis DEV capability matrix + Pareto frontier
```

The v2.2 canonical corpus contains 40,500 prompt-contained examples across 27 domains and 15 reasoning archetypes. It additionally includes 3,000 TRAIN-only conversational rehearsal rows used by the social specialist. No canonical DEV/TEST row enters any training or composition-search view.

Every candidate is measured on:

1. v2.2 27-domain direct reasoning;
2. v2.1.2 general-reasoning continuity;
3. original Phase 9 Real-RAG under `query_aware_v21`;
4. v2.1.1 procedural challenge under `query_aware_v21`.

Protected Phase 9 metrics must be present and non-null. Missing preservation evidence is technical invalidity, never zero quality loss. This strict rule is also backported to the v2.1.2 selector.

Primary runner:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_capability_composition_v22.ps1 `
  -StudentModel "C:\path\to\student" `
  -StudentAdapter "C:\path\to\original_adapter" `
  -RequireCuda
```

Track B defaults to `-AnchorDevice auto`; use `-AnchorDevice cpu` only when lower GPU pressure is worth the substantial speed cost.

The first run is DEV-only. Sealed TEST may be run later with the same output root and `-ReuseExisting -RunSealedTest` only after the DEV-selected winner is frozen and policy sensitivity is resolved.

See `CAPABILITY_COMPOSITION_V2_2.md` for the full experiment contract.
