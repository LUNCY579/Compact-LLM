# Architecture review — Phase 10 and Phase 11 extension

## Review scope

This review was performed against the supplied `Output(9)` repository after its Phase 9 bug fixes. The review covered the phase handoffs, shared evaluator and opportunity policy, model-introspection and structural-reduction helpers, Phase 9 student creation/training/evaluation, PowerShell orchestration, artifact provenance, human gates, release packaging, and regression tests.

The repository is organized around two distinct products:

1. **Phases 0–8: teacher development.** These phases create, reduce, align, validate, and release the authoritative short-response Qwen teacher.
2. **Phases 9–11: student productization.** Phase 9 proves evidence-conditioned distillation; Phase 10 searches for a materially smaller architecture; Phase 11 turns the approved student into a measured local RAG runtime and release package.

The extension does not overwrite teacher artifacts. Every student handoff is explicit, hashable, human-approved, and rollback-aware.

## Latest-codebase findings

### Strong foundations retained

- The shared Evaluator v4.5 is used rather than introducing a Phase-specific language-polish gate.
- Promotion uses `scripts/common/opportunity_evaluation.py`; Phase 10 declares serialized deployed weights as its primary resource, while Phase 11 declares measured total-process memory.
- Existing depth removal uses the proven shared Qwen `layer_types`-aware helper.
- Phase 9 has a real warm-start path, same-tokenizer completion-only KD, reviewed teacher targets, lexical RAG, four-condition evaluation, and a human gate.
- The Phase 9 document-connected split fix prevents train/dev/test leakage through shared documents, including transitive overlap.
- Generated checkpoints remain outside the source archive and are reproduced on the target machine.

### Defects found and corrected in this extension

1. **Phase 10 recovery-name mismatch.** Raw summaries were labeled `raw_<name>` before recovery selection, while the recovery runner expected the underlying candidate folder `<name>`. Recovery would look in a nonexistent directory. The recovery selector now receives canonical candidate names; the final report still uses `raw_<name>` to distinguish unrecovered candidates.
2. **Residual profiling was first-layer-only.** Hidden-coordinate importance was collected only at decoder layer 0. It now aggregates residual-stream magnitude across every retained decoder block.
3. **Phase 11 conflict calibration.** Conflicting evidence could receive a high numerical answerability score during calibration. Curated conflicts now receive score zero, matching the runtime conflict branch.
4. **Fallback tokenizer mismatch.** A fallback model could be generated with the student tokenizer. The fallback model now loads and uses its own tokenizer and chat template.
5. **llama.cpp empty-prefill summary crash.** llama.cpp has no separate Transformers prefill probe, so the previous median calculation could receive an empty list. Median and P90 are now nullable diagnostics.
6. **Incomplete process-memory reporting.** Phase 11 now reports current RSS, peak process RSS where the OS exposes it, and the total-process-memory value used by the release comparison.
7. **Unbounded retrieved context.** The production path now enforces a tokenizer-measured evidence/procedure budget and records passages rejected by the budget.
8. **Quantized reload device mismatch.** Freshly reloaded bitsandbytes candidates now receive smoke-test tensors on the reloaded model's device.
9. **Overeager named-entity diagnostics.** Common sentence-initial words are excluded from the conservative proper-noun support check.
10. **No persistent production entry point.** Phase 11 now includes a long-lived JSONL/interactive process suitable for desktop-app integration without adding an HTTP framework.

## Phase 9 hybrid-suite compatibility review

The latest Phase 9 assessment can contain legacy names such as `kd` or profile-qualified names such as `kd_warm_344m`, `sft_narrow_100m`, and `kd_narrow_170m`. The Phase 10 handoff now resolves any exact key present in `PHASE_9_ASSESSMENT.json`; it no longer constrains the CLI to four legacy names. Candidates listed under `research_controls` cannot be handed off as deployment/intermediate artifacts without an explicit acknowledgement, while human reviewers may still preserve them as research-only references. Relative model and adapter paths are normalized against the repository root before materialization.

## Phase 10 architecture

### Inputs

- Human-approved Phase 9 student reference: `artifacts/references/phase_9_student_reference.json`
- Original authoritative teacher, retained as the supervising teacher at every recovery rung
- Phase 9 train/dev/test RAG records
- Phase 9 knowledge index and calibrated evidence gate

### Candidate path

```text
Accepted Phase 9 student
        ↓
Activation profile
  ├─ residual coordinates across all blocks
  ├─ gated-MLP channel contribution proxy
  └─ grouped-query attention-head contribution proxy
        ↓
Physical dense candidates
  ├─ complete-layer removal
  ├─ MLP row/column slicing
  ├─ GQA group slicing
  ├─ residual hidden-coordinate slicing
  └─ coordinated combinations
        ↓
Fresh save/reload/smoke validation
        ↓
Four-condition raw evaluation
        ↓
Bounded candidate selection for recovery
        ↓
Original-teacher completion-only KD recovery
        ↓
Four-condition recovered evaluation
        ↓
Relative opportunity frontier + human selection
```

### Target bands

The full candidate matrix contains explicit experiments for:

- intermediate-teacher band: approximately 180M–300M parameters;
- small-student band: approximately 100M–180M parameters;
- deployment band: approximately 80M–120M parameters.

Those are target bands, not promises. The actual count is written from real tensors, and a candidate outside its band remains evidence rather than being mislabeled.

### Structural validity boundary

Layer, MLP, attention, and hidden slicing create physically smaller dense tensors. The hidden-coordinate operation is an activation-ranked coordinate slice, **not** a complete SliceGPT orthogonal-invariance reproduction. It is intentionally treated as an aggressive candidate requiring recovery and evaluation. A technically loadable model is not automatically a good model.

### Compact challenger

An optional pretrained compact source, such as a SmolLM-family instruct checkpoint, can be cloned and trained with sequence-level RAG SFT. Direct Qwen vocabulary KL is disabled for cross-tokenizer challengers. Cross-tokenizer GOLD remains an optional advanced experiment, not a hidden requirement for Phase 10 completion.

## Phase 11 architecture

### Runtime path

```text
Request
  ↓
Auditable mode router
  ├─ social/role-play
  ├─ grounded factual
  ├─ trusted procedure
  ├─ high risk
  └─ complex/fallback
  ↓
SQLite FTS5 retrieval where applicable
  ↓
Sanitization + tokenizer context budget
  ↓
Question/evidence answerability gate
  ├─ missing/weak → deterministic abstention
  ├─ conflict → deterministic conflict response
  └─ sufficient → selected student
  ↓
One-sentence response control
  ↓
Numeric/entity/polarity grounding checks
  ├─ supported → return
  └─ hard unsupported claim → replace with abstention
```

### Production artifacts

- immutable knowledge/procedure JSONL snapshot;
- rebuilt SQLite FTS5 index;
- snapshot manifest and hashes;
- development-calibrated answerability policy;
- dense reference runtime measurement;
- deployment candidate runtime measurement;
- quantization manifest;
- release candidate registry;
- human stop gate;
- manifest-only or self-contained copy release ZIP;
- local persistent JSONL/interactive runtime process.

### Quantization/runtime choices

Supported deployment candidates are:

- dense checkpoint copy;
- bitsandbytes NF4 directory;
- bitsandbytes INT8 directory;
- llama.cpp GGUF using caller-supplied conversion, quantization, and runtime executables.

A dense release remains valid when quantization does not reduce measured total process memory or causes disproportionate quality loss. Idealized four-bit arithmetic never substitutes for a measured artifact and runtime.

## Human and technical gates

Technical invalidity remains binary: missing checkpoints, incompatible tensors/tokenizers, failed fresh reload, corrupt indexes, nonfinite outputs, incomplete coverage, or missing provenance cannot be promoted.

Quality and efficiency are compared proportionally. Phase 10 and Phase 11 do not introduce private 0.90/0.95/0.99 quality floors. A product-specific unsupported-confident-answer cap may be supplied only when deliberately calibrated and documented; by default it is unset.

Both phases write `PENDING` gates. The scripts never self-approve.

## What this source release proves

The source release proves that:

- the Phase 10/11 orchestration and handoffs are implemented;
- structural operations reduce real tensor shapes in unit tests;
- candidate/release policy is centralized and auditable;
- runtime and packaging paths exist;
- source compiles and the full regression suite passes.

It does **not** prove that an 80M–120M candidate will preserve sufficient quality, that KD will recover a particular architecture, that NF4/GGUF will win on the user's RTX 4070, or that a production knowledge base has adequate retrieval recall. Those are machine- and data-dependent experimental results that must be generated by running the phases.
