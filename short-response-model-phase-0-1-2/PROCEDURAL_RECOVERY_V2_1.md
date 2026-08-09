# v2.1 — Phase 9 Procedural Transfer Recovery

## Why v2.1 exists

The packaged v2.0 Phase 9 stop gate contains one unusually informative failure. The selected `sft` student is highly compressed and broadly strong, but its procedural behavior collapses specifically when the same task is moved from supplied/Oracle evidence to Real RAG.

From `reports/PHASE_9_STOP_GATE.md` and `reports/PHASE_9_ASSESSMENT.json`:

| Measurement | Selected `sft` student |
|---|---:|
| Weight reduction vs teacher | 0.520189 |
| Generic relative quality loss | 0.006214 |
| Oracle supported correctness | 1.000000 |
| Real-RAG supported correctness | 0.359615 (187/520) |
| Real-RAG evidence faithfulness | 0.960905 |
| Real-RAG correct abstention | 0.764706 (104/136) |
| Real-RAG conflict detection | 0.645161 (40/62) |
| **Real-RAG procedure correctness** | **0.000000 (0/150)** |
| Real-RAG social usability | 1.000000 (20/20) |
| Real-RAG prompt support recall | 1.000000 |
| Prompt/decode integrity failures | 0 / 0 |

The stop gate therefore names a single explicit blocker: `real_rag demonstrates zero correct procedures`.

The package does not expose a separate Oracle-procedure scalar in the stop-gate report itself. The user's reported 1.0 Oracle procedural result is nevertheless consistent with the packaged Oracle-supported correctness of 1.0 and with the failure anatomy below. v2.1 therefore treats the first question as *Oracle → Real transfer*, not as “does the student know any procedure at all?”

## What the forensic audit found

A model-free postmortem of the already-exposed 150 answerable procedure cases in the historical Phase 9 TEST split is stored in:

- `reports/PHASE_9_PROCEDURAL_RETRIEVAL_COLLISION_AUDIT_v2.1.json`
- `reports/PHASE_9_PROCEDURAL_RETRIEVAL_COLLISION_AUDIT_v2.1.md`

The result is stronger than “retrieval might be weak”:

- answerable procedural cases: **150**;
- authored support retrieved in top 3: **150/150**;
- support rank 1: **143/150**;
- support rank 2: **7/150**;
- equivalent trusted procedure retrieved: **150/150**;
- legacy top-3 prompts containing same-topic sibling records: **150/150**;
- total same-topic sibling distractors in those top-3 prompts: **300**;
- query-aware v2.1 same-topic sibling collisions in the same model-free audit: **0/150**.

A representative legacy prompt shape is:

```text
Question: In Birch latency audit 5154, record 1, which relay is faster?

rank 1: audit 5154, record 1  <-- authored support
rank 2: audit 5154, record 2  <-- near-identical sibling, different values
rank 3: audit 5154, record 3  <-- near-identical sibling, different values
```

This means the old Real-RAG failure cannot be explained by “the retriever did not find the evidence.” It found the exact evidence. The more precise hypothesis is **evidence-binding interference**: the tiny student must bind the procedure to the exact requested record while multiple structurally identical records with conflicting numeric/name values occupy the same prompt.

This remains a hypothesis until the same-weight A/B evaluation is run. v2.1 explicitly measures that A/B before any new training.

## A second data problem found during the audit

The old v1.7.5 training corpus also contains a smaller but concrete supervision defect. `audit_procedural_gold_consistency_v21.py` finds **18 / 114** answerable latency-training rows where both target latencies are equal even though the trusted rule says to choose the *smaller* latency, while the gold still names one relay. For example, one historical training record contains 18 ms vs 18 ms but labels `Relay Birch.`.

Those rows are preserved as historical artifacts rather than silently rewritten. v2.1 does two things instead:

1. its new deterministic generator prohibits implicit ties unless a procedure explicitly defines the tie-break; and
2. its preservation slice excludes old procedure rows entirely, so these known-bad labels cannot leak into recovery training.

The new v2.1 training audit reports **0** such latency violations.

## Decision: what to do next

Do **not** spend the next cycle on another broad LoRA learning-rate/rank sweep, and do **not** immediately retrain every other student.

The next cycle should answer three questions in order:

1. **How much does retrieval/prompt selection alone fix, with identical weights?**
2. **How much additional gain comes from hard collision-aware procedural training?**
3. **Can that learned gain be kept without damaging the student's already-good behaviors?**

Only after those are answered should the same recipe be replicated on one or two other students to test generality.

## v2.1 recovery architecture

### Stage 0 — same-weight retrieval A/B

Evaluate the current student three ways before changing weights:

```text
A. original Phase 9 DEV + legacy retrieval
B. SAME weights + original Phase 9 DEV + query_aware_v21 retrieval
C. SAME weights + new v2.1 procedural challenge DEV + query_aware_v21 retrieval
```

This separates a system failure from a model failure. If B jumps sharply above A, retrieval collision was a major cause. That gain must be reported as a **retrieval-policy gain**, not as learned student improvement.

### Stage 1 — collision/evidence-binding repair

Train one focused rank-16 LoRA path rather than another large sweep:

- train view: `train_answerable_hard.reviewed.jsonl`;
- target shape: exact requested record + shuffled sibling records;
- target record is intentionally moved across prompt positions;
- equivalent duplicate procedures may also appear;
- default: 250 optimizer steps;
- default LR: `3e-5`;
- first-answer and evidence-anchor token weights: `2.25`.

If the current student is already represented by a LoRA adapter, v2.1 continues that adapter rather than stacking another independent adapter. If the current student is a full checkpoint, v2.1 adds a new rank-16 recovery adapter so the repair remains isolated and reversible.

### Stage 2 — consolidation + preservation

Continue the Stage-1 adapter on:

- all v2.1 train procedures, including supported/missing/conflict cases; plus
- a deterministic **train-only, non-procedural** preservation slice from the previous Phase 9 corpus.

Defaults:

- 350 optimizer steps;
- LR `2e-5`;
- no new adapter/rank change.

This is intended to keep the procedural repair from buying gains by destroying supported factual, abstention, conflict, or social behavior.

### Stage 3 — decompose the gain

`compare_procedural_recovery_v21.py` reports:

```text
retrieval-policy gain
= same weights/query-aware - same weights/legacy

training-only gain
= recovered/query-aware - baseline/query-aware

combined gain
= recovered/query-aware - baseline/legacy
```

It separately reports preservation losses. This is important research evidence: a code-side retrieval fix and a learned procedural repair are both useful, but they are not the same result.

### Stage 4 — sealed test only after DEV selection

The newly generated v2.1 TEST topics are not used for training or selection. Run them only after the DEV comparison is frozen. The historical Phase 9 test has already been exposed through the 0/150 postmortem, so it is a regression set for v2.1, not a pristine new sealed set.

## New dataset

`data/phase_9_student_distillation/procedural_recovery_v21`

| Asset | Count |
|---|---:|
| Canonical procedural rows | 6,120 |
| Train | 4,648 |
| Dev | 736 |
| Test | 736 |
| Factual knowledge documents | 2,880 |
| Trusted/equivalent procedure records | 1,560 |
| Hard answerable train rows | 4,104 |
| Non-procedural preservation rows | 1,162 |
| Final recovery mix rows | 5,810 |
| Procedure families | 14 |

Families include semantic-version comparison, date comparison, latency selection, priority selection, capacity fitting, quality-then-budget filtering, threshold decisions, inclusive ranges, approval routing, exponential backoff, deployment prerequisites, explicit score/latency tie-breaking, absolute tolerance, and quota arithmetic.

The important increase is not merely row count. The new data deliberately trains **near-miss binding**:

```text
same audit + record 1 + values A
same audit + record 2 + values B
same audit + record 3 + values C
question asks for exactly record 2
```

That pattern directly mirrors the observed Real-RAG failure.

## Query-aware retrieval selection

`phase9_common.py` now provides `query_aware_v21` passage selection.

It is deliberately conservative:

- it recognizes only explicitly named locator types such as `record 2`, `audit 5154`, `batch X`, `ticket Y`, `run Z`;
- ordinary numbers such as dates, prices, versions, and measurements are **not** treated as hard retrieval filters;
- when an exact explicit locator match exists, passages with conflicting explicit locators are suppressed;
- identical procedure text is deduplicated so duplicate copies do not consume the procedure budget;
- the original FTS normalized score is preserved for answerability calibration; the v2.1 selection score is a separate ordering diagnostic.

The low-level Phase 9 evaluation/calibration tools keep `legacy` as their default for historical reproduction. The official Phase 9 PowerShell wrappers and the dedicated v2.1 recovery runner now explicitly default to `query_aware_v21` (with a `legacy` override). Phase 11 production calibration, benchmark, and service paths also default to the new policy so the production runtime does not reintroduce the same collision.

## Should other students receive the same training?

**Not yet as a mass rerun.** First establish causality on this student because it has the strongest evidence that the underlying capability exists and the transfer path is broken.

After v2.1 is measured, run the fixed recipe on **one or two** other students as a replication study:

- same v2.1 data;
- same rank/LR/steps;
- same query-aware retrieval;
- no per-student hyperparameter tuning initially.

That tells us whether the procedural recovery pattern generalizes. If all students improve similarly, promote it into the standard Phase 9 curriculum. If only this student improves, the result is architecture/lineage-specific and should stay a recovery branch.

## Should we try more LoRA rates/ranks?

Not as the next action. The current evidence makes a broad sweep a weak use of compute because:

- the exact support is already retrieved;
- the trusted procedure is already retrieved;
- the reported Oracle capability is strong;
- every historical Real-RAG procedure prompt had the same sibling-collision structure.

v2.1 therefore fixes rank 16 and uses two conservative learning rates (`3e-5` then `2e-5`). A rank-32 or higher-LR rescue should be considered **only if**:

1. same-weight query-aware retrieval does not solve enough of the gap;
2. Stage A/B training loss and DEV results demonstrate genuine underfitting; and
3. retrieval/prompt diagnostics remain healthy.

If those conditions are not met, another rank/LR sweep is more likely to obscure the mechanism than teach us something useful.

## Relationship to v2.0 student optimization

v2.0 is still the correct downstream structural/runtime optimization program. The order changes only because the accepted-student premise is currently false for the procedural contract:

```text
Phase 9 current student
    ↓
v2.1 procedural diagnosis + recovery
    ↓
Phase 9 human approval of recovered student
    ↓
Phase 9B if required for the selected lineage
    ↓
v2.0 Student Optimization S0–S8
    ↓
Phase 10 / Phase 11 release work
```

Do not delete layers, shrink width, introduce contextual sparsity, or choose final quantization while the critical procedural behavior is still 0/150. Behavior repair is cheaper to interpret before architectural damage is introduced.

## Main command

For a full student checkpoint:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_procedural_recovery_v21.ps1 `
  -StudentModel "C:\path\to\current_student_checkpoint" `
  -RequireCuda
```

For a base checkpoint + existing selected student adapter:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_procedural_recovery_v21.ps1 `
  -StudentModel "C:\path\to\student_base" `
  -StudentAdapter "C:\path\to\selected_student_adapter" `
  -RequireCuda
```

Do DEV diagnosis/recovery first. Only after the DEV comparison is reviewed:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_procedural_recovery_v21.ps1 `
  -StudentModel "C:\path\to\student_base_or_checkpoint" `
  -StudentAdapter "C:\path\to\selected_student_adapter_if_any" `
  -OutputRoot ".\artifacts\phase_9_procedural_recovery_v21_YYYYMMDD_HHMMSS" `
  -ReuseExisting `
  -RunSealedTest `
  -RequireCuda
```

The runner validates/generates the corpus, audits known historical gold defects, builds both indexes, performs the same-weight legacy/query-aware A/B, diagnoses Oracle→Real failures, trains the focused two-stage repair, reevaluates the original DEV and v2.1 challenge DEV, and writes a comparison that separates retrieval gain from training gain.

## Files added or materially changed

See `CHANGED_FILES_v2.1.md` for the exact inventory.
