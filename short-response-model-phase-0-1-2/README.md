# Short-Response Model — runnable mature phase pipeline

## v1.6.0 appended Phase 10 architecture search and Phase 11 production RAG release

Phase 10 now consumes the exact human-approved Phase 9 student, profiles residual/MLP/attention structure, creates physically smaller dense candidates across intermediate, small, and 80M–120M deployment target bands, performs fresh reload checks, recovery-distills a bounded set under the original teacher, optionally trains a compact pretrained challenger, and selects from a shared relative-opportunity frontier.

Phase 11 consumes the exact human-approved Phase 10 student and implements an immutable knowledge snapshot, SQLite FTS5 retrieval, development-calibrated answerability, deterministic abstention/conflict handling, evidence-token budgeting, conservative post-generation grounding, dense/NF4/INT8/GGUF deployment candidates, total-process memory and end-to-end benchmarks, a persistent local JSONL runtime, human release gate, rollback evidence, and release packaging. See `PHASE_10_11_ARCHITECTURE_REVIEW.md` and `PHASE_10_11_WORKFLOW.md`.

## v1.5.0 appended Phase 9 student distillation pipeline

The official Phase 0–8 program remains the teacher-development pipeline. Phase 9 now consumes an exact human-approved teacher and builds smaller RAG-specialized students through warm-started student construction, reviewed teacher-data generation, document/topic-disjoint pilot data, calibrated evidence gating, matched SFT versus completion-only logit distillation, optional on-policy recovery, four-condition RAG evaluation, and proportional human-gated selection.

The first retriever is SQLite FTS5 so generator savings are not silently offset by a neural retrieval stack. Direct vocabulary KL requires identical tokenizer fingerprints; cross-tokenizer challengers start with sequence-level SFT. Randomly initialized narrow-Qwen models are explicitly marked research controls. See `STUDENT_DISTILLATION_PIPELINE.md` and `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`.

The latest Phase 9 runner also supports `-StudentSuite Hybrid`, which evaluates one warm 344M depth-copy student and 100M/130M/170M narrow same-tokenizer research controls in a single assessment. Candidate keys are profile-qualified (`sft_warm_344m`, `kd_warm_344m`, and so on), and the Phase 10 handoff accepts the exact key recorded in `reports/PHASE_9_ASSESSMENT.json`. Narrow random-initialized profiles remain research controls unless a separate human language-readiness review explicitly authorizes promotion.

## v1.4.7 Phase 2C reliability and correctness hotfix

v1.4.7 fixes the repeated Windows file-lock failures and closes a Phase 2C evaluation-validity gap. Phase 2C deterministic reasoning rows now carry explicit prompt-grounded correctness contracts, so fluent but wrong arithmetic, ordering, comparison, implication, contradiction, set-membership, and temporal answers cannot receive high scores merely for being short and well formed.

Existing v1.4.6 Phase 2C work is preserved through a narrowly authorized migration: completed adapters and raw generated replies are reused, replies are rescored under the corrected contracts, teacher-forced NLL rows are reused, partial evaluations continue by stable ID, and affected output comparisons are rebuilt. Run the original Phase 2C wrapper with identical settings plus `-Resume`; do not delete the current artifact root or run child evaluators manually.

Reliability hardening also includes bounded atomic-replace retries for transient Windows locks, best-effort progress sidecars, resumable generation and teacher forcing, normalized manifest paths, strict stable-ID coverage, corrupt-sentinel quarantine, safer staged adapter publication, default suppression of intermediate checkpoints, free-disk checks, stronger Phase 2C artifact/gate audits, and `None`-safe report rendering. See `PHASE_2C_RELIABILITY_AND_CORRECTNESS_HOTFIX_v1.4.7.md`.

This repository implements the ordered Phase 0–8 teacher program, two structural teacher branches, targeted recovery, and an appended Phase 9–11 student distillation, architecture-reduction, and production-RAG pipeline:

- **Phase 2B progressive reduction** tests every current layer, recovery-tunes promising deletions, and preserves full-quality/balanced/minimum-memory model profiles before DPO.
- **Phase 2C targeted recovery** ranks surviving Phase 2B layers and trains selected-layer adapters whose proportional training-resource savings exceed their proportional quality opportunity loss.
- **Phase 4B post-alignment structural reduction** revalidates or further refines the accepted Phase 4 architecture before context/sparsity calibration.
- **Phase 9 student distillation** proves evidence-conditioned SFT/KD transfer into a warm-started student.
- **Phase 10 student architecture search** creates and recovers physically smaller dense students and compact challengers.
- **Phase 11 production RAG** measures and packages the complete grounded local runtime.

The post-Phase-3 runners execute training, merging, pruning, evaluation, profiling, router training, benchmarking, sealed evaluation, and packaging; they do not merely print instructions.

## v1.4.6 Phase 2B incremental evidence and reporting

Phase 2B no longer waits for the entire progressive sweep before producing useful evidence. Generation rows are appended and flushed after every completed benchmark example, live JSON is replaced atomically after every example, live Markdown is refreshed at a configurable interval, and every fully evaluated candidate is gated before it can participate in parent selection.

The run root now contains:

- `LIVE_PROGRESS.json` — machine-readable current stage, candidate, parent, counts, provisional quality, defects, and interruption/error state.
- `LIVE_PROGRESS.md` — human-readable polling view with an explicit provisional warning while evaluation is incomplete.
- `RUN_STATE.json` — durable orchestration/frontier state used by partial reporting.
- `PARTIAL_RUN_REPORT.md` — cumulative accepted/rejected candidate and completed/incomplete stage history before the final report exists.
- `evidence/stage_XX/stage_status.json` and `STAGE_XX_REPORT.md` — durable stage state and immediately available stage report.
- `evidence/stage_XX/<candidate>/candidate_status.json` and `CANDIDATE_REPORT.md` — candidate lifecycle, gate decision, promotion permission, metrics, errors, and artifact paths.

Reports can be rebuilt without loading a model:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2_structural_reduction.ps1 `
  -OutputRoot .\artifacts\<run-name> `
  -RebuildReport `
  -AllowPartialReport
```

The persisted promotion contract is strict: a candidate must have `status=accepted` and `promotion_allowed=true` before `progressive_reduction.py` can materialize it as the next parent. A rejected candidate cannot become the next pruning-stage parent.

## v1.4.5 fresh-message contract evaluation

Quality Evaluator v4.5 rebuilds every official Phase 2B response contract from the visible system and user messages. It does not consult `category`, the reference answer, or inherited v4.1–v4.4 contracts. Categories remain reporting-only tags for benchmark coverage and regression analysis.

The release fixes false critical and major labels found in real Phase 2B output: common function words can no longer trigger a token-loop collapse by themselves; uncertainty, visibility limits, generic safe refusals, pilot-evidence requests, expert-scope limitations, and imperfect LoRA paraphrases receive proportional partial credit. Concrete wrong definitions, fabricated live events, direct contradictions, privacy violations, leakage, and genuine decoding collapse remain scored.

The calibration packet contains 639 cases covering all 31 reporting categories, all 150 official references, all 63 scored labels, all 150 real collapsed outputs, 150 human-adjudicated coherent outputs, and 41 concentrated regression cases. Run `python .\scripts\common\verify_quality_evaluation_v4_5.py` before official rescoring.

The governing files are `EVALUATION_V4_5_FRESH_MESSAGE_CONTRACTS.md`, `configs/quality_issue_taxonomy_v4_5.json`, `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`, and `BLUEPRINT_AMENDMENT_RELATIVE_OPPORTUNITY_v1.0.md`.

## Start here

- `Phases.md` — phase boundaries and handoffs.
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md` — every-layer/progressive deletion workflow.
- `scripts/phase_2c_targeted_recovery/EXECUTION_MANUAL.md` — targeted recovery, review, resume, and handoff.
- `POST_PHASE_3_EXECUTION_GUIDE.md` — Phase 4–11 execution order, including single and hybrid Phase 9 handoffs.
- `STUDENT_DISTILLATION_PIPELINE.md` — Phase 9 distillation foundation.
- `PHASE_10_11_ARCHITECTURE_REVIEW.md` — codebase review, corrected defects, boundaries, and Phase 10/11 architecture.
- `PHASE_10_11_WORKFLOW.md` — runnable Phase 10/11 commands and human handoffs.
- `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md` — runnable Phase 9 commands and gates.
- `scripts/phase_10_student_architecture/EXECUTION_MANUAL.md` — structural ladder, recovery, challenger, and selection.
- `scripts/phase_11_production_rag/EXECUTION_MANUAL.md` — production snapshot, quantization, runtime, release, and packaging.
- Each Phase 4–8 folder contains `README.md`, `EXECUTION_MANUAL.md`, and `FILE_RESPONSIBILITIES.md`.
- `verify_post_phase_3.ps1` — compiles source, runs tests, and checks runner references.

## Canonical handoffs

Phase 2B writes `artifacts/phase_2_progressive_reduction/model_catalog.json`. Phase 2C may start only from an approved reduced profile and writes `artifacts/phase_2c_targeted_recovery/model_catalog.json`. Phase 3 may consume either an approved Phase 2B profile or an approved Phase 2C model/adapter pair; the selectors are mutually exclusive. Phase 4 writes `artifacts/references/phase_4_reference.json`. Phase 4B writes `artifacts/references/selected_inference_reference.json`. Phases 5–8 consume the latter automatically. Phase 9 starts only from an explicitly accepted teacher checkpoint and writes isolated student artifacts under `artifacts/phase_9_student_distillation/`; it never overwrites the teacher reference. Human-approved Phase 9 selection writes `artifacts/references/phase_9_student_reference.json`; human-approved Phase 10 selection writes `artifacts/references/phase_10_student_reference.json`; Phase 11 packages only an explicitly approved registry candidate and preserves the Phase 10 rollback.


## Response-control architecture

The repository now separates learned brevity from the deployable hard boundary:

- Phase 1 base/adapted evaluations use `eos_only` so weight-learning effects remain measurable.
- The smoke test and Phases 5–8 use `one_sentence`, which stops after model-native EOS or the first validated sentence boundary and retains `max_new_tokens` as a fail-safe.
- The controller protects decimals, versions, abbreviations, URLs/domains, Markdown code, ordered-list markers, internal ellipses, and unmatched quotes/brackets.
- Phase 8 records the strict response-control mode and its observed violation rate for human release review; no absolute numerical violation floor independently promotes or rejects the model.

See `RESPONSE_CONTROL_AND_EARLY_EXIT_AUDIT.md` and `scripts/phase_7_sparse_runtime/RESPONSE_CONTROL_CONTRACT.md`. This controller avoids decode work; it does not remove model weights or turn selective training into inference-time layer skipping.

## Quality evaluation v4.5

All generation-based phases and the independent Phase 2B playground use `scripts/common/quality_evaluation.py`, `scripts/common/observable_quality.py`, and `scripts/common/opportunity_evaluation.py`.

The evaluator separates universal decoding/mechanical integrity, explicit case-level contract fulfillment, category-neutral prompt-function alignment, and proportional candidate comparison. Category statistics and review-only diagnostics never reduce the score or affect promotion. Existing generations can be rescored without model inference using `scripts/common/rescore_quality_artifacts.py`; complete Phase 2B trees can be migrated with `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_5.ps1`. The official contracted probes are `probe_50_v4_5.jsonl` and `probe_150_thorough_v4_5.jsonl`.

Verification commands:

```powershell
python .\scripts\common\verify_quality_evaluation_v4_5.py
pytest -q
python .\scripts\common\audit_relative_opportunity_policy.py
```

See `EVALUATION_V4_4_MIGRATION.md` for review-only rescoring, official in-place migration, corrected-frontier replay, and clean-run commands.

## Production principle

A technique ships only when it creates measured value. The best release may be the full dense teacher, a shallower recovered teacher, a dense quantized teacher with context engineering, or an accepted RAG-specialized student while custom sparsity remains disabled. Negative selective-tuning, deletion, soup, and sparse-runtime results are valid and retained.

Generated checkpoints and benchmark artifacts are not included in the source ZIP; they must be created on the target machine so results reflect its actual model, CUDA stack, and hardware.
