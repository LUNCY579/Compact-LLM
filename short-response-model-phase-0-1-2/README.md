# v2.1 — Procedural Transfer Recovery

v2.1 is a focused recovery release for the Phase 9 student that is strong under Oracle evidence but collapses on Real-RAG procedure application. The v2.0 stop gate records a 52.0% weight reduction and only ~0.6% generic relative quality loss for the selected SFT student, yet Real-RAG procedure correctness is 0/150. A model-free postmortem found that all 150 answerable historical procedure cases retrieved the authored support and an equivalent trusted procedure, while all 150 legacy top-3 prompts also contained near-identical same-topic sibling records. v2.1 therefore diagnoses and repairs **evidence binding under retrieval collisions** before doing any further depth/width/sparsity optimization.

The release adds a query-aware retrieval selector, a 6,120-row procedural recovery corpus spanning 14 deterministic procedure families, gold-consistency audits, Oracle→Real failure classification, a same-weight legacy/query-aware A/B, and one focused two-stage rank-16 LoRA recovery path rather than another broad LR/rank sweep. It also keeps Phase 11 calibration/runtime/service on the same retrieval-selection policy. See `PROCEDURAL_RECOVERY_V2_1.md`, `CHANGED_FILES_v2.1.md`, and `RELEASE_VERIFICATION_v2.1.md`.

**Run recovery before v2.0 structural optimization** (selected Phase 9 student = warm 344M + `lr1e4_s1200`):

```powershell
cd .\short-response-model-phase-0-1-2   # if starting from the repo root
$StudentBase = '.\artifacts\phase_9_grounded_recovery_v1_7_5_qwen\warm_344m_initialization'
$StudentAdapter = '.\artifacts\phase_9_grounded_recovery_v1_7_5_qwen\warm_sft_sweep\lr1e4_s1200\training\adapter'
$RecoveryRoot = '.\artifacts\phase_9_procedural_recovery_v21_lr1e4_s1200'

.\scripts\phase_9_student_distillation\run_phase_9_procedural_recovery_v21.ps1 `
  -StudentModel $StudentBase `
  -StudentAdapter $StudentAdapter `
  -OutputRoot $RecoveryRoot `
  -RequireCuda
```

DEV diagnosis and two-stage recovery run first. After reviewing the DEV comparison, re-run the same command with `-ReuseExisting -RunSealedTest`. Full detail: `PROCEDURAL_RECOVERY_V2_1.md`.

# v2.0 — Automated Student Optimization Program

v2.0 turns the accepted Phase 9/9B student into a first-class post-distillation optimization program. It preserves official Phases 9/9B/10/11, but adds a durable cross-phase orchestrator with an empirical every-layer deletion scan, importance-guided Phase 10 structural reduction/recovery, DEV-only context/KV budget selection, contextual-sparsity simulation and measured runtime, DEV-only quantization backend selection, human checkpoints, final Phase 11 runtime evaluation, and reproducible packaging.

The master runner is `scripts/student_optimization_v2/run_student_optimization_v2.ps1`. It writes `artifacts/student_optimization_v2/WORKFLOW_STATE.json` and incrementally rewrites `reports/STUDENT_OPTIMIZATION_V2_MASTER_REPORT.md`. A run is **not** labelled complete because a process exited; `COMPLETE` is reserved for the state in which the Phase 11 human gate is PASS and the approved release package exists. See `STUDENT_OPTIMIZATION_V2_WORKFLOW.md`, `CHANGED_FILES_v2.0.md`, and `RELEASE_VERIFICATION_v2.0.md`.

## v1.7.7 Phase 9 Stage-B/SFT selection gate hotfix

v1.7.7 fixes a policy mismatch in the Phase 9 grounded-recovery ladder. Stage B is an intentionally untrained warm-student baseline, so prompt/decode-integrity failures recorded there remain visible diagnostics but no longer invalidate every trained Stage D SFT candidate. Initialization behavior-coverage failures remain blocking because missing denominators make the numeric comparison unreliable. Every trained candidate is still required to pass its **own** integrity, behavior coverage, non-degeneracy, behavior-success, and positive Oracle-capability-improvement checks.

The grounded-recovery PowerShell runner now treats Stage D and final DEV selections as cheap derived artifacts: when `-ReuseExisting` is supplied, it recomputes them from the already-existing evaluation summaries under the current selector policy instead of silently reusing a stale `selection.json`. This allows a v1.7.5/v1.7.6 run that already completed SFT training/evaluation to be repaired and continued without retraining. See `PHASE_9_SFT_SELECTION_GATE_HOTFIX_v1.7.7.md`.

## v1.7.5 large corrective RAG and memory-reconstruction curricula

v1.7.5 expands the evidence-specialization program specifically around the observed weak behaviors: supported factual answering and trusted-procedure application. Phase 9 now defaults to `rag_reasoning_v3_large` with 5,900 canonical rows, 6,000 factual documents, 200 trusted procedures, an answer-heavy 55% supported / 25% procedure corrective training profile, and a pre-evaluation FTS5 audit proving every supported DEV row can retrieve its authored support document in the top 3. Phase 9B now defaults to `reconstruction_corpus_v2_large` with 3,600 source-state reconstruction rows, 1,500 memories, all five reconstruction decisions, 600 counterfactual pairs, 10,800 curated preferences, and 14,400 evaluator candidates.

Both corpora use authored topic-family train/dev/sealed-test splits and deterministic `auto_verified` gold rather than falsely claiming human review. Phase 10 and Phase 11 consume the same v1.7.5 assets. See `PHASE_9_9B_LARGE_CORRECTIVE_DATASETS_v1.7.5.md`.

## v1.7.4 dedicated Phase 9B memory-reconstruction corpus

Phase 9B now has a first-class 600-row source-state reconstruction curriculum rather than relying only on reconstruction examples derived from the Phase 9 evidence corpus. The integrated asset lives at `data/phase_9b_memory_reconstruction/reconstruction_corpus_v1` and contains 300 external memories, all five reconstruction decisions, 120 counterfactual state pairs, 1,800 curated preference pairs, and 2,400 evaluator candidates under a 440 train / 80 dev / 80 sealed-test topic-family split.

v1.7.4 adds a bounded validity-condition DSL (`all`, `any`, named safe predicates, `item`, legacy mappings), audits counterfactual edit metadata, keeps known-wrong preference/evaluator alternatives out of SFT, selects reconstruction strategy entirely on DEV, freezes exactly one strategy before TEST, and evaluates safe conflict handling separately from explicit conflict wording. Phase 10 validates the reconstruction corpus when consuming an approved Phase 9B checkpoint. Phase 11 exposes the same `off` / `prompted` / `trained` reconstruction modes in both the service and measured production benchmark and records the extra two-pass cost. See `PHASE_9B_MEMORY_RECONSTRUCTION_CORPUS_INTEGRATION_v1.7.4.md`.

# Short-Response Model — runnable mature phase pipeline

## v1.7.3 Phase 9–11 RAG reasoning and memory-reconstruction integration

The 530-row `rag_reasoning_v2` corpus is now a first-class Phase 9–11 research asset rather than an external optional dataset. Its authored topic-family split is preserved exactly at 392 train / 67 dev / 71 sealed test rows. Phase 9 uses the dev split for teacher/warm preflight, SFT selection, Real-RAG diagnosis, and optional KD selection; the sealed test is touched only after one final candidate is frozen. Phase 10 likewise performs structural/recovery selection on dev and evaluates exactly the reference plus one frozen final candidate on test. Phase 11 calibrates answerability on dev and benchmarks the final runtime on test.

The integration normalizes procedure-answer semantics, accepts safe canonical abstention for genuine conflicts, distinguishes explicit conflict acknowledgment as a diagnostic, filters superseded and untrusted evidence before conflict resolution, and exposes trust/version metadata to the grounded prompt. Two procedure rows whose original conflict label contradicted the supplied comparison procedure are repaired to their objectively computable answers. Phase 9B now derives reconstruction candidates directly from the same corpus and treats stale/current or malicious retrieval as applicability adaptation rather than false equal-authority conflict. Optional DPO preference refinement and evaluator-calibration assets are included but remain separate from ordinary SFT. See `PHASE_9_11_RAG_REASONING_INTEGRATION_v1.7.3.md`.

## v1.7.2 Phase 9 grounded-recovery capability ladder

Phase 9 now proves one warm 344M depth-copy student before permitting another broad architecture comparison. The official grounded-recovery runner separates teacher Oracle behavior, the untrained warm baseline, a 12–24-row deliberate overfit wiring test, a three-run warm-SFT sweep, full Oracle evidence use, Real-RAG retrieval, and only then progressive KD. A degenerate all-zero frontier is reported as `NO_VALID_QUALITY_FRONTIER`; it is never ranked as a successful compression result.

The training view is deterministically rebalanced to approximately 45% supported extraction, 25% missing-evidence abstention, 10% conflict, 10% injection resistance, 5% procedures, and 5% social behavior. Completion loss now explicitly emphasizes the first answer token, required evidence anchors, the approved abstention sequence, and EOS/end-of-turn. Initialization→SFT, SFT→KD, initialization→KD, and candidate→teacher comparisons are reported separately. Hybrid 100M/130M/170M controls remain locked until the grounded-recovery ladder proves both overfit wiring and full Oracle improvement. See `PHASE_9_GROUNDED_RECOVERY_v1.7.2.md`.

## v1.7.1 Phase 9 RAG integrity and diagnosis hotfix

Phase 9 now fails closed when evidence, the grounding contract, or the final question would be silently truncated. Evaluation schema 1.2 records prompted support IDs, prompt hashes and token counts, decode-integrity diagnostics, conflict handling, and filtered/passthrough injection resistance. The opportunity gate now charges an explicit Phase 9 task-contract deficiency with a proportional core-metric bottleneck, so zero supported correctness and 100% unsupported confident answering produce a full functional cost instead of appearing lossless merely because the teacher or generic evaluator is also weak. Near-zero comparisons are bounded and finite.

The official runner audits the teacher before student training, and `run_phase_9_diagnostics.ps1` can diagnose existing artifact trees without retraining. Old evaluations must be regenerated; old student artifacts remain useful research controls but are not silently reused for v1.7.1 training. See `PHASE_9_RAG_INTEGRITY_HOTFIX_v1.7.1.md` and the Phase 9 execution manual.

## v1.7.0 Phase 9B contextual memory reconstruction

Phase 9B is now inserted between Phase 9 student distillation and Phase 10 architecture reduction. It upgrades retrieved evidence into source-state-aware memory records, generates human-reviewable ACCEPT/ADAPT/REJECT/CONFLICT/INSUFFICIENT targets, builds leakage-safe reconstruction/answer/direct-distillation curricula, trains a low-memory LoRA reconstruction adapter, and compares memory-off, naive raw replay, prompted reconstruction, trained two-pass reconstruction, and a one-pass distilled candidate.

The production RAG process can enable the same inference-time protocol with `--memory-reconstruction-mode prompted` or `trained`. Invalid reconstruction JSON falls back to insufficient evidence and abstention. Phase 10 automatically prefers the human-approved `phase_9b_student_reference.json` when it exists, while retaining the accepted Phase 9 student as rollback. See `MEMORY_RECONSTRUCTION_PIPELINE.md` and `scripts/phase_9b_memory_reconstruction/EXECUTION_MANUAL.md`.

## v1.6.1 Phase 9 Behavior Curriculum v2

Phase 9 now treats the earlier six-row hybrid result as an execution smoke test rather than learning evidence. The refined pipeline requires class-covered, leakage-safe real data; supplements it with a 336-row fictional authored behavior foundation; expands only human-approved targets through controlled curriculum transforms; trains curriculum SFT before confidence-filtered progressive KD; emphasizes EOS/end-of-turn targets; and keeps direct KD from initialization as an optional ablation.

Evaluation now separates social conversation from unsupported factual questions, reports real denominators, returns `null` for inapplicable metrics, uses explicit response contracts plus grounding for functional correctness, blocks smoke/zero-success candidates from production recommendation, and uses bounded absolute deltas for SFT/KD comparisons. Filtered and passthrough adversarial conditions distinguish sanitizer strength from model-level injection resistance. See `scripts/phase_9_student_distillation/README.md` and `CHANGED_FILES_PHASE_9_BEHAVIOR_CURRICULUM_V2.md`.

## v1.6.0 appended Phase 10 architecture search and Phase 11 production RAG release

Phase 10 now consumes the exact human-approved Phase 9 student, profiles residual/MLP/attention structure, creates physically smaller dense candidates across intermediate, small, and 80M–120M deployment target bands, performs fresh reload checks, recovery-distills a bounded set under the original teacher, optionally trains a compact pretrained challenger, and selects from a shared relative-opportunity frontier.

Phase 11 consumes the exact human-approved Phase 10 student and implements an immutable knowledge snapshot, SQLite FTS5 retrieval, development-calibrated answerability, deterministic abstention/conflict handling, evidence-token budgeting, conservative post-generation grounding, dense/NF4/INT8/GGUF deployment candidates, total-process memory and end-to-end benchmarks, a persistent local JSONL runtime, human release gate, rollback evidence, and release packaging. See `PHASE_10_11_ARCHITECTURE_REVIEW.md` and `PHASE_10_11_WORKFLOW.md`.

## v1.5.0 appended Phase 9 student distillation pipeline

The official Phase 0–8 program remains the teacher-development pipeline. Phase 9 now consumes an exact human-approved teacher and builds smaller RAG-specialized students through warm-started student construction, reviewed teacher-data generation, document/topic-disjoint pilot data, calibrated evidence gating, curriculum SFT followed by confidence-filtered completion-only logit distillation, optional on-policy recovery, five-condition RAG evaluation, and proportional human-gated selection.

The first retriever is SQLite FTS5 so generator savings are not silently offset by a neural retrieval stack. Direct vocabulary KL requires identical tokenizer fingerprints; cross-tokenizer challengers start with sequence-level SFT. Randomly initialized narrow-Qwen models are explicitly marked research controls. See `STUDENT_DISTILLATION_PIPELINE.md` and `scripts/phase_9_student_distillation/EXECUTION_MANUAL.md`.

The latest Phase 9 runner also supports `-StudentSuite Hybrid`, which evaluates one warm 344M depth-copy student and 100M/130M/170M narrow same-tokenizer research controls in a single assessment. Candidate keys are profile-qualified (`sft_warm_344m`, `progressive_kd_warm_344m`, and so on), and the Phase 10 handoff accepts the exact key recorded in `reports/PHASE_9_ASSESSMENT.json`. Narrow random-initialized profiles remain research controls unless a separate human language-readiness review explicitly authorizes promotion.

## v1.4.7 Phase 2C reliability and correctness hotfix

v1.4.7 fixes the repeated Windows file-lock failures and closes a Phase 2C evaluation-validity gap. Phase 2C deterministic reasoning rows now carry explicit prompt-grounded correctness contracts, so fluent but wrong arithmetic, ordering, comparison, implication, contradiction, set-membership, and temporal answers cannot receive high scores merely for being short and well formed.

Existing v1.4.6 Phase 2C work is preserved through a narrowly authorized migration: completed adapters and raw generated replies are reused, replies are rescored under the corrected contracts, teacher-forced NLL rows are reused, partial evaluations continue by stable ID, and affected output comparisons are rebuilt. Run the original Phase 2C wrapper with identical settings plus `-Resume`; do not delete the current artifact root or run child evaluators manually.

Reliability hardening also includes bounded atomic-replace retries for transient Windows locks, best-effort progress sidecars, resumable generation and teacher forcing, normalized manifest paths, strict stable-ID coverage, corrupt-sentinel quarantine, safer staged adapter publication, default suppression of intermediate checkpoints, free-disk checks, stronger Phase 2C artifact/gate audits, and `None`-safe report rendering. See `PHASE_2C_RELIABILITY_AND_CORRECTNESS_HOTFIX_v1.4.7.md`.

This repository implements the ordered Phase 0–8 teacher program, two structural teacher branches, targeted recovery, and an appended Phase 9–11 student distillation, architecture-reduction, and production-RAG pipeline:

- **Phase 2B progressive reduction** tests every current layer, recovery-tunes promising deletions, and preserves full-quality/balanced/minimum-memory model profiles before DPO.
- **Phase 2C targeted recovery** ranks surviving Phase 2B layers and trains selected-layer adapters whose proportional training-resource savings exceed their proportional quality opportunity loss.
- **Phase 4B post-alignment structural reduction** revalidates or further refines the accepted Phase 4 architecture before context/sparsity calibration.
- **Phase 9 student distillation** proves evidence-conditioned SFT/KD transfer into a warm-started student.
- **Phase 9B memory reconstruction** teaches the student to judge whether retrieved memory still applies before answering.
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
- `MEMORY_RECONSTRUCTION_PIPELINE.md` — Phase 9B source-state reconstruction design, commands, and gates.
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

## Student Optimization v2.0 — automated post-distillation efficiency ladder

After a Phase 9/9B student has an explicit human PASS and is materialized, the new cross-phase runner can carry it through the remaining efficiency program with durable resume/checkpoint semantics:

```powershell
.\scripts\student_optimization_v2\run_student_optimization_v2.ps1 `
  -TeacherModel C:\path\to\authoritative_teacher
```

It adds an empirical every-layer student deletion scan, importance-guided Phase 10 depth candidates, disjoint-DEV context/KV budget selection, student contextual-sparsity simulation/runtime measurement, and DEV-only quantization selection before the frozen Phase 11 sealed runtime experiment. It automatically stops at Phase 10 and Phase 11 human gates and resumes from those checkpoints when rerun. The workflow is only labelled `COMPLETE` after an approved Phase 11 release package exists.

See `STUDENT_OPTIMIZATION_V2_WORKFLOW.md` and `scripts/student_optimization_v2/README.md`.
