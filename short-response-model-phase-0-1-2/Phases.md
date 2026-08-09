## v2.1 Phase 9 procedural transfer-recovery checkpoint

Before Phase 9B/10/11 structural and runtime optimization, a Phase 9 student with a severe Oracle→Real procedural gap must run `run_phase_9_procedural_recovery_v21.ps1`. v2.1 first A/B tests the same weights under legacy versus query-aware retrieval, then performs focused collision-binding SFT and preservation consolidation. Retrieval-policy gain and learned training gain are reported separately. Other students are replication controls only after the primary student establishes the mechanism. The old v1.7.5 test is now postmortem/regression evidence; the new v2.1 test remains unused for selection.

## v1.7.4 Phase 9B dedicated memory-reconstruction corpus

Phase 9 remains the prerequisite evidence-use/abstention curriculum. Phase 9B now has a dedicated source-state reconstruction corpus with explicit current state and validity conditions. It trains and evaluates ACCEPT / ADAPT / REJECT / CONFLICT / INSUFFICIENT behavior, counterfactual state sensitivity, negative-transfer reduction, and optional reconstruction preference alignment. DEV freezes one strategy before sealed TEST. Phase 10 may consume the human-approved direct-compatible Phase 9B checkpoint; Phase 11 may execute and measure the approved reconstruction behavior at inference.

# Project scripts by phase

| Phase | Folder | Main responsibility | Downstream artifact |
|---|---|---|---|
| 0 | `phase_0_data` | Data contract, validation, leakage control, human audit. | Approved data manifests. |
| 1 | `phase_1_dense_sft` | Dense concise-behavior baseline. | Dense SFT adapter/checkpoint. |
| 2A | `phase_2_layer_selection` | Static selected-layer and module-group tuning study. | Preserved positive or negative training-efficiency result. |
| 2B | `phase_2_layer_selection` | Every-layer deletion, recovery SFT, progressive depth reduction, profile catalog. | Full-quality/balanced/min-memory branches. |
| 2C | `phase_2c_targeted_recovery` | Rank surviving layers and train low-resource selected-layer recovery controls. | Maximum-quality/balanced-training/minimum-training-memory adapter profiles. |
| 3 | `phase_3_preference_alignment` | DPO for concise-correct preference on one selected profile. | Profile-specific aligned adapter. |
| 4 | `phase_4_model_soups` | Compatible delta merging or best-individual decision for that profile. | Accepted Phase 4 dense/full checkpoint. |
| 4B | `phase_4b_structural_reduction` | Post-alignment structural revalidation/refinement and exact checkpoint selection. | `selected_inference_reference.json`. |
| 5 | `phase_5_context_system` | Retrieval, budgets, prefix/KV reuse. | Approved context system. |
| 6 | `phase_6_sparse_simulation` | Activation traces, oracle masks, quality frontier. | Router dataset and frontier. |
| 7 | `phase_7_sparse_runtime` | Strict response control, routers, dense fallback, measured runtime. | Response-control verification, runtime comparison, and optional router checkpoint. |
| 8 | `phase_8_release_candidate` | Strict sealed evaluation, full release audit, and human decision. | Release package with response-control policy. |
| 9 | `phase_9_student_distillation` | Prove Oracle evidence use, diagnose Oracle→Real retrieval transfer, and recover exact evidence/procedure binding before progressive KD or architecture reduction. v2.1 adds a same-weight retrieval A/B and collision-aware procedural repair when a zero-procedure blocker appears. | Grounded-recovery ladder, v2.1 procedural-recovery evidence when needed, full Phase 9 assessment, and human gate. |
| 9B | `phase_9b_memory_reconstruction` | Add source-state-aware applicability reconstruction between retrieval and answer generation; train and compare two-pass and one-pass behavior. | Accepted reconstruction-aware student/runtime policy plus Phase 9 rollback. |
| 10 | `phase_10_student_architecture` | Physically reduce the approved student, recover promising candidates under the original teacher, and compare compact challengers. | Accepted Phase 10 checkpoint/reference plus rollback. |
| 11 | `phase_11_production_rag` | Build the immutable RAG runtime, calibrate answerability, quantize/benchmark, and package the approved deployment. | Measured production release package and manifest. |

## v1.7.3 evidence-corpus contract

Phases 9–11 share `data/phase_9_student_distillation/rag_reasoning_v2` as the canonical evidence-reasoning corpus. Its authored topic-family split is preserved: 392 training rows, 67 development rows, and 71 sealed test rows. Training/hyperparameter/retrieval calibration may use train/dev only. The sealed test cannot select SFT, KD, reconstruction, pruning, recovery, or answerability settings.

- Phase 9 trains evidence use and abstention from the train split, selects on dev, and evaluates one frozen candidate on test.
- Phase 9B derives source-state-aware ACCEPT/ADAPT/REJECT/CONFLICT/INSUFFICIENT examples from the same train/dev structure; untrusted and superseded memories are filtered before equal-authority conflict detection.
- Phase 10 performs structure/recovery/challenger selection on dev, then evaluates one frozen final architecture on test.
- Phase 11 calibrates the runtime answerability gate on dev and uses test only for the final production-runtime comparison.
- Preference alternatives live under `rag_reasoning_v2/preferences`; evaluator calibration/heldout examples live under `rag_reasoning_v2/evaluator`. Neither is silently mixed into ordinary SFT.

## Responsibility boundaries

- Phase 2A asks **where to train** and may remain a negative result.
- Phase 2B asks **which complete layers can be removed and recovered before alignment**.
- Phase 2C asks **which surviving layers should receive the smallest useful recovery adapter**.
- Phase 4B asks **whether the aligned checkpoint should keep, refine, or reject structural reduction**.
- Phase 6 asks **what can be temporarily masked** while preserving dense computation for scientific validation.
- Phase 7 asks **what can be dynamically skipped in the measured runtime**.
- Phase 9 asks **whether one inherited warm depth-copy student can first prove Oracle evidence use and training wiring; if Oracle succeeds but Real RAG fails, v2.1 must separate retrieval/prompt collisions from learned evidence-binding weakness and recover the blocker before progressive KD, narrow controls, or downstream structural optimization**.
- Phase 9B asks **whether judging, adapting, or rejecting retrieved memory reduces negative transfer enough to justify its full runtime cost**.
- Phase 10 asks **which physically smaller architecture or compact challenger gives the best measured quality per deployed byte**.
- Phase 11 asks **whether the selected student and RAG stack produce a measured, grounded, rollback-safe local release**.

A reduced Phase 2B or Phase 2C profile is never silently substituted. Phase 3 resolves exactly one named, human-approved branch, and profile-specific artifacts are written side by side. Phase 5 still requires an approved Phase 4B selected-reference manifest. Phase 9 requires an explicitly approved teacher and keeps all student artifacts isolated from the teacher catalog. Phase 10 prefers a materialized Phase 9B reference when one exists and otherwise uses the Phase 9 reference; Phase 11 requires a materialized Phase 10 student reference.

- Learned short behavior and hard response control are measured separately: Phase 1 uses `eos_only`, while production-oriented Phases 5–11 default to `one_sentence` where generation is model-controlled; deterministic external abstention may bypass generation.
- The strict boundary is an output/runtime mechanism. It does not authorize whole-layer early exit or reduce checkpoint weight memory.


## Shared evaluation and promotion rule

Every phase uses Quality Evaluator v4.5 and the same relative-opportunity engine. Each phase declares one primary proportional benefit—training state, loaded weight memory, context/KV tokens, active compute, measured latency/VRAM, or net quality improvement. A technically comparable candidate advances only when that benefit is strictly greater than its combined proportional quality opportunity loss. No phase may add a private absolute score, pass-rate, completion, category, NLL, recall, speedup, or issue-count floor. Official response contracts are freshly authored from visible system/user messages; category, reference wording, and inherited contracts cannot create requirements. Informal understandable replies and natural follow-up questions remain valid unless the visible prompt creates a concrete requirement. Human review remains mandatory at stop gates.

## v2.0 cross-phase student optimization orchestrator

The official numeric phase boundaries remain 9 / 9B / 10 / 11. v2.0 adds a **meta-workflow**, not a competing set of official phases:

| v2 stage | Responsibility | Official implementation boundary |
|---|---|---|
| S0 | Resolve the exact accepted student lineage and authoritative teacher. | Phase 9/9B handoff |
| S1 | Physically delete every student layer once and rank DEV sensitivity. | New student optimization evidence |
| S2 | Importance-guided depth + width/GQA structural search and recovery KD. | Phase 10 |
| S3 | Human structural checkpoint and materialization. | Phase 10 gate/finalizer |
| S4 | DEV-only retrieval/evidence token-budget optimization. | New pre-Phase-11 runtime selection |
| S5 | DEV-only contextual sparsity simulation and measured router runtime. | Reuses Phase 6/7 primitives without overwriting teacher gates |
| S6 | DEV-only dense/NF4/INT8/optional-GGUF runtime selection. | New pre-Phase-11 deployment selection |
| S7 | Frozen context/backend production RAG comparison on sealed TEST. | Phase 11 Full |
| S8 | Human release checkpoint and package. | Phase 11 Package |

The meta-runner is `scripts/student_optimization_v2/run_student_optimization_v2.ps1`. It maintains `artifacts/student_optimization_v2/WORKFLOW_STATE.json` plus `reports/STUDENT_OPTIMIZATION_V2_MASTER_REPORT.md`, resumes automatically after human checkpoints, and cannot mark the workflow complete before the approved release package exists.
