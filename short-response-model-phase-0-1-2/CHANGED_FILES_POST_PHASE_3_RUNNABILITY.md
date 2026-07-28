# Changed files — post-Phase-3 runnability repair

This is the exact comparison against `Output1_phase_4B_integrated_mature.zip` supplied before this repair. Generated artifacts/checkpoints are intentionally excluded.

- Added: **26** files
- Modified: **50** files
- Removed: **0** files

## Root installation/documentation

| Status | File | Responsibility/change |
|---|---|---|
| ADDED | `CHANGED_FILES_POST_PHASE_3_RUNNABILITY.md` | Exact added/modified/removed file inventory relative to the previously delivered ZIP. |
| MODIFIED | `COMMENT_VERIFICATION.md` | Documents source-comment and responsibility-document checks. |
| MODIFIED | `IMPLEMENTATION_CHANGELOG_PHASE_4B_INTEGRATION.md` | Records the post-Phase-3 runner and Phase 4B implementation changes. |
| MODIFIED | `PHASE_4B_PACKAGE_VERIFICATION.md` | Records Phase 4B/package checks and limits. |
| ADDED | `POST_PHASE_3_EXECUTION_GUIDE.md` | Copy-paste execution order for Phases 4, 4B, 5, 6, 7, and 8. |
| ADDED | `POST_PHASE_3_RUNNABILITY_VERIFICATION.md` | Records compilation, tests, static runner checks, synthetic smoke checks, and untested boundaries. |
| MODIFIED | `README.md` | Repository entry point describing real runners, handoffs, and dense fallback policy. |
| ADDED | `install_post_phase_3.ps1` | Installs the post-Phase-3 Python dependencies and verifies imports/CUDA visibility. |
| MODIFIED | `requirements-model-optimization.txt` | Compatibility alias that installs the canonical requirements file. |
| ADDED | `requirements.txt` | Complete runtime/training dependency list for the runnable post-Phase-3 pipeline. |
| ADDED | `verify_post_phase_3.ps1` | Runs source compilation, regression tests, runner-reference checks, and optional dependency import verification. |

## Shared runtime/evaluation support

| Status | File | Responsibility/change |
|---|---|---|
| MODIFIED | `scripts/common/experiment_utils.py` | Shared robust JSON/JSONL I/O, deterministic fingerprints, environment capture, and checked subprocess execution. |
| MODIFIED | `scripts/common/sparse_mlp.py` | Shared oracle masking and actual gathered sparse-decode reference with truthful prefill/fallback telemetry. |
| MODIFIED | `scripts/phase_1_dense_sft/evaluate_model.py` | Common model evaluator used by Phases 4–8; prevents trailing-assistant target leakage and supports local full checkpoints. |

## Phase 4 — model soups

| Status | File | Responsibility/change |
|---|---|---|
| MODIFIED | `scripts/phase_4_model_soups/EXECUTION_MANUAL.md` | Provides step-by-step commands, recovery guidance, and review requirements for phase_4_model_soups. |
| ADDED | `scripts/phase_4_model_soups/FILE_RESPONSIBILITIES.md` | Lists the responsibility and non-responsibility of every executable/document in phase_4_model_soups. |
| MODIFIED | `scripts/phase_4_model_soups/README.md` | Explains the purpose, inputs, outputs, gates, and expected workflow for phase_4_model_soups. |
| ADDED | `scripts/phase_4_model_soups/build_greedy_soup.py` | Ranks ingredients, materializes/evaluates each greedy trial, applies quality floors, and writes Phase 4 reports/manifests. |
| MODIFIED | `scripts/phase_4_model_soups/check_human_gate.py` | Compatibility gate wrapper/documentation for phase_4_model_soups; active runners use the shared gate checker. |
| MODIFIED | `scripts/phase_4_model_soups/greedy_soup_plan.py` | Retained legacy planning helper; explicitly not the production Phase 4 executor. |
| MODIFIED | `scripts/phase_4_model_soups/merge_delta_soup.py` | Reconstructs adapter deltas against one base, averages them in FP32, and saves a standalone Transformers checkpoint. |
| ADDED | `scripts/phase_4_model_soups/resolve_phase_4_model.py` | Resolves the exact local full checkpoint accepted by Phase 4. |
| MODIFIED | `scripts/phase_4_model_soups/run_phase_4.ps1` | Real Phase 4 orchestrator: checks Phase 3, trains/evaluates ingredients, builds greedy soups, and writes a full checkpoint. |
| MODIFIED | `scripts/phase_4_model_soups/validate_ingredients.py` | Checks actual PEFT configuration plus saved tensor keys/shapes before soup merging. |

## Phase 4B — structural reduction

| Status | File | Responsibility/change |
|---|---|---|
| MODIFIED | `scripts/phase_4b_structural_reduction/EXECUTION_MANUAL.md` | Provides step-by-step commands, recovery guidance, and review requirements for phase_4b_structural_reduction. |
| MODIFIED | `scripts/phase_4b_structural_reduction/FILE_RESPONSIBILITIES.md` | Lists the responsibility and non-responsibility of every executable/document in phase_4b_structural_reduction. |
| MODIFIED | `scripts/phase_4b_structural_reduction/README.md` | Explains the purpose, inputs, outputs, gates, and expected workflow for phase_4b_structural_reduction. |
| MODIFIED | `scripts/phase_4b_structural_reduction/apply_structured_pruning.py` | Physically slices MLP/GQA tensors, updates configuration, smoke-tests, saves, and reload-verifies a smaller checkpoint. |
| ADDED | `scripts/phase_4b_structural_reduction/build_long_form_calibration.py` | Creates matched short/long calibration pairs from non-sealed prompts and the accepted Phase 4 model. |
| MODIFIED | `scripts/phase_4b_structural_reduction/build_pruning_plan.py` | Converts attribution profiles into deterministic structured MLP/head pruning plans and random controls. |
| ADDED | `scripts/phase_4b_structural_reduction/create_phase_4b_gate.py` | Builds the pending Phase 4B human stop-gate report from the measured recommendation. |
| MODIFIED | `scripts/phase_4b_structural_reduction/evaluate_candidates.py` | Runs the same behavioral evaluator and size measurement on dense and every Phase 4B candidate. |
| MODIFIED | `scripts/phase_4b_structural_reduction/finalize_reference_selection.py` | Writes the canonical selected-inference-reference manifest after explicit human approval. |
| MODIFIED | `scripts/phase_4b_structural_reduction/measure_checkpoint.py` | Measures parameters and checkpoint bytes for a candidate. |
| MODIFIED | `scripts/phase_4b_structural_reduction/profile_long_form_components.py` | Profiles early-versus-late MLP-channel and attention-head contribution proxies. |
| MODIFIED | `scripts/phase_4b_structural_reduction/recommend_reference_candidate.py` | Applies practical quality floors and recommends a reduced candidate or dense fallback. |
| MODIFIED | `scripts/phase_4b_structural_reduction/reduce_depth.py` | Builds physically shallower checkpoints with suffix and equal-count uniform controls, then smoke/reload verifies them. |
| MODIFIED | `scripts/phase_4b_structural_reduction/run_phase_4b.ps1` | Real Phase 4B orchestrator: builds calibration, profiles components, creates pruning/depth candidates, evaluates and ranks them. |

## Phase 5 — context system

| Status | File | Responsibility/change |
|---|---|---|
| MODIFIED | `scripts/phase_5_context_system/EXECUTION_MANUAL.md` | Provides step-by-step commands, recovery guidance, and review requirements for phase_5_context_system. |
| ADDED | `scripts/phase_5_context_system/FILE_RESPONSIBILITIES.md` | Lists the responsibility and non-responsibility of every executable/document in phase_5_context_system. |
| MODIFIED | `scripts/phase_5_context_system/README.md` | Explains the purpose, inputs, outputs, gates, and expected workflow for phase_5_context_system. |
| ADDED | `scripts/phase_5_context_system/benchmark_prefix_cache.py` | Measures full prefill against exact stable-prefix KV reuse and checks logits consistency. |
| MODIFIED | `scripts/phase_5_context_system/check_human_gate.py` | Compatibility gate wrapper/documentation for phase_5_context_system; active runners use the shared gate checker. |
| ADDED | `scripts/phase_5_context_system/check_phase_5_gate.py` | Creates the Phase 5 report/gate from retrieval, reduction, generation, and prefix-cache evidence. |
| MODIFIED | `scripts/phase_5_context_system/context_system.py` | Implements deterministic relevance/salience/recency retrieval, filtering, tier budgets, and exact prefix keys. |
| MODIFIED | `scripts/phase_5_context_system/evaluate_context_system.py` | Measures retrieval accuracy/token reduction and optionally compares actual compact-versus-full generations. |
| MODIFIED | `scripts/phase_5_context_system/run_phase_5.ps1` | Real Phase 5 orchestrator: resolves the selected checkpoint, evaluates retrieval/budgets, benchmarks prefix KV reuse, and creates the gate. |

## Phase 6 — sparse simulation

| Status | File | Responsibility/change |
|---|---|---|
| MODIFIED | `scripts/phase_6_sparse_simulation/EXECUTION_MANUAL.md` | Provides step-by-step commands, recovery guidance, and review requirements for phase_6_sparse_simulation. |
| ADDED | `scripts/phase_6_sparse_simulation/FILE_RESPONSIBILITIES.md` | Lists the responsibility and non-responsibility of every executable/document in phase_6_sparse_simulation. |
| MODIFIED | `scripts/phase_6_sparse_simulation/README.md` | Explains the purpose, inputs, outputs, gates, and expected workflow for phase_6_sparse_simulation. |
| ADDED | `scripts/phase_6_sparse_simulation/build_calibration_corpus.py` | Builds a deduplicated, non-sealed, target-stripped activation calibration corpus. |
| MODIFIED | `scripts/phase_6_sparse_simulation/check_human_gate.py` | Compatibility gate wrapper/documentation for phase_6_sparse_simulation; active runners use the shared gate checker. |
| ADDED | `scripts/phase_6_sparse_simulation/check_phase_6_gate.py` | Selects a viable sparse ratio or records a commercially valid dense-only result. |
| MODIFIED | `scripts/phase_6_sparse_simulation/profile_activations.py` | Collects per-layer router inputs, block contribution scores, oracle masks, and model provenance. |
| MODIFIED | `scripts/phase_6_sparse_simulation/run_phase_6.ps1` | Real Phase 6 orchestrator: builds non-sealed calibration, collects labels, runs masked generations, and creates the quality frontier/gate. |
| MODIFIED | `scripts/phase_6_sparse_simulation/simulate_sparsity.py` | Runs contribution accounting plus actual dense-compute masked generation comparisons across active ratios. |

## Phase 7 — runtime

| Status | File | Responsibility/change |
|---|---|---|
| MODIFIED | `scripts/phase_7_sparse_runtime/EXECUTION_MANUAL.md` | Provides step-by-step commands, recovery guidance, and review requirements for phase_7_sparse_runtime. |
| ADDED | `scripts/phase_7_sparse_runtime/FILE_RESPONSIBILITIES.md` | Lists the responsibility and non-responsibility of every executable/document in phase_7_sparse_runtime. |
| MODIFIED | `scripts/phase_7_sparse_runtime/README.md` | Explains the purpose, inputs, outputs, gates, and expected workflow for phase_7_sparse_runtime. |
| MODIFIED | `scripts/phase_7_sparse_runtime/benchmark_runtime.py` | Measures dense versus gathered sparse decode, quality, latency, VRAM, active blocks, and fallback; supports dense-only mode. |
| MODIFIED | `scripts/phase_7_sparse_runtime/check_human_gate.py` | Compatibility gate wrapper/documentation for phase_7_sparse_runtime; active runners use the shared gate checker. |
| ADDED | `scripts/phase_7_sparse_runtime/check_phase_7_gate.py` | Creates the Phase 7 deployment recommendation: DENSE_RUNTIME or SPARSE_FEATURE_FLAG. |
| ADDED | `scripts/phase_7_sparse_runtime/resolve_phase_6_ratio.py` | Resolves the approved Phase 6 active ratio or returns NONE for dense-only execution. |
| MODIFIED | `scripts/phase_7_sparse_runtime/run_phase_7.ps1` | Real Phase 7 orchestrator: trains/benchmarks routers when viable or measures the dense-only runtime when sparsity fails. |
| MODIFIED | `scripts/phase_7_sparse_runtime/train_router.py` | Trains and validates one router per decoder layer and calibrates confidence thresholds. |

## Phase 8 — release candidate

| Status | File | Responsibility/change |
|---|---|---|
| MODIFIED | `scripts/phase_8_release_candidate/EXECUTION_MANUAL.md` | Provides step-by-step commands, recovery guidance, and review requirements for phase_8_release_candidate. |
| ADDED | `scripts/phase_8_release_candidate/FILE_RESPONSIBILITIES.md` | Lists the responsibility and non-responsibility of every executable/document in phase_8_release_candidate. |
| MODIFIED | `scripts/phase_8_release_candidate/README.md` | Explains the purpose, inputs, outputs, gates, and expected workflow for phase_8_release_candidate. |
| MODIFIED | `scripts/phase_8_release_candidate/build_release_audit.py` | Checks prior gates, sealed metrics, deployment mode, provenance hashes, and release readiness. |
| ADDED | `scripts/phase_8_release_candidate/collect_ablation_evidence.py` | Collects existing A–I ablation artifacts and marks missing evidence without inventing results. |
| ADDED | `scripts/phase_8_release_candidate/create_phase_8_gate.py` | Creates the final pending human release stop gate. |
| ADDED | `scripts/phase_8_release_candidate/package_release_candidate.py` | Creates a manifest-only or copied release package with hashes and audit/reference metadata. |
| MODIFIED | `scripts/phase_8_release_candidate/run_phase_8.ps1` | Real Phase 8 orchestrator: runs sealed evaluation, collects ablation evidence, audits, packages, and creates the final gate. |

## Data/evaluation fixtures

| Status | File | Responsibility/change |
|---|---|---|
| ADDED | `data/context/context_eval.jsonl` | Auditable Phase 5 retrieval/context test cases, including expected and forbidden memories. |

## Tests

| Status | File | Responsibility/change |
|---|---|---|
| ADDED | `tests/test_post_phase3_pipeline.py` | Regression tests for soup decisions, retrieval filtering, gate deviations, prompt budgets, and runner execution. |
| MODIFIED | `tests/test_sparse_mlp.py` | Regression tests for active-ratio/fallback accounting in the sparse decode reference. |

## Removed files

None. Existing paths were preserved to avoid breaking user commands.
