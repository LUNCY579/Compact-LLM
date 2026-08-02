# Changed files — Quality Evaluator v4.1

## Release scope

This release upgrades repository evidence from Quality Evaluator 4.0 / opportunity policy 1.2 to Quality Evaluator 4.1 / opportunity policy 1.3. It adds continuous functional-mode scoring, partial contract credit, adversarial keyword traps, complete supported-surface calibration, Phase 2B no-inference migration, and repository-wide version propagation.

## Main implementation files

- `scripts/common/observable_quality.py` — continuous response-mode confidence, weighted concept/evidence rules, partial contract scoring, and observable defect refinements.
- `scripts/common/quality_evaluation.py` — v4.1 dimensions, functional fulfillment, partial contract integration, separated contract/non-contract issue costs, and quieter diagnostics.
- `scripts/common/opportunity_evaluation.py` — opportunity policy 1.3 and explicit `response_mode_fulfillment` loss/gain.
- `configs/quality_evaluation.json` — authoritative v4.1 weights and behavior.
- `configs/quality_issue_taxonomy_v4_1.json` — versioned scored-label dictionary.
- `configs/relative_opportunity_evaluation.json` — balanced, logic-oriented, and conversational-permissive v1.3 profiles.

## Benchmark and calibration files

- `data/phase_2_structural_reduction/probe_50_v4_1.jsonl` and `probe_150_thorough_v4_1.jsonl` — explicit v4.1 contracts with paraphrase, partial-credit, and keyword-trap corrections.
- `data/evaluation/quality_v4_1_comprehensive_calibration.jsonl` — 276 cases covering 31 categories, 150 official references, 54 scored labels, 9 authored modes, partial answers, real regressions, and adversarial traps.
- `scripts/common/build_v4_1_response_contract_benchmark.py` and `build_v4_1_comprehensive_calibration.py` — deterministic builders.
- `scripts/common/verify_quality_evaluation_v4_1.py` — strict complete-surface verifier.

## Phase 2B migration and execution

- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_1.ps1` — official no-inference review/migration wrapper.
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py` — v4.1 contract validation, backup, deviation rebuild, lineage provenance, and resume preparation.
- `scripts/common/rescore_quality_artifacts.py` — strips all stale v2–v4.0 and v4.1-derived fields before deterministic rescoring.
- Phase 2B defaults, documentation, playground, and legacy wrappers now point to the v4.1 probes and evaluator.

## Repository-wide propagation

- Active Phase 2, 2C, 4, 4B, 7, and 8 outputs now use the current evaluator/opportunity constants instead of stale literal versions.
- `verify_post_phase_3.ps1` now runs the v4.1 comprehensive verifier and relative-opportunity audit.
- Tests and synthetic migration fixtures now require evaluator 4.1 and opportunity policy 1.3.

## Exact hotfix file list

- `CHANGED_FILES_EVALUATION_V4_1.md`
- `EVALUATION_V4_1_CONTINUOUS_FUNCTION_AND_PARTIAL_CREDIT.md`
- `EVALUATION_V4_1_MIGRATION.md`
- `EVALUATOR_V4_1_HOTFIX_MANIFEST.json`
- `Phase2B_Layer_Combination_Playground_v1.0/phase2b_layer_combination_playground.py`
- `README.md`
- `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`
- `configs/phase_2_progressive_reduction.yaml`
- `configs/phase_2c_targeted_recovery.yaml`
- `configs/phase_4b_structural_reduction.yaml`
- `configs/quality_evaluation.json`
- `configs/quality_issue_taxonomy_v4_1.json`
- `configs/relative_opportunity_evaluation.json`
- `configs/sparse_runtime_pilot.yaml`
- `data/evaluation/quality_v4_1_comprehensive_calibration.jsonl`
- `data/phase_2_structural_reduction/probe_150_thorough_v4_1.jsonl`
- `data/phase_2_structural_reduction/probe_150_thorough_v4_1_manifest.json`
- `data/phase_2_structural_reduction/probe_50_v4_1.jsonl`
- `data/phase_2_structural_reduction/probe_50_v4_1_manifest.json`
- `reports/QUALITY_V4_ADJUDICATION_REPORT.md`
- `reports/QUALITY_V4_1_COMPREHENSIVE_CALIBRATION_REPORT.md`
- `reports/QUALITY_V4_1_REAL_PHASE2B_REGRESSION_REVIEW.md`
- `scripts/common/audit_quality_evidence.py`
- `scripts/common/build_v4_1_comprehensive_calibration.py`
- `scripts/common/build_v4_1_response_contract_benchmark.py`
- `scripts/common/calibrate_quality_profiles.py`
- `scripts/common/observable_quality.py`
- `scripts/common/opportunity_evaluation.py`
- `scripts/common/quality_evaluation.py`
- `scripts/common/quality_gate.py`
- `scripts/common/rescore_quality_artifacts.py`
- `scripts/common/semantic_quality.py`
- `scripts/common/verify_quality_evaluation.py`
- `scripts/common/verify_quality_evaluation_v4_1.py`
- `scripts/phase_2_layer_selection/README.md`
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md`
- `scripts/phase_2_layer_selection/build_structural_probe.py`
- `scripts/phase_2_layer_selection/progressive_reduction.py`
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py`
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v3_1.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_1.ps1`
- `scripts/phase_2_layer_selection/run_static_layer_study.py`
- `scripts/phase_4_model_soups/greedy_soup_plan.py`
- `scripts/phase_4b_structural_reduction/evaluate_candidates.py`
- `scripts/phase_4b_structural_reduction/recommend_reference_candidate.py`
- `scripts/phase_7_sparse_runtime/benchmark_runtime.py`
- `scripts/phase_8_release_candidate/build_release_audit.py`
- `scripts/phase_8_release_candidate/collect_ablation_evidence.py`
- `tests/test_phase_2_progressive_reduction.py`
- `tests/test_phase_2c_targeted_recovery.py`
- `tests/test_post_phase3_pipeline.py`
- `tests/test_quality_evaluation.py`
- `tests/test_quality_v31_rescore.py`
- `tests/test_quality_v41_comprehensive.py`
- `tests/test_quality_v4_phase2b_migration.py`
- `tests/test_relative_opportunity_policy.py`
- `verify_post_phase_3.ps1`

**Total changed/new repository files listed: 59.**
