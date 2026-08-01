# Changed Files — Relative Opportunity Evaluation v3

**Baseline:** v1.2.1 Phase2B Wrapper Sync  
**Release:** v1.3 Relative Opportunity Evaluation

This release replaces absolute promotion floors with one shared proportional opportunity engine, introduces conversationally permissive Quality Evaluator v3, synchronizes every official phase and the independent playground, and adds non-destructive rescoring/migration support.

**Total changed or added paths: 82**

## Policy, configuration, and migration

- `BLUEPRINT_AMENDMENT_RELATIVE_OPPORTUNITY_v1.0.md`
- `EVALUATION_V3_MIGRATION.md`
- `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`
- `configs/phase_2_progressive_reduction.yaml`
- `configs/phase_2c_targeted_recovery.yaml`
- `configs/phase_4b_structural_reduction.yaml`
- `configs/quality_evaluation.json`
- `configs/relative_opportunity_evaluation.json`
- `configs/sparse_runtime_pilot.yaml`

## Shared evaluation infrastructure

- `scripts/common/audit_quality_evidence.py`
- `scripts/common/audit_relative_opportunity_policy.py`
- `scripts/common/calibrate_quality_profiles.py`
- `scripts/common/opportunity_evaluation.py`
- `scripts/common/quality_evaluation.py`
- `scripts/common/quality_gate.py`
- `scripts/common/rescore_quality_artifacts.py`
- `scripts/common/verify_quality_evaluation.py`

## Phase implementations

- `scripts/phase_0_data/README.md`
- `scripts/phase_1_dense_sft/README.md`
- `scripts/phase_1_dense_sft/check_phase_1_gate.py`
- `scripts/phase_1_dense_sft/evaluate_model.py`
- `scripts/phase_2_layer_selection/README.md`
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md`
- `scripts/phase_2_layer_selection/check_phase_2_gate.py`
- `scripts/phase_2_layer_selection/phase2_common.py`
- `scripts/phase_2_layer_selection/progressive_reduction.py`
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py`
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1`
- `scripts/phase_2_layer_selection/run_static_layer_study.py`
- `scripts/phase_2_layer_selection/scan_layers.py`
- `scripts/phase_2c_targeted_recovery/EXECUTION_MANUAL.md`
- `scripts/phase_2c_targeted_recovery/README.md`
- `scripts/phase_2c_targeted_recovery/check_phase_2c_gate.py`
- `scripts/phase_2c_targeted_recovery/phase2c_common.py`
- `scripts/phase_2c_targeted_recovery/rank_surviving_layers.py`
- `scripts/phase_2c_targeted_recovery/run_phase_2c.ps1`
- `scripts/phase_2c_targeted_recovery/targeted_recovery.py`
- `scripts/phase_3_preference_alignment/README.md`
- `scripts/phase_3_preference_alignment/check_phase_3_gate.py`
- `scripts/phase_4_model_soups/FILE_RESPONSIBILITIES.md`
- `scripts/phase_4_model_soups/README.md`
- `scripts/phase_4_model_soups/build_greedy_soup.py`
- `scripts/phase_4_model_soups/greedy_soup_plan.py`
- `scripts/phase_4_model_soups/run_phase_4.ps1`
- `scripts/phase_4b_structural_reduction/FILE_RESPONSIBILITIES.md`
- `scripts/phase_4b_structural_reduction/README.md`
- `scripts/phase_4b_structural_reduction/create_phase_4b_gate.py`
- `scripts/phase_4b_structural_reduction/evaluate_candidates.py`
- `scripts/phase_4b_structural_reduction/recommend_reference_candidate.py`
- `scripts/phase_5_context_system/FILE_RESPONSIBILITIES.md`
- `scripts/phase_5_context_system/README.md`
- `scripts/phase_5_context_system/check_phase_5_gate.py`
- `scripts/phase_5_context_system/context_system.py`
- `scripts/phase_5_context_system/evaluate_context_system.py`
- `scripts/phase_6_sparse_simulation/README.md`
- `scripts/phase_6_sparse_simulation/check_phase_6_gate.py`
- `scripts/phase_6_sparse_simulation/profile_activations.py`
- `scripts/phase_6_sparse_simulation/run_phase_6.ps1`
- `scripts/phase_6_sparse_simulation/simulate_sparsity.py`
- `scripts/phase_7_sparse_runtime/README.md`
- `scripts/phase_7_sparse_runtime/benchmark_runtime.py`
- `scripts/phase_7_sparse_runtime/check_phase_7_gate.py`
- `scripts/phase_7_sparse_runtime/resolve_phase_6_ratio.py`
- `scripts/phase_7_sparse_runtime/train_router.py`
- `scripts/phase_8_release_candidate/README.md`
- `scripts/phase_8_release_candidate/build_release_audit.py`
- `scripts/phase_8_release_candidate/collect_ablation_evidence.py`

## Independent playground

- `Phase2B_Layer_Combination_Playground_v1.0/README.md`
- `Phase2B_Layer_Combination_Playground_v1.0/phase2b_layer_combination_playground.py`
- `Phase2B_Layer_Combination_Playground_v1.0/run_playground.ps1`

## Tests and calibration

- `data/evaluation/CALIBRATION_GUIDE.md`
- `data/evaluation/quality_v3_calibration.jsonl`
- `reports/QUALITY_V3_CALIBRATION_REPORT.md`
- `tests/test_phase_2_progressive_reduction.py`
- `tests/test_phase_2c_targeted_recovery.py`
- `tests/test_post_phase3_pipeline.py`
- `tests/test_quality_evaluation.py`
- `tests/test_relative_opportunity_policy.py`

## Repository documentation

- `CHANGED_FILES_RELATIVE_OPPORTUNITY_V3.md`
- `Phases.md`
- `README.md`
- `reports/README_EVALUATION_POLICY.md`

## Validation performed

- Python compilation: passed.
- Relative-opportunity static policy audit: passed.
- Quality Evaluator v3 deterministic contract: passed.
- Calibration packet report generation: passed.
- Test suite: 54 passed.
- JSON/YAML parsing: passed.
- PowerShell-to-Python referenced-file scan: passed.

Actual Qwen/CUDA generation, training, Windows PowerShell execution, and hardware timing are target-machine activities and were not fabricated in this package.
