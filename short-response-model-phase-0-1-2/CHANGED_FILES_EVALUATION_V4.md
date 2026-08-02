# Changed files — Quality Evaluator v4.0

This hotfix replaces Evaluator v3.1 with an observable-defect and explicit-response-contract architecture. It is intentionally repository-wide because every phase that reads or compares quality evidence must use the same evaluator version and field names.

- Modified existing files: **52**
- New files: **18**
- Deleted files: **0**

## Behavioral changes

- Removed subjective or non-problem catch-all labels from scoring, including `semantically_unresponsive_output`, weak lexical relevance, vocabulary rarity, and surface-word polarity contradiction.
- Scored labels now identify a concrete mechanical/structural defect, a direct safety/context/leakage problem, or an explicit `response_contract` violation.
- References, lexical overlap, multilingual text, uncommon vocabulary, minor grammar, and alternate conversational strategies are review-only diagnostics with zero score cost.
- Phase 2B official probes now carry explicit v4 contracts. The official migration refuses uncontracted legacy probes rather than silently inferring requirements.
- Existing generated replies can be rescored without model inference; stale frontier decisions are archived and replayed using compatible evidence.

## Evaluator core and policy

- `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`
- `configs/phase_2_progressive_reduction.yaml`
- `configs/phase_2c_targeted_recovery.yaml`
- `configs/phase_4b_structural_reduction.yaml`
- `configs/quality_evaluation.json`
- `configs/quality_issue_taxonomy_v4.json`
- `configs/relative_opportunity_evaluation.json`
- `configs/sparse_runtime_pilot.yaml`
- `scripts/common/audit_quality_evidence.py`
- `scripts/common/build_v4_response_contract_benchmark.py`
- `scripts/common/calibrate_quality_profiles.py`
- `scripts/common/observable_quality.py`
- `scripts/common/opportunity_evaluation.py`
- `scripts/common/quality_evaluation.py`
- `scripts/common/quality_gate.py`
- `scripts/common/rescore_quality_artifacts.py`
- `scripts/common/review_logged_generation_outputs.py`
- `scripts/common/semantic_quality.py`
- `scripts/common/verify_quality_evaluation.py`

## Phase 2B migration and structural workflow

- `Phase2B_Layer_Combination_Playground_v1.0/phase2b_layer_combination_playground.py`
- `scripts/phase_2_layer_selection/README.md`
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md`
- `scripts/phase_2_layer_selection/build_phase_2_human_review.py`
- `scripts/phase_2_layer_selection/build_structural_probe.py`
- `scripts/phase_2_layer_selection/check_phase_2_gate.py`
- `scripts/phase_2_layer_selection/phase2_common.py`
- `scripts/phase_2_layer_selection/progressive_reduction.py`
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py`
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v3_1.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4.ps1`
- `scripts/phase_2_layer_selection/run_static_layer_study.py`
- `scripts/phase_2_layer_selection/scan_layers.py`

## Contracted benchmark data and adjudication

- `data/evaluation/CALIBRATION_GUIDE.md`
- `data/evaluation/quality_v4_adjudication.jsonl`
- `data/phase_2_structural_reduction/probe_150_thorough_v4.jsonl`
- `data/phase_2_structural_reduction/probe_150_thorough_v4_manifest.json`
- `data/phase_2_structural_reduction/probe_50_v4.jsonl`
- `data/phase_2_structural_reduction/probe_50_v4_manifest.json`

## Repository-wide phase integration

- `scripts/phase_1_dense_sft/evaluate_model.py`
- `scripts/phase_2c_targeted_recovery/phase2c_common.py`
- `scripts/phase_3_preference_alignment/EXECUTION_MANUAL.md`
- `scripts/phase_3_preference_alignment/check_phase_3_gate.py`
- `scripts/phase_4_model_soups/README.md`
- `scripts/phase_4_model_soups/build_greedy_soup.py`
- `scripts/phase_4_model_soups/greedy_soup_plan.py`
- `scripts/phase_4b_structural_reduction/evaluate_candidates.py`
- `scripts/phase_4b_structural_reduction/recommend_reference_candidate.py`
- `scripts/phase_5_context_system/evaluate_context_system.py`
- `scripts/phase_7_sparse_runtime/benchmark_runtime.py`
- `scripts/phase_7_sparse_runtime/check_phase_7_gate.py`
- `scripts/phase_8_release_candidate/build_release_audit.py`
- `scripts/phase_8_release_candidate/collect_ablation_evidence.py`

## Tests, reports, and documentation

- `CHANGED_FILES_EVALUATION_V4.md`
- `EVALUATOR_V4_HOTFIX_MANIFEST.json`
- `EVALUATION_V3_1_SEMANTIC_INTEGRITY_HOTFIX.md`
- `EVALUATION_V3_MIGRATION.md`
- `EVALUATION_V4_MIGRATION.md`
- `EVALUATION_V4_OBSERVABLE_DEFECTS_AND_CONTRACTS.md`
- `README.md`
- `reports/QUALITY_V4_ADJUDICATION_REPORT.md`
- `reports/QUALITY_V4_PASTED_150_REVIEW.json`
- `reports/QUALITY_V4_PASTED_150_REVIEW.md`
- `tests/test_phase_2_progressive_reduction.py`
- `tests/test_phase_2c_targeted_recovery.py`
- `tests/test_post_phase3_pipeline.py`
- `tests/test_quality_evaluation.py`
- `tests/test_quality_v31_rescore.py`
- `tests/test_quality_v4_phase2b_migration.py`
- `tests/test_relative_opportunity_policy.py`

## Verification

- `python -m pytest -q` → **67 passed**
- `python scripts/common/verify_quality_evaluation.py` → **24/24 human-adjudicated cases passed**
- `python scripts/common/audit_relative_opportunity_policy.py` → **passed**
- `python -m compileall -q .` → **passed**
- The supplied 150-output log was rescored and reviewed in `reports/QUALITY_V4_PASTED_150_REVIEW.md`.

## Primary command

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v4.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v4.jsonl
```

See `EVALUATION_V4_MIGRATION.md` for preview, replay/resume, and clean-rerun commands.
