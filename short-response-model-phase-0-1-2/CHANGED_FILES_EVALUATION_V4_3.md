# Changed files — Evaluator v4.3

Evaluator v4.3 is a focused quality-scoring refinement over v1.4.2. It preserves category-neutrality and decoding-integrity behavior while improving open-ended prompt adequacy and strict-contract paraphrase credit.

## Core evaluator and policy

- `scripts/common/quality_evaluation.py` — version 4.3, prompt-adequacy dimension integration, smooth open-case adequacy factor, and new evidence fields.
- `scripts/common/observable_quality.py` — scored evidence options, defaults, adversarial score caps, prompt-grounded adequacy, and robust partial-credit validation.
- `configs/quality_evaluation.json` — v4.3 policy name, universal weights, and v4.3 taxonomy path.
- `configs/quality_issue_taxonomy_v4_3.json` — current 60-label concrete defect taxonomy.
- `scripts/common/audit_quality_evidence.py` — v4.3 schema wording.
- `scripts/common/rescore_quality_artifacts.py` — strips and rebuilds v4.3 quality evidence.

## Benchmark contracts and calibration

- `scripts/common/build_v4_3_response_contract_benchmark.py` — migrates v4.2 cases to case-level v4.3 contracts, adds prompt-grounded adequacy rules, broader paraphrase evidence, partial scores, and adversarial caps without using category or reference wording.
- `data/phase_2_structural_reduction/probe_50_v4_3.jsonl`
- `data/phase_2_structural_reduction/probe_50_v4_3_manifest.json`
- `data/phase_2_structural_reduction/probe_150_thorough_v4_3.jsonl`
- `data/phase_2_structural_reduction/probe_150_thorough_v4_3_manifest.json`
- `data/evaluation/phase2b_v4_2_coherent_outputs_150_human_adjudication.jsonl` — human grades and score bands for the real coherent run.
- `scripts/common/build_v4_3_comprehensive_calibration.py` — builds the 611-case packet.
- `data/evaluation/quality_v4_3_comprehensive_calibration.jsonl`
- `scripts/common/verify_quality_evaluation_v4_3.py` — verifies all expectations, category/reference invariance, legacy-mode invariance, 60 labels, 31 reporting categories, 150 coherent cases, 17 targeted cases, and 150 collapsed cases.
- `scripts/common/verify_quality_evaluation.py` — current compatibility entry point.
- `scripts/common/verify_quality_evaluation_v4_2.py` — deprecated compatibility alias forwarding to v4.3.

## Phase 2B migration and execution

- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py` — v4.3 contract validation and migration naming.
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_3.ps1` — official preview/in-place v4.3 wrapper.
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_2.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_1.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v3_1.ps1` — old wrappers now forward to v4.3.
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1` — v4.3 default probe and help text.
- `scripts/phase_2_layer_selection/progressive_reduction.py` — v4.3 default and contracted-probe guidance.
- `scripts/phase_2_layer_selection/build_structural_probe.py` — builds v4.3 contracts by default.
- `configs/phase_2_progressive_reduction.yaml`
- `configs/phase_2c_targeted_recovery.yaml`
- `configs/phase_4b_structural_reduction.yaml` — all require quality evaluator 4.3 evidence.

## Tests

- `tests/test_quality_v43_focused.py` — focused false-high/false-low, paraphrase, subject-reversal, and invariance regressions.
- Current quality, opportunity, Phase 2B, Phase 2C, migration, and post-Phase-3 tests were updated to v4.3 evidence.
- Compatibility test entry points for v4.1/v4.2 now verify the authoritative v4.3 surface.

## Documentation and reports

- `EVALUATION_V4_3_FOCUSED_PROMPT_ADEQUACY.md`
- `EVALUATION_V4_3_MIGRATION.md`
- `README.md`
- `scripts/phase_2_layer_selection/README.md`
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md`
- `reports/QUALITY_V4_3_COMPREHENSIVE_CALIBRATION_REPORT.md`
- `reports/QUALITY_V4_3_REAL_COHERENT_REGRESSION_REPORT.md`
- `EVALUATOR_V4_3_HOTFIX_MANIFEST.json` — machine-readable repository-relative file list and hashes.

## Verification result

- Pytest: 76 passed.
- Unittest: 47 passed.
- Calibration: 611/611 expectations passed.
- Real collapsed responses: 150/150 zero-quality critical.
- Real coherent responses: 150/150 within human-authored score bands.
- Relative-opportunity policy audit: passed.
- Python compilation: passed.
