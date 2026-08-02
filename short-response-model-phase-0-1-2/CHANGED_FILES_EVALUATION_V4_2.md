# Changed files — Evaluator v4.2

Evaluator v4.2 is a repository-wide quality-policy update over v1.4.1.

## Core evaluator and policy

- `scripts/common/observable_quality.py` — category-independent multi-signal decoding corruption, technical/multilingual false-positive protections, and prompt-grounded functional inference.
- `scripts/common/quality_evaluation.py` — evaluator version 4.2, universal weights, category-neutral contracts, prompt-function scoring, and category-invariance evidence.
- `scripts/common/opportunity_evaluation.py` — policy 1.4; category statistics remain visible but contribute zero promotion weight.
- `configs/quality_evaluation.json` — one universal weight profile; category overrides removed.
- `configs/quality_issue_taxonomy_v4_2.json` — 60 concrete scored labels and zero-cost review diagnostics.
- `configs/relative_opportunity_evaluation.json` — category-tail weights removed from every profile.

## Benchmark and calibration

- `scripts/common/build_v4_2_response_contract_benchmark.py` — migrates explicit objective contracts while removing category-derived response modes and question rules.
- `data/phase_2_structural_reduction/probe_50_v4_2.jsonl`
- `data/phase_2_structural_reduction/probe_150_thorough_v4_2.jsonl`
- corresponding v4.2 manifests.
- `scripts/common/build_v4_2_comprehensive_calibration.py` — builds the supported-surface regression packet.
- `data/evaluation/quality_v4_2_comprehensive_calibration.jsonl` — 444 adjudicated cases.
- `scripts/common/verify_quality_evaluation_v4_2.py` — validates all expectations, category invariance, legacy-mode invariance, all scored labels, and 150/150 real collapse cases.
- `scripts/common/verify_quality_evaluation.py` — compatibility entry point now invokes v4.2.
- `reports/QUALITY_V4_2_COMPREHENSIVE_CALIBRATION_REPORT.md`
- `reports/QUALITY_V4_2_REAL_COLLAPSE_REGRESSION_REPORT.md`

## Rescoring and Phase 2B

- `scripts/common/rescore_quality_artifacts.py` — strips and rebuilds v4.2 evidence fields.
- `scripts/common/audit_quality_evidence.py` — current schema wording.
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py` — requires v4.2 case contracts.
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_2.ps1` — official review-only or in-place migration wrapper.
- older rescore wrappers now forward to v4.2.
- `scripts/phase_2_layer_selection/build_structural_probe.py` — category-balanced sampling remains, but contract creation no longer infers behavior from category.
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1` — current evaluator documentation.
- Phase 2B README and manual updated to v4.2 commands.

## Phase configuration propagation

- `configs/phase_2_progressive_reduction.yaml`
- `configs/phase_2c_targeted_recovery.yaml`
- `configs/phase_4b_structural_reduction.yaml`

These now require `quality_evaluation_version: "4.2"` so mixed-policy evidence cannot be promoted accidentally.

## Tests

- Current quality, opportunity, Phase 2B, Phase 2C, post-Phase-3, rescore, and migration fixtures now use v4.2.
- `tests/test_quality_v42_comprehensive.py` verifies the 444-case packet.
- `tests/test_quality_v41_comprehensive.py` remains as a compatibility test entry point and now tracks the current v4.2 surface.
- New tests prove category changes/removal cannot alter scoring, legacy `expected_modes` cannot constrain output, category statistics cannot alter opportunity loss, and real decoding corruption is critical regardless of category.

## Documentation

- `README.md`
- `EVALUATION_V4_2_CATEGORY_NEUTRAL_AND_DECODING_INTEGRITY.md`
- `EVALUATION_V4_2_MIGRATION.md`
- this file.
