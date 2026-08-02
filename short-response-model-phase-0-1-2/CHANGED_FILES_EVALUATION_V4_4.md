# Evaluator v4.4 changed-file guide

## Core evaluator

- `scripts/common/observable_quality.py` — adds narrow, case-authored semantic cues and keeps them category-neutral.
- `scripts/common/quality_evaluation.py` — advances the official evaluator schema to 4.4.
- `configs/quality_evaluation.json` — selects the v4.4 policy and taxonomy.
- `configs/quality_issue_taxonomy_v4_4.json` — documents 61 scored labels, including the new precise compression claim label.

## Benchmark and calibration

- `scripts/common/build_v4_4_response_contract_benchmark.py` — patches final paraphrase, rejection, visibility, scope, update, and compression rules.
- `data/phase_2_structural_reduction/probe_150_thorough_v4_4.jsonl` — official thorough Phase 2B benchmark.
- `data/phase_2_structural_reduction/probe_50_v4_4.jsonl` — official fast probe.
- `scripts/common/build_v4_4_comprehensive_calibration.py` — builds the complete v4.4 packet.
- `data/evaluation/quality_v4_4_comprehensive_calibration.jsonl` — 620 adjudicated expectations.
- `data/evaluation/phase2b_v4_2_collapsed_outputs_150_human_adjudication.jsonl` — durable collapse regression corpus.
- `data/evaluation/phase2b_v4_3_stage11_outputs_149_rescored_v4_4.jsonl` — old/new scores for the complete rows available in the latest pasted log.
- `scripts/common/verify_quality_evaluation_v4_4.py` — strict coverage and calibration verifier.
- `tests/test_quality_v44_final_edges.py` — direct final-edge regressions.

## Phase 2B migration

- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_4.ps1` — no-inference preview or official migration wrapper.
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py` — requires explicit v4.4 contracts and writes v4.4 migration provenance.
- `scripts/phase_2_layer_selection/progressive_reduction.py` — requires v4.4 evidence for automated selection.
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1` — defaults to the v4.4 probe.
- `scripts/phase_2_layer_selection/build_structural_probe.py` — builds v4.4 contracts.
- older rescore wrappers now forward to the authoritative v4.4 wrapper.

## Repository-wide version propagation

The Phase 2B, Phase 2C, Phase 4B, common audit/rescore utilities, and affected tests now require `quality_evaluation_version: 4.4`. Documentation and default commands point to the v4.4 benchmark.

## Reports and documentation

- `EVALUATION_V4_4_FINAL_EDGE_REFINEMENT.md`
- `EVALUATION_V4_4_MIGRATION.md`
- `reports/QUALITY_V4_4_COMPREHENSIVE_CALIBRATION_REPORT.md`
- `reports/QUALITY_V4_4_FINAL_EDGE_REGRESSION_REPORT.md`
- `CHANGED_FILES_EVALUATION_V4_4.md`
