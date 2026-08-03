# Changed Files — Quality Evaluator v4.5

## Release scope

Evaluator v4.5 is a focused correction over v1.4.4. It rebuilds official contracts from visible messages, removes inherited category-authored requirements, fixes the remaining false critical/major labels, and preserves proportional candidate selection.

## Core evaluator and policy

- `scripts/common/quality_evaluation.py`
  - sets the authoritative evaluator version to `4.5`;
  - adds the concrete `direct_self_contradiction` defect;
  - loads the v4.5 taxonomy.
- `scripts/common/observable_quality.py`
  - prevents common function words from independently triggering token-loop collapse;
  - broadens uncertainty, rejection, visibility, and evidence-request paraphrases;
  - preserves precise current-event and contradiction failures.
- `configs/quality_evaluation.json`
  - requires fresh contract provenance;
  - raises the proportional contribution of genuine explicit-contract failures;
  - keeps review diagnostics zero-cost.
- `configs/quality_issue_taxonomy_v4_5.json`
  - contains 63 scored labels;
  - adds `direct_self_contradiction` and `unsupported_current_event_claim`;
  - refines token-loop, consent, quantization, visibility, and uncertainty definitions.
- `scripts/common/audit_quality_evidence.py`
- `scripts/common/rescore_quality_artifacts.py`
- `scripts/common/verify_quality_evaluation.py`
  - point active verification and rescoring documentation to v4.5.

## Fresh contract construction

- `scripts/common/build_v4_5_response_contract_benchmark.py`
  - discards inherited response contracts;
  - authors contracts from visible system/user messages only;
  - never consults category or reference wording;
  - records auditable provenance fields.
- `data/phase_2_structural_reduction/probe_150_thorough_v4_5.jsonl`
- `data/phase_2_structural_reduction/probe_150_thorough_v4_5_manifest.json`
- `data/phase_2_structural_reduction/probe_50_v4_5.jsonl`
- `data/phase_2_structural_reduction/probe_50_v4_5_manifest.json`
  - official v4.5 contracted probes.

## Calibration and real-output regression

- `scripts/common/build_v4_5_comprehensive_calibration.py`
- `scripts/common/verify_quality_evaluation_v4_5.py`
- `data/evaluation/quality_v4_5_comprehensive_calibration.jsonl`
- `reports/QUALITY_V4_5_COMPREHENSIVE_CALIBRATION_REPORT.md`
  - 639/639 expectations;
  - 31/31 reporting categories;
  - 63/63 scored labels;
  - 150/150 real collapses;
  - 150 human-adjudicated coherent outputs;
  - 41 focused regression cases.
- `data/evaluation/phase2b_v4_4_false_positive_outputs_132_rescored_v4_5.jsonl`
- `reports/QUALITY_V4_5_LATEST_132_OUTPUT_REGRESSION_REPORT.md`
  - no-inference rescore of the complete rows from the supplied v4.4 terminal excerpt.

## Phase 2B integration and migration

- `scripts/phase_2_layer_selection/progressive_reduction.py`
  - defaults to the v4.5 probe;
  - requires v4.5 schema and fresh-message provenance;
  - rejects mixed or inherited contract evidence.
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py`
  - validates fresh-message contract provenance before migration;
  - writes `PHASE2B_QUALITY_V4_5_MIGRATION.json`.
- `scripts/phase_2_layer_selection/build_structural_probe.py`
  - uses the v4.5 builder and emits v4.5 probe manifests.
- `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1`
  - defaults to `probe_50_v4_5.jsonl`.
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_5.ps1`
  - new official no-inference Phase 2B migration wrapper.
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v3_1.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_1.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_2.ps1`
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v4_4.ps1`
  - deprecated compatibility aliases now forward to v4.5.
- `scripts/phase_2_layer_selection/README.md`
- `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md`
  - commands and provenance requirements updated.

## Downstream evaluator-version propagation

- `configs/phase_2_progressive_reduction.yaml`
- `configs/phase_2c_targeted_recovery.yaml`
- `configs/phase_4b_structural_reduction.yaml`
  - require Evaluator v4.5 evidence.

## Tests

- `tests/test_quality_v45_fresh_contracts.py`
  - verifies category/reference/inherited-contract invariance;
  - verifies the exact false critical and major cases from the supplied output;
  - verifies precise genuine failures remain detected.
- Updated evaluator-version expectations and v4.5 probe paths in:
  - `tests/test_phase_2_progressive_reduction.py`
  - `tests/test_phase_2c_targeted_recovery.py`
  - `tests/test_post_phase3_pipeline.py`
  - `tests/test_quality_evaluation.py`
  - `tests/test_quality_v31_rescore.py`
  - `tests/test_quality_v41_comprehensive.py`
  - `tests/test_quality_v42_comprehensive.py`
  - `tests/test_quality_v44_final_edges.py`
  - `tests/test_quality_v4_phase2b_migration.py`
  - `tests/test_relative_opportunity_policy.py`

## Documentation

- `EVALUATION_V4_5_FRESH_MESSAGE_CONTRACTS.md`
- `EVALUATION_V4_5_MIGRATION.md`
- `README.md`
- `Phases.md`
  - document the current policy, verification, migration, and commands.

## Verification result

```text
Comprehensive calibration:       639/639 passed
Reporting categories:             31/31 covered
Scored labels:                    63/63 covered
Real collapsed outputs:          150/150 rejected
Latest supplied complete rows:   132 re-scored without inference
Pytest suite:                      85 passed
Unittest suite:                    47 passed
Relative-opportunity audit:       passed
Python compilation:               passed
```
