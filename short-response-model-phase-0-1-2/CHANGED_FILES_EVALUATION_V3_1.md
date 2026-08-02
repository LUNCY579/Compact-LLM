# Changed Files — Quality Evaluator v3.1

## Core evaluator

- `scripts/common/semantic_quality.py` — new deterministic semantic-coherence, relevance, intent, numeric, polarity, and content-preservation analysis.
- `scripts/common/quality_evaluation.py` — integrates semantic dimensions, issue labels, summary rates, and schema 3.1.
- `configs/quality_evaluation.json` — versioned v3.1 dimensions, category weights, semantic diagnostics, and issue costs.
- `scripts/common/opportunity_evaluation.py` — adds semantic dimensions and semantic-failure rates to proportional comparison.
- `configs/relative_opportunity_evaluation.json` — opportunity-policy schema 1.1 and three updated sensitivity profiles.

## Rescoring and Phase 2B replay

- `scripts/common/rescore_quality_artifacts.py` — strips/replaces all v3.1 quality fields and writes versioned sibling directories.
- `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py` — inference-free tree migration, backups, deviation rebuilding, decision invalidation, and replay preparation.
- `scripts/phase_2_layer_selection/run_phase_2b_rescore_v3_1.ps1` — Windows migration wrapper.
- `scripts/phase_2_layer_selection/progressive_reduction.py` — evaluator-version provenance and parent-lineage validation; stale downstream evidence is removed after a corrected frontier divergence.
- `scripts/phase_2_layer_selection/compare_outputs.py` — deviation evidence now records the current evaluator schema.

## Calibration, audit, and tests

- `data/evaluation/quality_v3_1_calibration.jsonl` — policy examples plus actual Phase 2B semantic-collapse regressions.
- `scripts/common/calibrate_quality_profiles.py` — v3.1 defaults and semantic diagnostic columns.
- `reports/QUALITY_V3_1_CALIBRATION_REPORT.md` — generated calibration evidence.
- `scripts/common/verify_quality_evaluation.py` — deterministic v3.1 contract verification.
- `scripts/common/audit_quality_evidence.py` — requires semantic-integrity summary fields.
- `tests/test_quality_evaluation.py` — regression coverage for informal language, irrelevant nonempty replies, and actual terminal-output collapse.
- Version/provenance fixtures in other tests and phase producers were updated from 3.0 to 3.1.

## Documentation and version propagation

- `EVALUATION_V3_1_SEMANTIC_INTEGRITY_HOTFIX.md`
- `EVALUATION_V3_MIGRATION.md`
- `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md`
- `README.md`
- Phase README files and evaluator-version configuration files.
