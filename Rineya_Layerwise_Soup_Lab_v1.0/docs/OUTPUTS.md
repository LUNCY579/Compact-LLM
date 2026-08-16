# Output Artifacts

## Compatibility and provenance

- `manifests/compatibility.json`: tensor/config/tokenizer checks and source hashes.
- `manifests/compatibility.md`: human-readable summary.
- `manifests/dataset_split_audit.json`: dev/test leakage audit.
- `manifests/environment.json`: Python, package, CUDA, and GPU details.
- `manifests/preflight.json`: top-level preflight record.

## Evaluation evidence

Each candidate/split/protocol directory contains:

- `lab_evaluation_manifest.json`
- `results.jsonl`
- `summary.json`
- `candidate_result.json`
- `MODEL_EVALUATION.md`

## Global analysis

- `endpoint_comparison.json`
- `uniform.json`
- `interpolation_curve.csv`
- `global_selection.json`

## Layer analysis

- `per_response_layer_influence.jsonl`: case × layer matrix.
- `layer_influence_summary.csv`: means, intervals, reliability, targets, and seed coefficients.
- `layer_influence_by_category.csv`: category-specific mean effect and sign rates.
- `delta_influence_join.csv`: weight-delta diagnostics joined to behavioral influence.
- `delta_influence_correlation.json`: descriptive permutation/bootstrap correlation analysis.
- `centered_layer_grid.csv`: every single-layer coefficient candidate.
- `layerwise_seed_selection.json`
- `coordinate_search_trace.jsonl`
- `layerwise_best_selection.json`

## Probability and secondary-protocol diagnostics

- `analysis/probability/token_profile.jsonl`: teacher-forced endpoint token diagnostics.
- `analysis/probability/summary.json`: coarse JS lower bound, target/EOS probabilities, and overlap.
- `analysis/protocols/secondary_protocols.json`: post-freeze finalist results under non-selection protocols.

## Selection and test

- `selection/selection_freeze.json`: immutable development-based choice.
- `sealed/access_ledger.json`: started/completed access records.
- `sealed/sealed_test_result.json`: final descriptive evidence.

## Reports

- `reports/RESEARCH_REPORT.md`
- `reports/report_manifest.json`
- `reports/HUMAN_REVIEW_PACKET.jsonl`
- `reports/HUMAN_REVIEW_PACKET.md`
- `reports/plots/global_interpolation.png`
- `reports/plots/layer_influence.png`
- `reports/plots/layer_coefficients.png`
- `reports/plots/probability_target_scatter.png` when endpoint probability profiling is enabled and usable

## Checkpoint artifacts

Every materialized candidate contains `merge_recipe.json`, `merge_manifest.json`, and `tensor_coefficients.json`. `merge_manifest.json` records endpoint fingerprints and a path-independent output-checkpoint digest. These files are required to reconstruct and audit the candidate.
