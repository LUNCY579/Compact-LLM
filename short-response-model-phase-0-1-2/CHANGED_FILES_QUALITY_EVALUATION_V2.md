# Changed Files — Quality Evaluation v2.0

This update changes evaluation and promotion logic, not model-training weights. It adds a shared quality contract, migrates every generation-based phase and the Phase 2B playground, strengthens documentation, and adds deterministic verification.

## New files

| File | Responsibility |
|---|---|
| `configs/quality_evaluation.json` | Versioned dimension weights, category overrides, hard-failure thresholds, and aggregate defaults |
| `scripts/common/quality_evaluation.py` | Case scoring, hard collapse detection, semantic audit, category/tail aggregation, and compatibility aliases |
| `scripts/common/quality_gate.py` | Tail-aware quality index, reference validation, global and category-specific comparison gates |
| `scripts/common/audit_quality_evidence.py` | Read-only discovery of legacy, incomplete, malformed, and valid v2 summaries |
| `scripts/common/verify_quality_evaluation.py` | GPU-free deterministic evaluator contract smoke test |
| `tests/test_quality_evaluation.py` | Regression coverage for collapse, leakage, required content, fragmented lines, tails, and category semantic loss |
| `QUALITY_EVALUATION_V2_AUDIT.md` | Full problem analysis, weighting rationale, phase crosswalk, and evidence boundary |
| `QUALITY_EVALUATION_V2_MIGRATION.md` | Safe instructions for re-evaluating old checkpoints without unnecessary retraining |
| `QUALITY_EVALUATION_V2_VERIFICATION.md` | Source-level verification results and limitations |

## Core and configuration changes

- `README.md` — adds the repository-wide quality-v2 contract and migration warning.
- `verify_post_phase_3.ps1` — runs compilation, full pytest, quality-v2 verification, runner checks, response-control verification, and optional imports.
- `configs/phase_2_progressive_reduction.yaml` — tighter mean, category, semantic, completion, tail, alignment, intelligibility, major-issue, and NLL limits.
- `configs/phase_2c_targeted_recovery.yaml` — v2 quality-loss/index floors and compatibility aliases.
- `configs/phase_4b_structural_reduction.yaml` — v2 structural candidate floors.
- `configs/sparse_runtime_pilot.yaml` — v2 sparse candidate quality requirements.

## Phase 1

- `scripts/phase_1_dense_sft/evaluate_model.py` — replaces the three-check aggregate with quality-v2 while retaining the legacy score as a diagnostic; prints explicit hard/major flags and writes dimensions, category, tail, and issue metrics.
- `scripts/phase_1_dense_sft/check_phase_1_gate.py` — rewrites the gate around independent quality, tail, category, semantic, completion, brevity, critical, and major checks.
- `scripts/phase_1_dense_sft/README.md` — documents evaluator versioning and stale-evidence behavior.

## Phase 2 and Phase 2B

- `scripts/phase_2_layer_selection/phase2_common.py` — propagates v2 fields into layer-study rows.
- `scripts/phase_2_layer_selection/run_static_layer_study.py` — invalidates stale caches, uses v2 contribution/ranking, and keeps historical aliases only for compatibility.
- `scripts/phase_2_layer_selection/scan_layers.py` — records hard, pass, tail, category, and mean quality for every layer.
- `scripts/phase_2_layer_selection/collect_importance_signals.py` — uses v2 quality for direct layer contribution.
- `scripts/phase_2_layer_selection/check_phase_2_gate.py` — requires v2 finalists, independent quality floors, resource savings, and repeated-seed stability.
- `scripts/phase_2_layer_selection/progressive_reduction.py` — requires a valid v2 reference, rebuilds stale summaries, tightens quality/NLL gates, rejects new hard/critical/major issues, and reprioritizes ranking.
- `scripts/phase_2_layer_selection/compare_outputs.py` — adds quality deltas and new hard-failure evidence to case comparisons.
- `scripts/phase_2_layer_selection/build_phase_2_human_review.py` — prioritizes quality, intelligibility, pass, hard, and major fields for review.
- `scripts/phase_2_layer_selection/README.md` — documents v2 evidence requirements.

## Phase 2C

- `scripts/phase_2c_targeted_recovery/phase2c_common.py` — uses v2 mean/index/tail and independent category quality, semantic, and completion loss; enforces hard/new-major rules and resource reduction.
- `scripts/phase_2c_targeted_recovery/rank_surviving_layers.py` — reads the corrected v2 Phase 2B assessment keys while retaining legacy fallbacks.
- `scripts/phase_2c_targeted_recovery/README.md` — documents v2 migration and gates.

## Phase 3

- `scripts/phase_3_preference_alignment/check_phase_3_gate.py` — rewrites the DPO comparison around v2 mean, index, tail, category, semantic, alignment, completion, intelligibility, and issue preservation.
- `scripts/phase_3_preference_alignment/README.md` — documents v2 evidence requirements.

## Phase 4 and Phase 4B

- `scripts/phase_4_model_soups/build_greedy_soup.py` — rejects ineligible summaries, uses quality-index improvement, and reports exact category regression names.
- `scripts/phase_4_model_soups/greedy_soup_plan.py` — ranks ingredients by v2 hard/critical/major, tail, and mean evidence.
- `scripts/phase_4_model_soups/README.md` — documents v2 evidence requirements.
- `scripts/phase_4b_structural_reduction/evaluate_candidates.py` — writes a v2 candidate registry.
- `scripts/phase_4b_structural_reduction/recommend_reference_candidate.py` — rewrites candidate recommendation around valid dense reference, quality index, tail, categories, semantics, alignment, issues, and structural savings.
- `scripts/phase_4b_structural_reduction/README.md` — documents v2 evidence requirements.

## Phase 5

- `scripts/phase_5_context_system/evaluate_context_system.py` — scores compact and full-context replies case by case with the same v2 contract and preserves both aggregate and trace evidence.
- `scripts/phase_5_context_system/check_phase_5_gate.py` — prevents retrieval success from hiding reply-quality damage.
- `scripts/phase_5_context_system/README.md` — documents v2 evidence requirements.

## Phase 6

- `scripts/phase_6_sparse_simulation/simulate_sparsity.py` — applies identical v2 scoring to dense and masked candidates and records quality comparison evidence per ratio.
- `scripts/phase_6_sparse_simulation/check_phase_6_gate.py` — requires v2 mean/index/tail/category/semantic/alignment/intelligibility preservation and zero new hard/major collapse.
- `scripts/phase_6_sparse_simulation/README.md` — documents v2 evidence requirements.

## Phase 7

- `scripts/phase_7_sparse_runtime/benchmark_runtime.py` — scores dense/sparse runtime replies with v2 and includes case-level quality deltas.
- `scripts/phase_7_sparse_runtime/check_phase_7_gate.py` — permits sparse enablement only when both quality and measured runtime gates pass; dense fallback remains valid.
- `scripts/phase_7_sparse_runtime/resolve_phase_6_ratio.py` — rejects stale or quality-ineligible Phase 6 ratios.
- `scripts/phase_7_sparse_runtime/README.md` — documents v2 evidence requirements.

## Phase 8

- `scripts/phase_8_release_candidate/build_release_audit.py` — requires sealed quality-v2 evidence, strict one-sentence control, tail/category floors, and zero hard, critical, major, and category-hard failures.
- `scripts/phase_8_release_candidate/collect_ablation_evidence.py` — marks legacy summaries as stale rather than silently treating them as current evidence.
- `scripts/phase_8_release_candidate/README.md` — documents release-grade v2 requirements.

## Independent Phase 2B playground

- `Phase2B_Layer_Combination_Playground_v1.0/phase2b_layer_combination_playground.py` — internal version 1.1; rejects stale/collapsed references, rebuilds stale summaries on resume, evaluates all variants with v2, and writes exploratory PASS/FAIL reasons.
- `Phase2B_Layer_Combination_Playground_v1.0/README.md` — adds the full all-mode command, report location, v2 interpretation, and resume behavior. The folder name remains unchanged for command compatibility.

## Test fixture updates

- `tests/test_phase_2_progressive_reduction.py`
- `tests/test_phase_2c_targeted_recovery.py`
- `tests/test_post_phase3_pipeline.py`

These fixtures now use evaluator schema 2.0 and validate the stricter phase logic.
