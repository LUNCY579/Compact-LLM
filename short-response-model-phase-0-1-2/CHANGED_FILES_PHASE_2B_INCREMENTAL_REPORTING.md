# Phase 2B Incremental Reporting — Changed Files

Release target: `v1.4.6 Incremental Reporting`

## Added

| File | Purpose |
|---|---|
| `scripts/common/incremental_results.py` | Durable append-and-fsync JSONL store with stable-ID resume, deduplication, recovery warnings, and deterministic final ordering. |
| `scripts/phase_2_layer_selection/incremental_reporting.py` | Model-free candidate lifecycle, live progress, stage state/reports, cumulative partial report, legacy artifact inference, and offline reconstruction. |
| `scripts/phase_2_layer_selection/build_phase_2b_report.py` | Runnable no-inference command for rebuilding candidate, stage, partial, and eligible final reports. |
| `tests/test_phase_2b_incremental_reporting.py` | Lightweight unit/integration coverage for persistence, resume, recovery, status, gating, promotion safety, partial reports, compatibility, failure handling, paths, and smoke rebuilding. |
| `tests/test_evaluate_model_incremental_progress.py` | Regression coverage for live progress during the full-reference evaluation when no candidate status file exists. |
| `CHANGED_FILES_PHASE_2B_INCREMENTAL_REPORTING.md` | Release change inventory and file responsibility summary. |

## Modified

| File | Purpose of change |
|---|---|
| `scripts/common/experiment_utils.py` | Added atomic bytes/text/JSON/JSONL replacement, orphaned sibling-temp recovery, durable JSONL append, truncated-final-line recovery, and stable-ID deduplication. Existing hashing, environment, and subprocess helpers remain compatible. |
| `scripts/phase_1_dense_sft/evaluate_model.py` | Added per-example durable persistence, evaluation manifests, stable-ID resume, partial progress files, Phase 2B live/status integration, atomic summary/Markdown output, and interruption/failure handling. |
| `scripts/phase_2_layer_selection/progressive_reduction.py` | Added candidate status initialization/transitions, immediate gate persistence, candidate/stage/partial/live reports, in-progress catalog snapshots, resume reuse of gated candidates, explicit promotion authorization, interruption reporting, exact resume command, and final-state synchronization. |
| `scripts/phase_2_layer_selection/run_phase_2_structural_reduction.ps1` | Added progress interval, live Markdown toggle, real report rebuild mode, partial-report option, and exit-code propagation while preserving existing run parameters. |
| `scripts/phase_2_layer_selection/README.md` | Documented live files, artifact tree, candidate states, watch/rebuild/resume commands, backward compatibility, and promotion safety; corrected evaluator version to v4.5. |
| `scripts/phase_2_layer_selection/STRUCTURAL_REDUCTION_MANUAL.md` | Added operational guidance for incremental persistence, interruption, resume, report rebuilding, temp recovery, and limitations. |
| `scripts/phase_2_layer_selection/FILE_RESPONSIBILITIES.md` | Added responsibilities and prohibitions for incremental reporting, report rebuilding, durable JSONL storage, and atomic utilities. |
| `README.md` | Added the v1.4.6 incremental evidence/reporting overview and no-inference rebuild command. |
| `tests/test_phase_2_progressive_reduction.py` | Extended existing wrapper regression checks for progress controls, real report rebuilding, and Python exit-code propagation. |

## Deleted

None.

## Promotion-safety invariant

`progressive_reduction.py` refuses to materialize a candidate as the next parent unless its durable `candidate_status.json` records both:

```text
status = accepted
promotion_allowed = true
```

Rejected candidates remain preserved as evidence but cannot become the parent of another pruning stage.
