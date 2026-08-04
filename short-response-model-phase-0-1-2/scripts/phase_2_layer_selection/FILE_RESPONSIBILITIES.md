# Phase 2 file responsibilities

| File | Responsibility | Must not claim/do |
|---|---|---|
| `run_phase_2.ps1` | Select Phase 2A, Phase 2B, or Phase 2C targeted-recovery execution. | Merge their gates into one ambiguous result. |
| `run_phase_2_structural_reduction.ps1` | Validate inputs and launch the complete Phase 2B pipeline; expose live-update controls and the real offline report rebuild command; propagate Python failures. | Auto-approve a reduced profile or merely print commands without executing them. |
| `progressive_reduction.py` | Test every current layer; resume directly from the last accepted frontier; enforce cumulative and parent-relative opportunity; gate and checkpoint each completed candidate before selection; reject parent promotion unless persisted status permits it; refresh live/stage/partial/final reports; preserve profiles, provenance, termination evidence, and model-keyed validation results. | Replay completed generation examples, let a rejected candidate become the next parent, let earlier gains subsidize a damaging deletion, use sealed data, or treat a positive ratio as release approval. |
| `incremental_reporting.py` | Maintain atomic live progress, candidate lifecycle, stage state/reports, cumulative partial reports, legacy status inference, and model-free report reconstruction. | Load a model, perform inference, infer an acceptance decision that saved artifacts do not prove, or label an incomplete stage complete. |
| `build_phase_2b_report.py` | Rebuild candidate, stage, partial, and eligible final reports from saved artifacts without inference. | Rerun evaluation or generate a final report for an incomplete run. |
| `create_depth_candidate.py` | Physically delete one block, reindex cache IDs, preserve original layer lineage, verify reload. | Modify the parent checkpoint in place. |
| `build_structural_probe.py` | Select 50 balanced development records deterministically. | Draw from sealed data. |
| `evaluate_teacher_forcing.py` | Measure target-token NLL as a continuous sensitivity signal. | Present NLL as a complete human-quality metric. |
| `compare_outputs.py` | Produce case-level reply and semantic-flag differences. | Mark every wording change as a regression. |
| `resolve_phase_2_profile.py` | Resolve only human-approved reduced profiles for Phase 3. | Bypass the Phase 2B gate. |
| `run_static_layer_study.py` | Preserve the original Phase 2A selective-tuning experiment. | Reinterpret structural deletion as selective-tuning success. |
| `build_phase_2_human_review.py` | Surface the strongest Phase 2A candidate beside the all-layer, RANDOM4, and BOTTOM4 outputs. | Replace the formal gate or hide missing raw evaluations. |
| `collect_importance_signals.py` | Summarize selected-layer training signals. | Claim unmeasured ablation or Fisher evidence. |


## Shared structural utilities

- `scripts/common/depth_reduction.py` owns physical decoder removal, original-layer lineage, and cache-index reindexing shared with Phase 4B.
- `scripts/common/incremental_results.py` owns append-and-fsync JSONL storage, stable-ID deduplication, deterministic final ordering, and resume lookup.
- `scripts/common/experiment_utils.py` owns atomic JSON/text/Markdown replacement, orphaned temp recovery, and truncated-final-line JSONL recovery.
- `scripts/common/materialize_model.py` merges an optional adapter into a reloadable full checkpoint without modifying its source.
- `scripts/common/measure_model_footprint.py` measures parameter/tensor/serialized weight size and explicitly excludes KV cache and temporary runtime buffers.


Phase 2C has a separate responsibility table at `scripts/phase_2c_targeted_recovery/FILE_RESPONSIBILITIES.md`.
