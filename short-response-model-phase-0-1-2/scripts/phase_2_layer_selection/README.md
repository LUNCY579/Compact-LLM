# Phase 2 — Layer Selection Study

This folder now implements the blueprint's complete **mandatory static Phase 2** workflow and an explicit optional dynamic-selection plan.

## Before running

1. Phase 1 base and all-layer adapter evaluations must exist.
2. `reports/PHASE_1_STOP_GATE.md` must contain `Human decision: PASS`.
3. Do not include or inspect the sealed split during Phase 2 tuning.

## Main command

```powershell
.\scripts\phase_2_layer_selection\run_phase_2.ps1
```

## What the runner does

1. Runs a coarse early/middle/late single-layer scan.
2. Refines neighboring layers around the strongest coarse regions.
3. Computes the project contribution metric against the Phase 1 base/all-layer references.
4. Trains adjacent/complementary pair candidates.
5. Trains TOP4, RANDOM4, and BOTTOM4 controls.
6. Repeats the strongest candidate with seeds 11, 23, and 37.
7. Records quality, category scores, semantic hard-failure flags, training VRAM, runtime, and target-module count.
8. Builds `reports/PHASE_2_HUMAN_REVIEW_PACKET.md` with prioritized risky and low-scoring outputs.
9. Produces `reports/PHASE_2_STOP_GATE.md`; candidates with critical semantic failures are ineligible.
10. Creates an optional IST-style dynamic-selection experiment plan.

## Files

- `run_static_layer_study.py` — complete coarse-to-fine scan, static controls, contribution metric, and repeated seeds.
- `phase2_common.py` — shared subprocess, result, and metric helpers.
- `collect_importance_signals.py` — records direct single-layer scores, contribution, VRAM, and runtime; explicitly labels unmeasured signals.
- `check_phase_2_gate.py` — enforces quality, category, completion, length, resource, semantic-safety, and seed-stability checks.
- `build_phase_2_human_review.py` — creates a focused human-review packet from finalist and control outputs.
- `plan_dynamic_selection.py` — creates the optional dynamic IST-style comparison contract without making unsupported memory claims.
- `run_phase_2.ps1` — ordered Windows runner.
- `scan_layers.py` and `run_layer_scan.ps1` — retained legacy coarse-scan entry points; do not use them as the complete Phase 2 workflow.

## Stop Gate 2

A one-, two-, or four-layer configuration must match the all-layer reference within the configured quality tolerance, pass category floors, and materially reduce training memory or time. The strongest configuration is repeated across at least three seeds before human approval.

Dynamic selection is **not required** to pass Stop Gate 2. The blueprint says static first and dynamic later. A dynamic experiment is valid only after a stable ranking exists and must report optimizer-state memory separately.


## Semantic evaluation update

`phase_1_dense_sft/evaluate_model.py` now preserves the original pilot heuristic and adds a separate conservative semantic audit. It flags generated placeholders, failed required clarifications, unsupported certainty, emotional contradiction, canned assistant phrasing, and narrowly defined critical gas/medication safety failures. These flags are recorded per case and aggregated into every Phase 2 result.

The heuristic remains useful for broad scanning, but a candidate with a critical semantic failure cannot pass Stop Gate 2.
